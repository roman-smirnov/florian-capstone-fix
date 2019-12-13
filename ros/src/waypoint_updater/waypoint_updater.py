#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import math
import numpy as np
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number

class MotionState():
    Drive, Stop = range(2)

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1, buff_size=2**17)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1, buff_size=2**17)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1, buff_size=2**17)

        # TODO: Add a subscriber for /obstacle_waypoint
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1, buff_size=2**17)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.vehicle = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.nearest_light = None
        self.vehicle_velocity = None # in m/s

        self.motion_state = MotionState.Drive
        self.deceleration_rate = None
        self.acceleration_rate = 0.75 # m/s

        self.previous_velocity = None

        self.loop()

    def pose_cb(self, msg):
        self.vehicle = msg

    # Extracts the x and y coordinates of a waypoint
    def waypoint_xy(self, waypoint):
        position = waypoint.pose.pose.position
        return [position.x, position.y]

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [self.waypoint_xy(waypoint) for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        if (msg.data == -1):
            self.nearest_light = None
        else:
            self.nearest_light = msg.data

    def velocity_cb(self, velocity):
        self.vehicle_velocity = velocity.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    # This is our main loop which is run at a set interval
    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.vehicle and self.waypoints_tree and self.vehicle_velocity:
                # If you turn around around in the simulator all the waypoint can end
                # up behind the vehicle which will cause nearest_forward_waypoint to
                # return None
                start_index = self.nearest_forward_waypoint()
                if start_index is not None:
                    end_index = start_index + LOOKAHEAD_WPS
                    lane_waypoints = self.base_waypoints[start_index:end_index]
                    if self.nearest_light != None and self.nearest_light >= start_index and self.nearest_light <= end_index:
                        self.motion_state = MotionState.Stop
                        lane_waypoints = self.decelerate(lane_waypoints, start_index)
                    elif self.motion_state == MotionState.Stop:
                        # We arrive were previously decelerating/stopped but are
                        # starting to accelerate/drive again
                        self.motion_state = MotionState.Drive
                        self.deceleration_rate = None

                    if self.motion_state == MotionState.Drive:
                        if abs(self.vehicle_velocity - self.get_waypoint_velocity(lane_waypoints[0])) > 1.0:
                            if self.previous_velocity == None:
                                start_velocity = self.vehicle_velocity
                            else:
                                start_velocity = max(self.previous_velocity+0.2, self.vehicle_velocity)
                            lane_waypoints = self.accelerate(lane_waypoints, start_velocity)
                        else:
                            self.acceleration_start_velocity = None

                    lane = Lane()
                    lane.waypoints = lane_waypoints

                    self.final_waypoints_pub.publish(lane)
            self.previous_velocity = self.vehicle_velocity
            rate.sleep()

    def accelerate(self, waypoints, start_velocity):
        processed_waypoints = []
        for i, waypoint in enumerate(waypoints):
            p = Waypoint()
            p.pose = waypoint.pose

            distance = self.distance(waypoints, 0, i)
            target_speed = start_velocity + distance * self.acceleration_rate
            if target_speed < 0.5:
                target_speed = 0.5

            target_speed = min(target_speed, self.get_waypoint_velocity(waypoint))
            p.twist.twist.linear.x = target_speed
            processed_waypoints.append(p)
        return processed_waypoints

    def decelerate(self, waypoints, start_index):
        stop_index = self.nearest_light - start_index - 2 # So that car doesn't stop on line
        processed_waypoints = []
        speed = self.vehicle_velocity
        for i, waypoint in enumerate(waypoints):
            p = Waypoint()
            p.pose = waypoint.pose

            distance = self.distance(waypoints, i, stop_index)
            if i >= stop_index:
                target_speed = 0
            elif distance < 15:
                if self.deceleration_rate == None:
                    self.deceleration_rate = self.vehicle_velocity / distance
                target_speed = self.deceleration_rate * distance
                if target_speed <= 1:
                    target_speed = 0
                target_speed = min(target_speed, self.get_waypoint_velocity(waypoint))
            else:
                target_speed = self.get_waypoint_velocity(waypoint)
            p.twist.twist.linear.x = target_speed
            processed_waypoints.append(p)
        return processed_waypoints

    # Returns the index of the nearest waypoint ahead of the vehicle
    def nearest_forward_waypoint(self):
        vehicle = [self.vehicle.pose.position.x, self.vehicle.pose.position.y]
        closest_index = self.waypoints_tree.query(vehicle, 1)[1]

        closest_waypoint = np.array(self.waypoints_2d[closest_index])
        previous_waypoint = np.array(self.waypoints_2d[closest_index - 1])
        vehicle = np.array(vehicle)

        waypoint_vector = closest_waypoint - previous_waypoint
        vehicle_vector = vehicle - closest_waypoint

        val = np.dot(waypoint_vector, vehicle_vector)

        if val > 0:
            closest_index = (closest_index + 1) % len(self.waypoints_2d)

        return closest_index

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
