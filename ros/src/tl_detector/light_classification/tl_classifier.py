from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2 
import rospy
import time



#SSD_GRAPH_FILE_SIM = 'light_classification/models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
SSD_GRAPH_FILE_SIM = 'light_classification/model/frozen_inference_graph.pb'

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.graph = self.load_graph(SSD_GRAPH_FILE_SIM)
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')



    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes



    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return graph
        

    def get_classification(self, image):
        
	"""Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8),0)

        with tf.Session(graph=self.graph) as sess:
            # Actual detection
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.8

            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        
        if len(classes) > 0:
            color_state = int(classes[np.argmax(scores)])
            #width, height = image.size
            #box_coords = to_image_coords(boxes, height, width)

            # Each class with be represented by a differently colored box
            #draw_boxes(image, box_coords, classes)


            if color_state == 1:
                rospy.loginfo("traffic light state: GREEN")
                return TrafficLight.GREEN 
            elif color_state == 2:
                rospy.loginfo("traffic light state: RED")
                return TrafficLight.RED 
            elif color_state == 3:
                rospy.loginfo("traffic light state: YELLOW")
                return TrafficLight.YELLOW 
	else:
        
            rospy.logerr("No model to predict traffic light state")

        
        return TrafficLight.UNKNOWN

