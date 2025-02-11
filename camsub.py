import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2

class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__("realsense_subscriber")
        
        # Create subscriptions for the color and depth image topics
        self.subscription_color = self.create_subscription(
            Image, 'camera/color/image_raw', self.color_frame_callback, 10
        )
        self.subscription_depth = self.create_subscription(
            Image, 'camera/depth/image_raw', self.depth_frame_callback, 10
        )
        self.subscription_detections = self.create_subscription(
            Detection2DArray, 'camera/detections', self.detection_callback, 10
        )
        
        # Create CvBridge to convert ROS messages to OpenCV
        self.br = CvBridge()
        
        # Initialize frame storage
        self.color_frame = None
        self.depth_frame = None
        self.detections = None

    def color_frame_callback(self, data):
        #Callback for receiving color frame.
        self.get_logger().info("Receiving color frame")
        self.color_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

    def depth_frame_callback(self, data):
        #Callback for receiving depth frame. 
        self.get_logger().info("Receiving depth frame")
        self.depth_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

    def detection_callback(self, data):
        # Callback for receiving object detection data. 
        self.get_logger().info(f"Received {len(data.detections)} detections")
        self.detections = data.detections

    def display_frames(self):
        #Display color, depth and detection 
        if self.color_frame is not None:
            cv2.imshow("Color Frame", self.color_frame)
        
        if self.depth_frame is not None:
            cv2.imshow("Depth Frame", self.depth_frame)

        if self.detections:
            for detection in self.detections:
                # bounding box and coordinates
                x = int(detection.bbox.center.position.x)
                y = int(detection.bbox.center.position.y)
                w = int(detection.bbox.size_x)
                h = int(detection.bbox.size_y)

                # Draw bounding box
                if self.color_frame is not None:
                    cv2.rectangle(self.color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(self.color_frame, f"Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.color_frame is not None:
                cv2.imshow("Detected Objects", self.color_frame)

    def spin(self):
        """ Periodically updates and shows frames. """
        rate = self.create_rate(30)  # 30 Hz rate for displaying frames
        while rclpy.ok():
            self.display_frames()
            rclpy.spin_once(self)
            rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    subscriber_node = RealSenseSubscriber()
    
    # Start spinning the node and displaying frames
    subscriber_node.spin()
    
    subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
