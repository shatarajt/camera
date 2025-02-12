import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2


class IntelSubscriber(Node):
    def __init__(self):
        super().__init__("intel_subscriber")
        
        # Create subscribers for RGB, Depth, and Object Detection topics
        self.subscription_rgb = self.create_subscription(
            Image, "rgb_frame", self.rgb_frame_callback, 10
        )
        self.subscription_depth = self.create_subscription(
            Image, "depth_frame", self.depth_frame_callback, 10
        )
        self.subscription_detections = self.create_subscription(
            Detection2DArray, "detections", self.detection_callback, 10
        )

        # Initialize CvBridge to convert ROS images to OpenCV images
        self.br_rgb = CvBridge()
        self.br_depth = CvBridge()

        # Variables to hold frames and detections
        self.current_rgb_frame = None
        self.current_depth_frame = None
        self.current_detections = []

    def rgb_frame_callback(self, data):
        """ Callback to handle RGB frame data """
        self.get_logger().info("Receiving RGB frame")
        self.current_rgb_frame = self.br_rgb.imgmsg_to_cv2(data)
        if self.current_rgb_frame is not None:
            cv2.imshow("RGB Frame", self.current_rgb_frame)
            cv2.waitKey(1)

    def depth_frame_callback(self, data):
        """ Callback to handle Depth frame data """
        self.get_logger().info("Receiving Depth frame")
        self.current_depth_frame = self.br_depth.imgmsg_to_cv2(data, "bgr8")  # Convert depth to color for visualization
        if self.current_depth_frame is not None:
            cv2.imshow("Depth Frame", self.current_depth_frame)
            cv2.waitKey(1)

    def detection_callback(self, detections_msg):
        """ Callback to handle object detection data """
        self.get_logger().info("Receiving Object Detection data")
        self.current_detections = detections_msg.detections

        # If we have both RGB and detections, overlay the detections on the RGB frame
        if self.current_rgb_frame is not None and len(self.current_detections) > 0:
            for detection in self.current_detections:
                # Draw bounding box and center of the detected object
                x = int(detection.bbox.center.position.x)
                y = int(detection.bbox.center.position.y)
                w = int(detection.bbox.size_x)
                h = int(detection.bbox.size_y)

                # Draw rectangle and text with detected object info
                cv2.rectangle(self.current_rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.current_rgb_frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the RGB image with the bounding boxes
            cv2.imshow("RGB with Object Detection", self.current_rgb_frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    intel_subscriber = IntelSubscriber()
    rclpy.spin(intel_subscriber)
    intel_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
