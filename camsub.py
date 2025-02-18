import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2


class MultiCameraSubscriber(Node):
    def __init__(self):
        super().__init__("multi_camera_subscriber")
        
        # Create subscribers for multiple cameras (RGB, Depth, and an extra camera)
        self.subscription_rgb1 = self.create_subscription(
            Image, "rgb_frame1", self.rgb_frame_callback1, 10
        )
        self.subscription_depth1 = self.create_subscription(
            Image, "depth_frame1", self.depth_frame_callback1, 10
        )
        
        self.subscription_rgb2 = self.create_subscription(
            Image, "rgb_frame2", self.rgb_frame_callback2, 10
        )
        self.subscription_depth2 = self.create_subscription(
            Image, "depth_frame2", self.depth_frame_callback2, 10
        )
        
        self.subscription_rgb3 = self.create_subscription(
            Image, "rgb_frame3", self.rgb_frame_callback3, 10
        )
        self.subscription_depth3 = self.create_subscription(
            Image, "depth_frame3", self.depth_frame_callback3, 10
        )
        
        self.subscription_detections = self.create_subscription(
            Detection2DArray, "detections", self.detection_callback, 10
        )

        # Initialize CvBridge to convert ROS images to OpenCV images
        self.br = CvBridge()

        # Variables to hold frames and detections
        self.current_rgb_frames = [None, None, None]
        self.current_depth_frames = [None, None, None]
        self.current_detections = []

    def rgb_frame_callback1(self, data):
        self.current_rgb_frames[0] = self.br.imgmsg_to_cv2(data, "bgr8")
        self.show_frame(self.current_rgb_frames[0], "RGB Camera 1")
    
    def rgb_frame_callback2(self, data):
        self.current_rgb_frames[1] = self.br.imgmsg_to_cv2(data, "bgr8")
        self.show_frame(self.current_rgb_frames[1], "RGB Camera 2")
    
    def rgb_frame_callback3(self, data):
        self.current_rgb_frames[2] = self.br.imgmsg_to_cv2(data, "bgr8")
        self.show_frame(self.current_rgb_frames[2], "RGB Camera 3")
    
    def depth_frame_callback1(self, data):
        self.current_depth_frames[0] = self.br.imgmsg_to_cv2(data, "bgr8")
        self.show_frame(self.current_depth_frames[0], "Depth Camera 1")
    
    def depth_frame_callback2(self, data):
        self.current_depth_frames[1] = self.br.imgmsg_to_cv2(data, "bgr8")
        self.show_frame(self.current_depth_frames[1], "Depth Camera 2")
    
    def depth_frame_callback3(self, data):
        self.current_depth_frames[2] = self.br.imgmsg_to_cv2(data, "bgr8")
        self.show_frame(self.current_depth_frames[2], "Depth Camera 3")
    
    def detection_callback(self, detections_msg):
        self.get_logger().info("Receiving Object Detection data")
        self.current_detections = detections_msg.detections
        
        for i, frame in enumerate(self.current_rgb_frames):
            if frame is not None:
                frame_with_detections = frame.copy()
                for detection in self.current_detections:
                    x = int(detection.bbox.center.position.x)
                    y = int(detection.bbox.center.position.y)
                    w = int(detection.bbox.size_x)
                    h = int(detection.bbox.size_y)

                    cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_with_detections, f"{x},{y}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.show_frame(frame_with_detections, f"RGB Camera {i+1} with Detections")
    
    def show_frame(self, frame, window_name):
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    multi_camera_subscriber = MultiCameraSubscriber()
    rclpy.spin(multi_camera_subscriber)
    multi_camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


