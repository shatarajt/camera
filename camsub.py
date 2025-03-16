import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class CamSubscriber(Node):
    def __init__(self):
        super().__init__("camera_subscriber")

        # Create subscribers for three cameras (RGB, Depth, and Detections)
        self.subscribers = []
        self.br = CvBridge()

        self.rgb_frames = [None] * 3
        self.depth_frames = [None] * 3
        self.detections = [[] for _ in range(3)]

        for i in range(3):
            self.subscribers.append({
                "rgb": self.create_subscription(
                    Image, f"rgb_frame{i+1}", lambda msg, idx=i: self.rgb_callback(msg, idx), 10
                ),
                "depth": self.create_subscription(
                    Image, f"depth_frame{i+1}", lambda msg, idx=i: self.depth_callback(msg, idx), 10
                ),
                "detections": self.create_subscription(
                    Detection2DArray, f"detections{i+1}", lambda msg, idx=i: self.detection_callback(msg, idx), 10
                )
            })

        self.get_logger().info("Camera Subscribers Initialized")

    def rgb_callback(self, msg, idx):
        """Callback for RGB frames"""
        try:
            self.rgb_frames[idx] = self.br.imgmsg_to_cv2(msg, "bgr8")
            self.display_frame(idx)
        except Exception as e:
            self.get_logger().error(f"Error processing RGB frame {idx+1}: {str(e)}")

    def depth_callback(self, msg, idx):
        """Callback for Depth frames"""
        try:
            depth_image = self.br.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            self.depth_frames[idx] = depth_colormap
            self.display_frame(idx)
        except Exception as e:
            self.get_logger().error(f"Error processing Depth frame {idx+1}: {str(e)}")

    def detection_callback(self, msg, idx):
        """Callback for Object Detections"""
        self.detections[idx] = msg.detections
        self.display_frame(idx)

    def display_frame(self, idx):
        """Display the RGB, Depth, and Detections overlayed"""
        if self.rgb_frames[idx] is None or self.depth_frames[idx] is None:
            return  # Wait until both frames are received

        # Overlay detections on RGB frame
        frame_with_detections = self.rgb_frames[idx].copy()

        for detection in self.detections[idx]:
            x = int(detection.bbox.center.position.x)
            y = int(detection.bbox.center.position.y)
            w = int(detection.bbox.size_x)
            h = int(detection.bbox.size_y)

            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_with_detections, f"{x},{y}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Stack RGB and Depth for visualization
        combined_view = np.hstack((frame_with_detections, self.depth_frames[idx]))
        cv2.imshow(f"Camera {idx+1} - RGB & Depth", combined_view)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    cam_subscriber = CamSubscriber()
    rclpy.spin(cam_subscriber)
    cam_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


