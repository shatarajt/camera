import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs
import numpy as np

class CamPublisher(Node):
    def __init__(self):
        super().__init__("intel_publisher")
        
        # Publishers
        self.intel_publisher_rgb = self.create_publisher(Image, "rgb_frame", 10)
        self.intel_publisher_depth = self.create_publisher(Image, "depth_frame", 10)
        self.intel_publisher_detections = self.create_publisher(Detection2DArray, "detections", 10)

        # Timer period to control frame rate
        timer_period = 1.0 / 30  # 30 FPS
        self.br_rgb = CvBridge()

        try:
            # Initialize RealSense pipeline
            self.pipe = rs.pipeline()
            self.cfg = rs.config()
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipe.start(self.cfg)
            self.get_logger().info("RealSense camera connected and streaming.")
            self.timer = self.create_timer(timer_period, self.timer_callback)
        except Exception as e:
            self.get_logger().error(f"INTEL REALSENSE IS NOT CONNECTED: {str(e)}")

    def timer_callback(self):
        try:
            frames = self.pipe.wait_for_frames()

            # Get color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # Convert RealSense frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Publish RGB image
            self.intel_publisher_rgb.publish(self.br_rgb.cv2_to_imgmsg(color_image))

            # Convert depth to a colormap for better visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            self.intel_publisher_depth.publish(self.br_rgb.cv2_to_imgmsg(depth_colormap, encoding='bgr8'))

            # Perform object detection (simple contour detection)
            detections = Detection2DArray()

            # Convert color image to grayscale and apply GaussianBlur
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Apply threshold to detect edges
            _, threshold_img = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

            # Find contours in the threshold image
            contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Filter out small contours
                    continue

                # Get bounding box of contour
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2

                # Get the distance of the object at the center of the bounding box
                distance = depth_frame.get_distance(center_x, center_y)

                # Draw bounding box and display distance
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(color_image, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Create ROS2 detection message
                detection = Detection2D()
                detection.bbox.center.position.x = float(center_x)
                detection.bbox.center.position.y = float(center_y)
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)
                detections.detections.append(detection)

            # Publish object detection information
            self.intel_publisher_detections.publish(detections)

            # Display combined image (RGB + depth)
            combined_image = np.hstack((color_image, depth_colormap))
            cv2.imshow("RGB and Depth Frame with Object Detection", combined_image)

            # Optional: Press 'q' to stop the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutting down.")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in frame acquisition: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    intel_publisher = CamPublisher()
    rclpy.spin(intel_publisher)
    intel_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
