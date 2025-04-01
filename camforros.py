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
        self.intel_publisher_rgb_1 = self.create_publisher(Image, "camera_1/rgb_frame", 10)
        self.intel_publisher_depth_1 = self.create_publisher(Image, "camera_1/depth_frame", 10)
        self.intel_publisher_detections_1 = self.create_publisher(Detection2DArray, "camera_1/detections", 10)
        
        self.intel_publisher_rgb_2 = self.create_publisher(Image, "camera_2/rgb_frame", 10)
        self.intel_publisher_depth_2 = self.create_publisher(Image, "camera_2/depth_frame", 10)
        self.intel_publisher_detections_2 = self.create_publisher(Detection2DArray, "camera_2/detections", 10)
        
        self.intel_publisher_rgb_3 = self.create_publisher(Image, "camera_3/rgb_frame", 10)
        self.intel_publisher_depth_3 = self.create_publisher(Image, "camera_3/depth_frame", 10)
        self.intel_publisher_detections_3 = self.create_publisher(Detection2DArray, "camera_3/detections", 10)

        # Timer period to control frame rate
        timer_period = 1.0 / 30  # 30 FPS
        self.br_rgb = CvBridge()

        try:
            # Initialize RealSense pipeline for three cameras
            self.pipe_1 = rs.pipeline()
            self.cfg_1 = rs.config()
            self.cfg_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.cfg_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.pipe_1.start(self.cfg_1)
            
            self.pipe_2 = rs.pipeline()
            self.cfg_2 = rs.config()
            self.cfg_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.cfg_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.pipe_2.start(self.cfg_2)
            
            self.pipe_3 = rs.pipeline()
            self.cfg_3 = rs.config()
            self.cfg_3.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.cfg_3.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.pipe_3.start(self.cfg_3)

            self.get_logger().info("RealSense cameras connected and streaming.")
            self.timer = self.create_timer(timer_period, self.timer_callback)
        except Exception as e:
            self.get_logger().error(f"Error with RealSense cameras: {str(e)}")

    def timer_callback(self):
        try:
            # Capture frames from each camera
            frames_1 = self.pipe_1.wait_for_frames()
            frames_2 = self.pipe_2.wait_for_frames()
            frames_3 = self.pipe_3.wait_for_frames()

            # Get color and depth frames for camera 1
            color_frame_1 = frames_1.get_color_frame()
            depth_frame_1 = frames_1.get_depth_frame()
            color_image_1 = np.asanyarray(color_frame_1.get_data())
            depth_image_1 = np.asanyarray(depth_frame_1.get_data())

            # Get color and depth frames for camera 2
            color_frame_2 = frames_2.get_color_frame()
            depth_frame_2 = frames_2.get_depth_frame()
            color_image_2 = np.asanyarray(color_frame_2.get_data())
            depth_image_2 = np.asanyarray(depth_frame_2.get_data())

            # Get color and depth frames for camera 3
            color_frame_3 = frames_3.get_color_frame()
            depth_frame_3 = frames_3.get_depth_frame()
            color_image_3 = np.asanyarray(color_frame_3.get_data())
            depth_image_3 = np.asanyarray(depth_frame_3.get_data())

            # Publish RGB and Depth images for each camera
            self.intel_publisher_rgb_1.publish(self.br_rgb.cv2_to_imgmsg(color_image_1))
            self.intel_publisher_depth_1.publish(self.br_rgb.cv2_to_imgmsg(cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.03), cv2.COLORMAP_JET), encoding='bgr8'))
            
            self.intel_publisher_rgb_2.publish(self.br_rgb.cv2_to_imgmsg(color_image_2))
            self.intel_publisher_depth_2.publish(self.br_rgb.cv2_to_imgmsg(cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=0.03), cv2.COLORMAP_JET), encoding='bgr8'))
            
            self.intel_publisher_rgb_3.publish(self.br_rgb.cv2_to_imgmsg(color_image_3))
            self.intel_publisher_depth_3.publish(self.br_rgb.cv2_to_imgmsg(cv2.applyColorMap(cv2.convertScaleAbs(depth_image_3, alpha=0.03), cv2.COLORMAP_JET), encoding='bgr8'))

            # Object detection and closest object for each camera
            self.detect_objects_and_publish(color_image_1, depth_frame_1, self.intel_publisher_detections_1)
            self.detect_objects_and_publish(color_image_2, depth_frame_2, self.intel_publisher_detections_2)
            self.detect_objects_and_publish(color_image_3, depth_frame_3, self.intel_publisher_detections_3)

            # Optional: Press 'q' to stop the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutting down.")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in frame acquisition: {str(e)}")

    def detect_objects_and_publish(self, color_image, depth_frame, publisher):
        detections = Detection2DArray()

        # Convert color image to grayscale and apply GaussianBlur
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply threshold to detect edges
        _, threshold_img = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

        # Find contours in the threshold image
        contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        closest_distance = float('inf')
        closest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter out small contours
                continue

            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2

            # Get the distance of the object at the center of the bounding box
            distance = depth_frame.get_distance(center_x, center_y)

            if distance < closest_distance:
                closest_distance = distance
                closest_contour = contour

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
        publisher.publish(detections)


def main(args=None):
    rclpy.init(args=args)
    intel_publisher = CamPublisher()
    rclpy.spin(intel_publisher)
    intel_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
