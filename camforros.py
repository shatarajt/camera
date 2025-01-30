import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2
import threading

class RealSenseObjectDetection(Node):
    def __init__(self):
        super().__init__('realsense_object_detection')

        # ROS2 Publishers
        self.bridge = CvBridge()
        self.color_publisher = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.depth_publisher = self.create_publisher(Image, 'camera/depth/image_raw', 10)
        self.detection_publisher = self.create_publisher(Detection2DArray, 'camera/detections', 10)

        # Initialize RealSense camera
        self.ctx = rs.context()
        self.devices = self.ctx.query_devices()
        self.num_cameras = len(self.devices)

        if self.num_cameras == 0:
            raise Exception("No RealSense camera detected!")

        # Configure RealSense streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

        # Start camera stream
        self.pipeline.start(self.config)
        self.running = True

        # Start object detection thread
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.start()

    def process_frames(self):
        """ Continuously processes RealSense frames and publishes data. """
        while self.running:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert frames to OpenCV format
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert depth to colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Convert OpenCV images to ROS messages
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding='bgr8')

            # Publish images
            self.color_publisher.publish(color_msg)
            self.depth_publisher.publish(depth_msg)

            # Object Detection Processing
            detections = Detection2DArray()

            # Convert to grayscale and apply threshold
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, threshold_img = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Filter small objects
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2
                distance = depth_frame.get_distance(center_x, center_y)

                # Draw bounding box
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(color_image, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # ROS2 Detection Message
                detection = Detection2D()
                detection.bbox.center.position.x = center_x
                detection.bbox.center.position.y = center_y
                detection.bbox.size_x = w
                detection.bbox.size_y = h
                detections.detections.append(detection)

            # Publish detected objects
            self.detection_publisher.publish(detections)

            # Display images
            combined_image = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense Object Detection", combined_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def cleanup(self):
        """ Stops camera and cleans up resources. """
        self.running = False
        self.thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.get_logger().info("Stopped RealSense camera.")

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseObjectDetection()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RealSense node.")

    node.cleanup()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
