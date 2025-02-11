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
        super().__init__('CAMERAS')

        self.bridge = CvBridge()
        self.ctx = rs.context()
        self.devices = self.ctx.query_devices()
        self.num_cameras = len(self.devices)

        if self.num_cameras == 0:
            raise Exception("Silly goose plug in a camera.")

        self.get_logger().info(f"Detected {self.num_cameras} RealSense cameras.")

        # Limit to 3 cameras
        self.num_cameras = min(self.num_cameras, 3)

        # Store multiple camera pipelines and publishers
        self.pipelines = []
        self.color_publishers = []
        self.depth_publishers = []
        self.detection_publishers = []
        self.threads = []
        self.running = True

        # Initialize each camera
        for i in range(self.num_cameras):
            serial = self.devices[i].get_info(rs.camera_info.serial_number)
            self.get_logger().info(f"Initializing Camera {i + 1} (Serial: {serial})")

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            pipeline.start(config)

            self.pipelines.append(pipeline)
            self.color_publishers.append(self.create_publisher(Image, f'camera{i + 1}/color/image_raw', 10))
            self.depth_publishers.append(self.create_publisher(Image, f'camera{i + 1}/depth/image_raw', 10))
            self.detection_publishers.append(self.create_publisher(Detection2DArray, f'camera{i + 1}/detections', 10))

            # Start a thread for each camera
            thread = threading.Thread(target=self.process_frames, args=(i,))
            thread.start()
            self.threads.append(thread)

    def process_frames(self, camera_index):
        """ Continuously processes RealSense frames for a specific camera. """
        pipeline = self.pipelines[camera_index]

        while self.running:
            frames = pipeline.wait_for_frames()
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
            self.color_publishers[camera_index].publish(color_msg)
            self.depth_publishers[camera_index].publish(depth_msg)

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

                print(f"Camera {camera_index + 1}: Object detected at ({center_x}, {center_y}) -> Distance: {distance:.2f} meters")

                # Draw bounding box
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(color_image, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # ROS2 Detection Message
                detection = Detection2D()
                detection.bbox.center.position.x = float(center_x)
                detection.bbox.center.position.y = float(center_y)
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)
                detections.detections.append(detection)

            # Publish detected objects
            self.detection_publishers[camera_index].publish(detections)

            # Display images
            combined_image = np.hstack((color_image, depth_colormap))
            cv2.imshow(f"Camera {camera_index + 1} - Object Detection", combined_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def cleanup(self):
        #cleans up resources. 
        self.running = False
        for thread in self.threads:
            thread.join()
        for pipeline in self.pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()
        self.get_logger().info("Stopped all cameras.")

def main(args=None):
    rclpy.init(args=args)
    node = MultiRealSenseObjectDetection()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down MultiRealSense node.")

    node.cleanup()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
