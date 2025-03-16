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
        super().__init__("cam_publisher")

        self.br = CvBridge()
        self.timer_period = 1.0 / 30  # 30 FPS

        self.context = rs.context()
        self.device_list = self.context.devices
        self.cameras = []
        self.publishers = []

        if len(self.device_list) == 0:
            self.get_logger().error("No RealSense cameras detected over Ethernet!")
            return

        for i, device in enumerate(self.device_list):
            try:
                serial_number = device.get_info(rs.camera_info.serial_number)
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial_number)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                
                # Handle potential Ethernet latency
                config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

                pipeline.start(config)

                self.cameras.append({
                    "serial": serial_number,
                    "pipeline": pipeline
                })

                self.publishers.append({
                    "rgb": self.create_publisher(Image, f"rgb_frame{i+1}", 10),
                    "depth": self.create_publisher(Image, f"depth_frame{i+1}", 10),
                    "detections": self.create_publisher(Detection2DArray, f"detections{i+1}", 10)
                })

                self.get_logger().info(f"RealSense Camera {i+1} (S/N: {serial_number}) is connected.")

            except Exception as e:
                self.get_logger().error(f"Failed to initialize RealSense Camera {i+1}: {str(e)}")

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        for i, cam in enumerate(self.cameras):
            try:
                pipeline = cam["pipeline"]
                frames = pipeline.wait_for_frames()

                # Handle possible Ethernet delays
                if not frames:
                    self.get_logger().warning(f"No frames received for Camera {i+1}. Retrying...")
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    self.get_logger().warning(f"Skipping frame for Camera {i+1} due to empty frame.")
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Publish RGB Image
                self.publishers[i]["rgb"].publish(self.br.cv2_to_imgmsg(color_image, encoding="bgr8"))

                # Publish Depth Image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                self.publishers[i]["depth"].publish(self.br.cv2_to_imgmsg(depth_colormap, encoding='bgr8'))

                # Object Detection
                detections = Detection2DArray()
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                _, threshold_img = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 1000:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w // 2, y + h // 2
                    distance = depth_frame.get_distance(center_x, center_y)

                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"{distance:.2f} m", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detection = Detection2D()
                    detection.bbox.center.position.x = float(center_x)
                    detection.bbox.center.position.y = float(center_y)
                    detection.bbox.size_x = float(w)
                    detection.bbox.size_y = float(h)
                    detections.detections.append(detection)

                self.publishers[i]["detections"].publish(detections)

                # Display images
                combined_image = np.hstack((color_image, depth_colormap))
                cv2.imshow(f"RGB and Depth Frame {i+1} (S/N: {cam['serial']})", combined_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info("Shutting down...")
                    rclpy.shutdown()

            except Exception as e:
                self.get_logger().error(f"Error in frame acquisition for Camera {i+1}: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    cam_publisher = CamPublisher()
    rclpy.spin(cam_publisher)
    cam_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

