import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__("multi_camera_subscriber")
        
        self.bridge = CvBridge()
        
        #3 cameras
        self.camera_frames = {f"camera{i}": None for i in range(1, 4)}
        self.depth_frames = {f"camera{i}": None for i in range(1, 4)}
        self.detections = {f"camera{i}": [] for i in range(1, 4)}
        
        # Subscribe to multiple cameras (color, depth, and detection topics)
        for i in range(1, 4):
            cam_topic = f"camera{i}/color/image_raw"
            depth_topic = f"camera{i}/depth/image_raw"
            detect_topic = f"camera{i}/detections"

            self.create_subscription(Image, cam_topic, partial(self.color_frame_callback, camera_id=f"camera{i}"), 10)
            self.create_subscription(Image, depth_topic, partial(self.depth_frame_callback, camera_id=f"camera{i}"), 10)
            self.create_subscription(Detection2DArray, detect_topic, partial(self.detection_callback, camera_id=f"camera{i}"), 10)

    def color_frame_callback(self, data, camera_topic):
        """Receives color frames from a specific camera."""
        camera_id = camera_topic.split('/')[0]  # Extract camera1, camera2, and camera3.
        self.camera_frames[camera_id] = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

    def depth_frame_callback(self, data, depth_topic):
        """Receives depth frames from a specific camera."""
        camera_id = depth_topic.split('/')[0]
        self.depth_frames[camera_id] = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    def detection_callback(self, data, detect_topic):
        """Receives object detection data from a specific camera."""
        camera_id = detect_topic.split('/')[0]
        self.detections[camera_id] = data.detections

    def display_frames(self):
        """Display color and depth frames for all active cameras."""
        for camera_id in self.camera_frames.keys():
            color_frame = self.camera_frames[camera_id]
            depth_frame = self.depth_frames[camera_id]

            if color_frame is not None:
                frame = color_frame.copy()

                # Draw bounding boxes for detected objects
                for detection in self.detections[camera_id]:
                    x = int(detection.bbox.center.position.x)
                    y = int(detection.bbox.center.position.y)
                    w = int(detection.bbox.size_x)
                    h = int(detection.bbox.size_y)

                    # Draw bounding box on color frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow(f"{camera_id} - Color Frame", frame)

            if depth_frame is not None:
                cv2.imshow(f"{camera_id} - Depth Frame", depth_frame)

        cv2.waitKey(1)

    def spin(self):
        """Continuously process incoming frames and display them."""
        while rclpy.ok():
            rclpy.spin_once(self)
            self.display_frames()

def main(args=None):
    rclpy.init(args=args)
    subscriber = CameraSubscriber()
    subscriber.spin()
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
