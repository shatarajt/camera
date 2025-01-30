import pyrealsense2 as rs
import numpy as np
import cv2
import threading

# Configure depth and color streams
ctx = rs.context()
devices = ctx.query_devices()
num_cameras = len(devices)

if num_cameras == 0:
    raise Exception("At least one camera is required to run this code.")

# Initialize pipelines and configurations
pipelines = [rs.pipeline() for _ in range(num_cameras)]
configs = [rs.config() for _ in range(num_cameras)]

for i, device in enumerate(devices):
    configs[i].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # Reduce frame rate to 15 FPS
    configs[i].enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    configs[i].enable_device(device.get_info(rs.camera_info.serial_number))

# Global variables for threading
running = True
frames = [None] * num_cameras
frame_available = [False] * num_cameras

def frame_capture(pipeline, frames_var, frame_flag_var, index):
    """Thread function to retrieve frames from a RealSense camera."""
    global running
    while running:
        try:
            frames_var[index] = pipeline.wait_for_frames(timeout_ms=5000)
            frame_flag_var[index] = True
        except RuntimeError:
            print(f"Camera {index + 1}: Frame timeout, retrying...")
            frame_flag_var[index] = False

try:
    # Start all pipelines
    for pipeline, config in zip(pipelines, configs):
        pipeline.start(config)

    # Initialize frame retrieval threads
    threads = []
    for i in range(num_cameras):
        thread = threading.Thread(target=frame_capture, args=(pipelines[i], frames, frame_available, i))
        thread.start()
        threads.append(thread)

    while running:
        for i, (pipeline, frame_flag, label) in enumerate(zip(pipelines, frame_available, [f"Camera {j + 1}" for j in range(num_cameras)])):
            if frame_flag:
                depth_frame = frames[i].get_depth_frame()
                color_frame = frames[i].get_color_frame()

                if depth_frame and color_frame:
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    # Convert depth image to colormap for visualization
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                    )

                    # Preprocess color image for contour detection
                    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)  # Increase Gaussian kernel size
                    _, threshold_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)  # Increase threshold value

                    # Find contours in the threshold image
                    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    objects = []  # List to store detected objects

                    # Process each contour
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < 1000:  # Increase minimum object area to filter noise
                            continue

                        # Get bounding box and center of the contour
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x, center_y = x + w // 2, y + h // 2

                        # Get depth value at the center of the bounding box
                        distance = depth_frame.get_distance(center_x, center_y)

                        # Store object details
                        objects.append((x, y, w, h, distance))

                    # Process and display each detected object
                    for obj in objects:
                        x, y, w, h, distance = obj

                        # Draw bounding box and display distance
                        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(color_image, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Print the distance of the detected object
                        print(f"{label}: Object at ({x}, {y}, {w}, {h}) -> Distance: {distance:.2f} meters")

                    # Combine depth and color images for display
                    if depth_image.shape != color_image.shape[:2]:
                        resized_color_image = cv2.resize(color_image, dsize=(depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA)
                        images = np.hstack((resized_color_image, depth_colormap))
                    else:
                        images = np.hstack((color_image, depth_colormap))

                    # Display the images
                    cv2.namedWindow(label, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(label, images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Stop everything
    running = False
    for thread in threads:
        thread.join()
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipelines stopped, resources released.")
