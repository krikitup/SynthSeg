import cv2
from pypylon import pylon
import numpy as np
import grab_by_ip
import apriltag
import csv
import time
import yaml
import sys
import os

# Function to view the camera feed only (continuous)
def configure_camera(camera):
    camera.DeviceLinkThroughputLimit.SetValue(125000000)
    camera.AcquisitionFrameRate.SetValue(5)

def update_yaml(filename, tag_data):
    import yaml
    try:
        # Load existing data from the YAML file, if it exists
        with open(filename, 'r') as file:
            data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        data = {}

    # Update the YAML file with new tag data
    for tag_id, info in tag_data.items():
        data[tag_id] = info

    # Write back the updated data
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def view_feed(camera, converter, detector, calibration_yaml_path, ip_address, output_csv_path):
    """
    - Press 'c' to start capturing (writing to CSV).
    - Press 's' to stop capturing (writing to CSV).
    - Press 'q' to quit.
    
    CSV columns:
      - frame_id
      - timestamp
      - tag_id
      - pose_matrix (the entire 4×4 homogeneous transformation as a Python list of lists)
    """

    # 1) Load your calibration data
    with open(calibration_yaml_path, 'r') as file:
        calib_data = yaml.safe_load(file)

    mtx = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
    dist = np.array(calib_data['dist_coeff']['data'])
    
    frame_width = 1920
    frame_height = 1200
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height)
    )

    # Construct a filename for your YAML (optional) if you still want to use update_yaml
    yaml_filename = f"{ip_address}_april_tag.yaml"

    # This boolean will control CSV recording
    is_capturing = False
    mapx, mapy = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (frame_width, frame_height), cv2.CV_32FC1
)
# Define frames folder
    frames_folder = f"{ip_address}_frames"

    # Clear the existing folder
    if os.path.exists(frames_folder):
        for file in os.listdir(frames_folder):
            file_path = os.path.join(frames_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    os.makedirs(frames_folder, exist_ok=True)

    print(f"Cleared existing frames in {frames_folder}. Ready for new captures.")

    # 2) Open the CSV file in append mode ('a')
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Optional: Check if file is empty to write a header
        csv_file.seek(0, 2)  # Move cursor to end of file
        if csv_file.tell() == 0:
            writer.writerow(["frame_id", "timestamp", "tag_id", "pose_matrix"])

        # 3) Start grabbing frames
        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                frame_id = grabResult.BlockID
                unix_timestamp = time.time()

                # Convert to OpenCV format
                image = converter.Convert(grabResult)
                frame = image.GetArray()

                # Undistort and crop
                undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = roi
                undistorted_frame = undistorted_frame[y:y+h, x:x+w]
                frame = undistorted_frame

                # Convert to grayscale for AprilTag detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect tags
                detections = detector.detect(gray)

                # Dictionary to store data for update_yaml, if you need it
                tag_data = {}
                if is_capturing:
                #save the frame in folder named ip_address_frames
                # cv2.imwrite(f"frame_{frame_id}.png", frame)
                    os.makedirs(f"{ip_address}_frames", exist_ok=True)
                    cv2.imwrite(f"{ip_address}_frames/{frame_id}.png", frame) 

                # For each detected tag in this frame
                for detection in detections:
                    tag_id = str(detection.tag_id)
                    corners = detection.corners

                    # Compute center (optional, for display)
                    center_x = (corners[0][0] + corners[2][0]) / 2
                    center_y = (corners[0][1] + corners[2][1]) / 2
                    center = (int(center_x), int(center_y))

                    # Compute the 4×4 pose                # if is_capturing:
                # #save the frame in folder named ip_address_frames
                # # cv2.imwrite(f"frame_{frame_id}.png", frame)
                #     os.makedirs(f"{ip_address}_frames", exist_ok=True)
                #     cv2.imwrite(f"{ip_address}_frames/{frame_id}.png", frame)       matrix
                    pose_4x4, _, _ = detector.detection_pose(
                        detection,
                        camera_params=(mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]),
                        tag_size=10.375 * 0.0254  # example conversion from inches to meters
                    )
                    tag_size = 10.375 * 0.0254
                    rotation_matrix = pose_4x4[:3, :3]  # Upper-left 3×3 is the rotation matrix
                    translation_vector = pose_4x4[:3, 3].reshape(3, 1)  # Last column is the translation

                    # Convert Rotation Matrix to Rodrigues vector (rvec)
                    rvec, _ = cv2.Rodrigues(rotation_matrix)
                    tvec = translation_vector 
                    # Convert to Python list (so it's easily stored in CSV)
                    pose_matrix = pose_4x4.tolist()

                    # Draw bounding box & text for visualization
                    cv2.polylines(frame, [corners.astype(int)], isClosed=True,
                                  color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, tag_id, center, cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, tag_size*0.5)

                    # Build data for update_yaml (optional)
                    tag_data[tag_id] = {
                        'center': {'x': center[0], 'y': center[1]},
                        'pose_matrix': pose_matrix
                    }

                    # 4) Write data to CSV *only if capturing is ON*
                    if is_capturing:
                        writer.writerow([
                            frame_id,
                            unix_timestamp,
                            tag_id,
                            pose_matrix  # The entire 4×4 matrix as a list of lists
                        ])

                # Update YAML if you still need it (only when capturing, or always—your choice)
                # if is_capturing:
                # #save the frame in folder named ip_address_frames
                # # cv2.imwrite(f"frame_{frame_id}.png", frame)
                #     os.makedirs(f"{ip_address}_frames", exist_ok=True)
                #     cv2.imwrite(f"{ip_address}_frames/{frame_id}.png", frame)                

                if tag_data and is_capturing:
                    update_yaml(yaml_filename, tag_data)

                # Show the frame
                cv2.imshow("Basler Camera Feed - View Only", frame)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    break

                # Press 'c' to start capturing
                if key == ord('c') and not is_capturing:
                    print("Frame data capturing started")
                    is_capturing = True

                # Press 's' to stop capturing
                if key == ord('s') and is_capturing:
                    print("Frame data capturing stopped")
                    is_capturing = False


                    
                        

            grabResult.Release()

    cv2.destroyAllWindows()

# def view_feed(camera, converter, detector, calibration_yaml_path, ip_address):
#     import yaml
#     with open(calibration_yaml_path, 'r') as file:
#         calib_data = yaml.safe_load(file)

#     # Extract camera matrix and distortion coefficients
#     mtx = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
#     dist = np.array(calib_data['dist_coeff']['data'])

#     # Camera resolution for undistortion
#     frame_width = 1920  # Update based on your camera resolution
#     frame_height = 1200
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height))

#     # YAML file for storing AprilTag data
#     yaml_filename = f"{ip_address}_xyz.yaml"

#     while camera.IsGrabbing():
#         grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#         if grabResult.GrabSucceeded():
#             image = converter.Convert(grabResult)
#             frame = image.GetArray()
#             undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
#             x, y, w, h = roi
#             undistorted_frame = undistorted_frame[y:y+h, x:x+w]
#             frame = undistorted_frame
#             # Convert the image to grayscale (AprilTag detection typically works on grayscale images)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Detect AprilTags in the image
#             detections = detector.detect(gray)

#             # Prepare data for updating YAML
#             tag_data = {}

#             # Loop through all detected tags
#             for detection in detections:
#                 # Get the tag's ID
#                 tag_id = str(detection.tag_id)

#                 # Get the corners of the detected tag
#                 corners = detection.corners
#                 top_left = tuple(corners[0])
#                 top_right = tuple(corners[1])
#                 bottom_right = tuple(corners[2])
#                 bottom_left = tuple(corners[3])

#                 # Calculate the center of the AprilTag
#                 center_x = (top_left[0] + bottom_right[0]) / 2
#                 center_y = (top_left[1] + bottom_right[1]) / 2
#                 center = (int(center_x), int(center_y))

#                 pose, _, _ = detector.detection_pose(
#                     detection,
#                     camera_params=(mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]),
#                     tag_size=10.375 * 0.0254
#                 )

#                 # Extract rotation and translation
#                 print(pose)
                
#                 rotation_matrix = pose[:3, :3]
#                 translation_vector = pose[:3, 3]

#                 # Store tag data
#                 tag_data[tag_id] = {
#                     'center': {'x': center[0], 'y': center[1]},
#                     'top_left': {'x': int(top_left[0]), 'y': int(top_left[1])},
#                     'top_right': {'x': int(top_right[0]), 'y': int(top_right[1])},
#                     'bottom_right': {'x': int(bottom_right[0]), 'y': int(bottom_right[1])},
#                     'bottom_left': {'x': int(bottom_left[0]), 'y': int(bottom_left[1])},
#                     'rotation_matrix': rotation_matrix.tolist(),
#                     'translation_vector': translation_vector.tolist()
#                 }

#                 # Draw a bounding box around the detected tag
#                 cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

#                 # Draw the tag ID at the center of the tag
#                 cv2.putText(frame, str(tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Update the YAML file with detected tags
#             update_yaml(yaml_filename, tag_data)

#             # Display the frame with detected tags
#             cv2.imshow('Basler Camera Feed - View Only', frame)

#             # Exit on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         grabResult.Release()

# Function to view and record the feed (continuous)
def view_and_record_feed(camera, converter):
    frame_width = 1920  # adjust based on your camera's resolution
    frame_height = 1200  # adjust based on your camera's resolution
    out = cv2.VideoWriter('camera_feed.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
    with open("frame_metadata.txt", "w") as metadata_file:
        metadata_file.write("FrameID,UnixTimestamp(seconds)\n")

        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()
                
                # Display the feed
                cv2.imshow('Basler Camera Feed - Recording', img)

                # Record the video frame
                out.write(img)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            grabResult.Release()
    
    # Release the VideoWriter object
    out.release()

# Function to capture and view just one frame
# def view_single_frame(camera, converter):
#     grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#     if grabResult.GrabSucceeded():
#         image = converter.Convert(grabResult)
#         img = image.GetArray()
        
#         # Display the single frame
#         cv2.imshow('Basler Camera Feed - Single Frame', img)
#         cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        
#     grabResult.Release()
#     cv2.destroyAllWindows()

# Main function
def main():
    # Camera IP address and calibration YAML file
    camera_ip = sys.argv[1]
    calibration_yaml_path = sys.argv[2]
    output_csv_path = camera_ip+".csv"

    # Create an instant camera object
    camera = grab_by_ip.get_camera_by_ip(camera_ip)
    print("Camera initialized.")
    
    # Open the camera connection
    camera.Open()
    configure_camera(camera)

    # Set the acquisition mode to continuous
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # ImageFormatConverter to convert images to OpenCV's format
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Initialize the AprilTag detector
    detector = apriltag.Detector()

    #check the orientation of the april tag
    # also the pose is relative to an origin, so is origin the centre or corner

    # View feed and save data to YAML
    view_feed(camera, converter, detector, calibration_yaml_path, camera_ip,output_csv_path)

    # Stop grabbing and close the camera connection
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

# Call the main function when the script is executed
if __name__ == "__main__":
    main()
