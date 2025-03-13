import os
import math
import cv2
from pypylon import pylon
import numpy as np
import yaml
import time

def get_current_fps(camera):
    try:
        fps = camera.AcquisitionFrameRate.GetValue()
        return fps
    except Exception as e:
        print(f"Error retrieving FPS: {e}")
        return None

def configure_camera_resolution(camera, width, height):
    camera.DeviceLinkThroughputLimit.SetValue(20000000)
    camera.Width.SetValue(min(camera.Width.Max, width))
    camera.Height.SetValue(min(camera.Height.Max, height))
    fps = get_current_fps(camera)
    print("FPS:", fps)
    print(f"Camera resolution set to: {camera.Width.GetValue()}x{camera.Height.GetValue()}")

def capture_images_at_intervals(camera, converter, interval_seconds=3, num_images=50, save_path="captured_images/", index=0):
    # Create a unique folder for each camera based on the index
    camera_folder = os.path.join(save_path, f"camera_{index}")
    os.makedirs(camera_folder, exist_ok=True)  # Create the folder if it doesn't exist

    if not camera.IsOpen():
        camera.Open()

    for i in range(num_images):
        if camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()

                # Save the image to the camera-specific folder
                img_filename = os.path.join(camera_folder, f"{index}_{i + 1}.jpg")
                cv2.imwrite(img_filename, img)
                print(f"Saved image: {img_filename}")

            grabResult.Release()
            time.sleep(interval_seconds)

    camera.StopGrabbing()
    camera.Close()
    print("Image capture complete for camera", index)

def get_camera_by_ip(ip_address):
    transport_layer_factory = pylon.TlFactory.GetInstance()
    all_devices = transport_layer_factory.EnumerateDevices()

    for device in all_devices:
        if device.GetDeviceClass() == "BaslerGigE" and device.GetIpAddress() == ip_address:
            camera = pylon.InstantCamera(transport_layer_factory.CreateDevice(device))
            print(f"Connected to camera with IP: {ip_address}")
            return camera

    print(f"No camera found with IP address: {ip_address}")
    return None

def load_ip_from_yaml(yaml_file):
    ip_addresses = []
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            ip_addresses.append(value['ip_address'])
    return ip_addresses

def main():
    config_file = 'cameras_config.yml'
    ip_addresses = load_ip_from_yaml(config_file)
    print("IP addresses:", ip_addresses)

    cameras = []
    converters = []

    if ip_addresses:
        for i, ip in enumerate(ip_addresses):
            camera = get_camera_by_ip(ip)

            if camera is not None:
                camera.Open()
                configure_camera_resolution(camera, width=1920, height=1200)

                try:
                    camera.PixelFormat.SetValue("BGR8Packed")
                except Exception as e:
                    print(f"Error setting pixel format: {e}")

                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

                cameras.append(camera)
                converters.append(converter)

        # for i in range(len(cameras)):
        capture_images_at_intervals(cameras[1], converters[1], index=1)

if __name__ == "__main__":
    main()
