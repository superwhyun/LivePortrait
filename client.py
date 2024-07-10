import socket
import pickle
import cv2
import numpy as np
import argparse
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.features import FeatureExtractor
from src.utils.io import load_image_rgb, resize_to_limit

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def run_client(host='127.0.0.1', port=65432, driving_video_path='driving_video.mp4'):
    # Load configs
    args = ArgumentConfig()
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    # Initialize FeatureExtractor
    extractor = FeatureExtractor(inference_cfg, crop_cfg)

    # Open video file
    cap = cv2.VideoCapture(driving_video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {driving_video_path}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to {host}:{port}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = resize_to_limit(frame_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)

            # Extract features for the frame
            driving_rgb_lst = [frame_rgb]
            driving_rgb_lst_256 = [cv2.resize(frame_rgb, (256, 256))]
            driving_lmk_lst = extractor.cropper.get_retargeting_lmk_info(driving_rgb_lst)
            I_d_lst = extractor.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)

            features = []
            for I_d_i in I_d_lst:
                x_d_i_info = extractor.live_portrait_wrapper.get_kp_info(I_d_i)
                R_d_i = extractor.live_portrait_wrapper.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
                features.append({
                    'kp_info': x_d_i_info,
                    'rotation': R_d_i
                })

            # Serialize and send the data
            data = pickle.dumps((features, driving_lmk_lst))
            s.sendall(len(data).to_bytes(4, byteorder='big'))  # Send data size first
            s.sendall(data)

            frame_count += 1
            print(f"Sent features for frame {frame_count}")

        # Send end-of-stream signal
        s.sendall(b'\x00\x00\x00\x00')

    cap.release()
    print("All frames processed and sent")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LivePortrait Animation Client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=65432, help="Server port")
    parser.add_argument("-d", "--driving", required=True, help="Path to driving video")

    args = parser.parse_args()

    run_client(host=args.host, port=args.port, driving_video_path=args.driving)
