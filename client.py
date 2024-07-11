import socket
import pickle
import cv2
import numpy as np
import argparse
from typing import List, Dict
from enum import Enum
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.features import LivePortraitWrapper, Cropper
from src.utils.io import load_driving_info
from src.utils.camera import get_rotation_matrix
import tyro


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

class MessageType(Enum):
    LANDMARK_LIST = 1
    FEATURE = 2
    END_OF_STREAM = 3

class RealTimeFeatureExtractor:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def extract_features(self, video_path: str, socket_conn, num_loops: int = 1):
        driving_rgb_lst = load_driving_info(video_path)
        driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        driving_lmk_lst = self.cropper.get_retargeting_lmk_info(driving_rgb_lst)

        # Send landmark list first
        self.send_message(socket_conn, MessageType.LANDMARK_LIST, driving_lmk_lst)
        print("Sent landmark list")

        I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)

        for loop in range(num_loops):
            print(f"Starting loop {loop + 1}/{num_loops}")
            for i, I_d_i in enumerate(I_d_lst):
                x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
                R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

                feature = {
                    'kp_info': x_d_i_info,
                    'rotation': R_d_i
                }

                # Send the feature
                self.send_message(socket_conn, MessageType.FEATURE, feature)
                print(f"Sent features for frame {i+1} in loop {loop + 1}")

        # Send end-of-stream signal
        self.send_message(socket_conn, MessageType.END_OF_STREAM, None)

    def send_message(self, socket_conn, msg_type: MessageType, data):
        msg = pickle.dumps((msg_type.value, data))
        socket_conn.sendall(len(msg).to_bytes(4, byteorder='big'))
        socket_conn.sendall(msg)

def run_client(args: ArgumentConfig):
    # Load configs
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    # Initialize RealTimeFeatureExtractor
    extractor = RealTimeFeatureExtractor(inference_cfg, crop_cfg)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        print(f"Connected to {args.host}:{args.port}")

        # Extract and send features
        extractor.extract_features(args.driving_info, s, args.num_loops)

    print("All frames processed and sent")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    run_client(args)
