import socket
import pickle
import cv2
import numpy as np
import argparse
import time
from typing import List, Dict
from enum import Enum
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.features import LivePortraitWrapper, Cropper
from src.utils.camera import get_rotation_matrix
import tyro

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

class MessageType(Enum):
    LANDMARK_LIST = 1
    FEATURE = 2
    END_OF_STREAM = 3

class RealTimeFeatureExtractor:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig, fps: int = 20):
        self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.fps = fps
        self.frame_time = 1.0 / fps

    def extract_features(self, socket_conn):
        cap = cv2.VideoCapture(1)  # 1은 두 번째 카메라를 의미합니다

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 크기 조정
            frame_256 = cv2.resize(frame, (256, 256))

            # 랜드마크 추출
            lmk = self.cropper.get_retargeting_lmk_info([frame])

            # 랜드마크 리스트 전송 (매 프레임마다)
            self.send_message(socket_conn, MessageType.LANDMARK_LIST, lmk)

            # 특징 추출
            I_d = self.live_portrait_wrapper.prepare_driving_videos([frame_256])[0]
            x_d_info = self.live_portrait_wrapper.get_kp_info(I_d)
            R_d = get_rotation_matrix(x_d_info['pitch'], x_d_info['yaw'], x_d_info['roll'])

            feature = {
                'kp_info': x_d_info,
                'rotation': R_d
            }

            # 특징 전송
            self.send_message(socket_conn, MessageType.FEATURE, feature)
            print("Sent features for frame")

            # 프레임 처리 시간 계산
            process_time = time.time() - start_time

            # 목표 프레임 시간에 맞추기 위해 대기
            if process_time < self.frame_time:
                time.sleep(self.frame_time - process_time)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # 스트림 종료 신호 전송
        self.send_message(socket_conn, MessageType.END_OF_STREAM, None)

    def send_message(self, socket_conn, msg_type: MessageType, data):
        msg = pickle.dumps((msg_type.value, data))
        socket_conn.sendall(len(msg).to_bytes(4, byteorder='big'))
        socket_conn.sendall(msg)

def run_client(args: ArgumentConfig):
    # 설정 로드
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    # RealTimeFeatureExtractor 초기화 (FPS 설정 추가)
    extractor = RealTimeFeatureExtractor(inference_cfg, crop_cfg, fps=10)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        print(f"Connected to {args.host}:{args.port}")

        # 특징 추출 및 전송
        extractor.extract_features(s)

    print("Stream ended")

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    run_client(args)
