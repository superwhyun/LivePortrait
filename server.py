import socket
import pickle
import cv2
import numpy as np
import argparse
import signal
import os
from enum import Enum
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.crop import _transform_img, prepare_paste_back, paste_back
from src.utils.io import load_image_rgb, resize_to_limit
from src.utils.camera import get_rotation_matrix
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.cropper import Cropper
import tyro

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

class MessageType(Enum):
    LANDMARK_LIST = 1
    FEATURE = 2
    END_OF_STREAM = 3

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

class AnimationGenerator:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig, source_image_path: str):
        self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.inference_cfg = inference_cfg
        self.source_image_path = source_image_path
        self.driving_lmk = None

        img_rgb = load_image_rgb(source_image_path)
        crop_info = self.cropper.crop_single_image(img_rgb)
        self.source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if self.inference_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            I_s = self.live_portrait_wrapper.prepare_source(img_rgb)

        self.x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        self.x_c_s = self.x_s_info['kp']
        self.R_s = get_rotation_matrix(self.x_s_info['pitch'], self.x_s_info['yaw'], self.x_s_info['roll'])
        self.f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        self.x_s = self.live_portrait_wrapper.transform_keypoint(self.x_s_info)

        if self.inference_cfg.flag_pasteback:
            self.mask_ori = prepare_paste_back(self.inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            self.I_p_paste_lst = []

        self.crop_info = crop_info
        self.img_rgb = img_rgb

    def set_driving_landmarks(self, driving_landmarks):
        self.driving_lmk = driving_landmarks
        print(f"Set driving landmarks. Length: {len(driving_landmarks)}")

    def generate_animation_frame(self, feature):
        if self.driving_lmk is None:
            raise ValueError("Driving landmarks not set. Cannot generate animation frame.")

        x_d_i_info = feature['kp_info']
        R_d_i = feature['rotation']

        if self.inference_cfg.flag_relative:
            R_new = (R_d_i @ self.R_s.permute(0, 2, 1)) @ self.R_s
            delta_new = self.x_s_info['exp'] + (x_d_i_info['exp'] - self.x_s_info['exp'])
            scale_new = self.x_s_info['scale'] * (x_d_i_info['scale'] / self.x_s_info['scale'])
            t_new = self.x_s_info['t'] + (x_d_i_info['t'] - self.x_s_info['t'])
        else:
            R_new = R_d_i
            delta_new = x_d_i_info['exp']
            scale_new = self.x_s_info['scale']
            t_new = x_d_i_info['t']

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new

        if self.inference_cfg.flag_stitching:
            x_d_i_new = self.live_portrait_wrapper.stitching(self.x_s, x_d_i_new)

        if self.inference_cfg.flag_eye_retargeting:
            c_d_eyes_i = self.live_portrait_wrapper.calc_eye_ratio(self.driving_lmk)
            combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, self.source_lmk)
            eyes_delta = self.live_portrait_wrapper.retarget_eye(self.x_s, combined_eye_ratio_tensor)
            x_d_i_new += eyes_delta.reshape(-1, self.x_s.shape[1], 3)

        if self.inference_cfg.flag_lip_retargeting:
            c_d_lip_i = self.live_portrait_wrapper.calc_lip_ratio(self.driving_lmk)
            combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, self.source_lmk)
            lip_delta = self.live_portrait_wrapper.retarget_lip(self.x_s, combined_lip_ratio_tensor)
            x_d_i_new += lip_delta.reshape(-1, self.x_s.shape[1], 3)

        out = self.live_portrait_wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]

        if self.inference_cfg.flag_pasteback:
            I_p_i_to_ori_blend = paste_back(I_p_i, self.crop_info['M_c2o'], self.img_rgb, self.mask_ori)
            return cv2.cvtColor(I_p_i_to_ori_blend, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(I_p_i, cv2.COLOR_RGB2BGR)

def receive_message(conn):
    size_data = conn.recv(4)
    if len(size_data) == 0:
        return None, None, 0
    size = int.from_bytes(size_data, byteorder='big')
    data = b''
    while len(data) < size:
        chunk = conn.recv(min(size - len(data), 4096))
        if not chunk:
            raise RuntimeError("socket connection broken")
        data += chunk
    msg_type, msg_data = pickle.loads(data)
    return MessageType(msg_type), msg_data, size

def process_client(conn, addr, animation_generator, killer):
    print(f"Connected by {addr}")

    cv2.namedWindow("LivePortrait Animation", cv2.WINDOW_NORMAL)

    frame_count = 0

    try:
        while not killer.kill_now:
            msg_type, msg_data, msg_size = receive_message(conn)
            if msg_type is None:
                break

            print(f"Received message: Type={msg_type.name}, Size={msg_size} bytes")

            if msg_type == MessageType.LANDMARK_LIST:
                animation_generator.set_driving_landmarks(msg_data)
                print(f"Executed set_driving_landmarks")
            elif msg_type == MessageType.FEATURE:
                if animation_generator.driving_lmk is None:
                    print("Error: Received feature before landmark list")
                    continue  # Skip this packet

                feature = msg_data
                try:
                    frame = animation_generator.generate_animation_frame(feature)
                    cv2.imshow("LivePortrait Animation", frame)
                    print(f"Executed generate_animation_frame")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frame_count += 1
                    print(f"Processed frame {frame_count} for {addr}")
                except ValueError as e:
                    print(f"Error generating animation frame: {e}")
            elif msg_type == MessageType.END_OF_STREAM:
                print(f"Received end-of-stream signal from {addr}")
                break

    except Exception as e:
        print(f"Error processing client {addr}: {e}")
    finally:
        cv2.destroyAllWindows()
        conn.close()
        print(f"Finished processing animation for {addr}")

def run_server(args: ArgumentConfig):
    if not os.path.exists(args.source_image):
        raise FileNotFoundError(f"Source image not found: {args.source_image}")

    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    animation_generator = AnimationGenerator(inference_cfg, crop_cfg, args.source_image)

    killer = GracefulKiller()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.server_name, args.port))
        s.listen()
        print(f"Server listening on {args.server_name}:{args.port}")

        while not killer.kill_now:
            try:
                s.settimeout(1.0)  # Set a timeout for accept()
                conn, addr = s.accept()
                process_client(conn, addr, animation_generator, killer)
            except socket.timeout:
                continue  # This allows checking the kill_now flag periodically

    print("Server shutting down gracefully...")

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    run_server(args)
