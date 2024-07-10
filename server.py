import socket
import pickle
import os
import cv2
import numpy as np
import argparse
import signal
import sys
from datetime import datetime
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.features import AnimationGenerator
from src.utils.io import load_image_rgb, resize_to_limit

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

def process_client(conn, addr, generator, inference_cfg, crop_info, f_s, x_s, img_rgb, output_dir, killer):
    print(f"Connected by {addr}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'output_{addr[0]}_{addr[1]}_{timestamp}.mp4'
    output_path = os.path.join(output_dir, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (img_rgb.shape[1], img_rgb.shape[0]))

    frame_count = 0
    try:
        while not killer.kill_now:
            size_data = conn.recv(4)
            if len(size_data) == 0:
                break
            size = int.from_bytes(size_data, byteorder='big')
            if size == 0:  # End-of-stream signal
                break

            data = b''
            while len(data) < size:
                chunk = conn.recv(min(size - len(data), 4096))
                if not chunk:
                    raise RuntimeError("socket connection broken")
                data += chunk

            features, driving_lmk_lst = pickle.loads(data)

            for feature in features:
                x_d_i_info = feature['kp_info']
                R_d_i = feature['rotation']

                x_d_i_new = generator.live_portrait_wrapper.transform_keypoint(x_d_i_info)

                out_data = generator.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
                I_p_i = generator.live_portrait_wrapper.parse_output(out_data['out'])[0]

                if inference_cfg.flag_pasteback:
                    I_p_i_to_ori_blend = generator.paste_back(I_p_i, crop_info['M_c2o'], img_rgb, generator.inference_cfg.mask_crop)
                    out.write(cv2.cvtColor(I_p_i_to_ori_blend, cv2.COLOR_RGB2BGR))
                else:
                    out.write(cv2.cvtColor(I_p_i, cv2.COLOR_RGB2BGR))

            frame_count += 1
            print(f"Processed frame {frame_count} for {addr}")

    except Exception as e:
        print(f"Error processing client {addr}: {e}")
    finally:
        out.release()
        conn.close()
        print(f"Animation for {addr} generated and saved to {output_path}")

def run_server(host='0.0.0.0', port=65432, source_image_path='source_image.jpg', output_dir='output'):
    if not os.path.exists(source_image_path):
        raise FileNotFoundError(f"Source image not found: {source_image_path}")

    args = ArgumentConfig()
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    generator = AnimationGenerator(inference_cfg, crop_cfg)

    os.makedirs(output_dir, exist_ok=True)

    img_rgb = load_image_rgb(source_image_path)
    img_rgb = resize_to_limit(img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
    crop_info = generator.cropper.crop_single_image(img_rgb)
    img_crop_256x256 = crop_info['img_crop_256x256']
    I_s = generator.live_portrait_wrapper.prepare_source(img_crop_256x256)
    x_s_info = generator.live_portrait_wrapper.get_kp_info(I_s)
    f_s = generator.live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = generator.live_portrait_wrapper.transform_keypoint(x_s_info)

    killer = GracefulKiller()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")

        while not killer.kill_now:
            try:
                s.settimeout(1.0)  # Set a timeout for accept()
                conn, addr = s.accept()
                process_client(conn, addr, generator, inference_cfg, crop_info, f_s, x_s, img_rgb, output_dir, killer)
            except socket.timeout:
                continue  # This allows checking the kill_now flag periodically

    print("Server shutting down gracefully...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LivePortrait Animation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=65432, help="Server port")
    parser.add_argument("--source", required=True, help="Path to source image")
    parser.add_argument("--output", default="output", help="Output directory")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, source_image_path=args.source, output_dir=args.output)
