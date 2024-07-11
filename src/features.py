from os import mkdir
import cv2
import numpy as np
import pickle
import os.path as osp
import os
from rich.progress import track
import sys

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.retargeting_utils import calc_lip_close_ratio
from .utils.io import load_image_rgb, load_driving_info, resize_to_limit
from .utils.helper import mkdir, basename, dct2cuda, is_video, is_template
from .utils.rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapper
from typing import List, Dict

class FeatureExtractor:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def extract_features(self, video_path: str) -> List[Dict]:
        driving_rgb_lst = load_driving_info(video_path)
        driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        driving_lmk_lst = self.cropper.get_retargeting_lmk_info(driving_rgb_lst)

        I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)

        features = []
        for i, I_d_i in enumerate(I_d_lst):
            x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
            R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

            features.append({
                'kp_info': x_d_i_info,
                'rotation': R_d_i
            })
            with open(f'./output/feature_{i}', "wb") as f:
                pickle.dump((x_d_i_info, R_d_i), f)

        return features, driving_lmk_lst

    def save_features(self, features: List[Dict], driving_lmk_lst: List, output_path: str):
        with open(output_path, 'wb') as f:
            pickle.dump((features, driving_lmk_lst), f)

import time
class AnimationGenerator:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.inference_cfg = inference_cfg

    def generate_animation(self, source_image_path: str, features_path: str, output_dir: str):
        # 소스 이미지 처리

        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, self.inference_cfg.ref_max_shape, self.inference_cfg.ref_shape_n)
        crop_info = self.cropper.crop_single_image(img_rgb)
        source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if self.inference_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            I_s = self.live_portrait_wrapper.prepare_source(img_rgb)

        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        # 특징 로드
        with open(features_path, 'rb') as f:
            features, driving_lmk_lst = pickle.load(f)


        # 붙여넣기 준비
        if self.inference_cfg.flag_pasteback:
            mask_ori = prepare_paste_back(self.inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            I_p_paste_lst = []

        # 애니메이션 생성
        I_p_lst = []
        for i, feature in enumerate(features):
            start_time = time.time()  # 루프 시작 시간 기록
            x_d_i_info = feature['kp_info']
            R_d_i = feature['rotation']

            if self.inference_cfg.flag_relative:
                R_new = (R_d_i @ features[0]['rotation'].permute(0, 2, 1)) @ R_s
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - features[0]['kp_info']['exp'])
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / features[0]['kp_info']['scale'])
                t_new = x_s_info['t'] + (x_d_i_info['t'] - features[0]['kp_info']['t'])
            else:
                R_new = R_d_i
                delta_new = x_d_i_info['exp']
                scale_new = x_s_info['scale']
                t_new = x_d_i_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # 스티칭 및 재타겟팅 로직
            if self.inference_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            if self.inference_cfg.flag_eye_retargeting:
                c_d_eyes_i = self.live_portrait_wrapper.calc_eye_ratio(driving_lmk_lst[i])
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                x_d_i_new += eyes_delta.reshape(-1, x_s.shape[1], 3)

            if self.inference_cfg.flag_lip_retargeting:
                c_d_lip_i = self.live_portrait_wrapper.calc_lip_ratio(driving_lmk_lst[i])
                combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)
                x_d_i_new += lip_delta.reshape(-1, x_s.shape[1], 3)

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]

            I_p_lst.append(I_p_i)

            if self.inference_cfg.flag_pasteback:
                I_p_i_to_ori_blend = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
                I_p_paste_lst.append(I_p_i_to_ori_blend)
            end_time = time.time()  # 루프 종료 시간 기록
            loop_duration = end_time - start_time  # 루프 실행 시간 계산
            print(f"프레임 {i+1} 처리 시간: {loop_duration:.4f} 초")

        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        wfp = os.path.join(output_dir, f'{os.path.basename(source_image_path)}--animation.mp4')
        if self.inference_cfg.flag_pasteback:
            images2video(I_p_paste_lst, wfp=wfp)
        else:
            images2video(I_p_lst, wfp=wfp)

        return wfp
