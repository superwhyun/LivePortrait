# coding: utf-8

import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


import os
from src.features import FeatureExtractor
from src.features import AnimationGenerator
def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

import os
import pickle

def is_pickle_file(file_path):
    return file_path.lower().endswith('.pkl')

def main(source_image_path: str, driving_video_path: str, output_dir: str):
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    if is_pickle_file(driving_video_path):
        # driving_video_path가 pickle 파일인 경우
        features_path = driving_video_path
        print(f"기존 특징 파일 {features_path}를 사용합니다.")
    else:
        # driving_video_path가 동영상 파일인 경우
        # 특징 추출
        extractor = FeatureExtractor(inference_cfg, crop_cfg)
        features, driving_lmk_lst = extractor.extract_features(driving_video_path)

        # 추출된 특징 저장
        features_path = os.path.join(output_dir, "extracted_features.pkl")
        extractor.save_features(features, driving_lmk_lst, features_path)
        print(f"특징이 {features_path}에 저장되었습니다.")

    # 애니메이션 생성
    generator = AnimationGenerator(inference_cfg, crop_cfg)
    output_video_path = generator.generate_animation(source_image_path, features_path, output_dir)

    print(f"애니메이션이 {output_video_path}에 생성되었습니다.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LivePortrait 애니메이션 생성기")
    parser.add_argument("-s", "--source", required=False, help="소스 이미지 경로")
    parser.add_argument("-d", "--driving", required=False, help="드라이빙 비디오 경로")
    parser.add_argument("-o", "--output", default="output", help="출력 디렉토리")

    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)

    main(args.source, args.driving, args.output)
