#!/usr/bin/env python3
"""Dump numerical reference data from the full dog detection pipeline for integration testing.

Prints JSON with SSD detections, species classification, body pose keypoints, face bbox, and face landmarks.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

# Reuse everything from the inference script
sys.path.insert(0, str(Path(__file__).parent))
from infer_dog_pose_tflite import (
    run_detector, expand_box, preprocess_letterbox, run_rtmpose, run_hrnet,
    unletterbox_keypoints, classify_species, load_species_mapping,
    run_face_localizer, get_face_landmarks_in_image,
    CLASSIFIER_PATH, SPECIES_MAPPING_PATH, FACE_LOCALIZER_PATH, FACE_MODEL_CONFIGS,
    NUM_KEYPOINTS,
)


def main():
    image_path = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "rtmpose"

    model_paths = {
        "hrnet": Path("artifacts/superanimal_pose/superanimal_hrnet_w32_float16.tflite"),
        "rtmpose": Path("artifacts/superanimal_pose/superanimal_rtmpose_s_float16.tflite"),
    }
    detector_path = Path("artifacts/superanimal_pose/superanimal_ssdlite_float16.tflite")
    anchors_path = Path("checkpoints/ssdlite_anchors.npy")

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Load species mapping
    species_lookup = load_species_mapping(SPECIES_MAPPING_PATH)
    try:
        from torchvision.models import MobileNet_V3_Small_Weights
        imagenet_categories = list(MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"])
    except Exception:
        imagenet_categories = []

    result = {
        "image": str(image_path),
        "image_size": [orig_w, orig_h],
        "model": model_name,
        "dogs": [],
    }

    # SSD detection
    detections = run_detector(detector_path, anchors_path, image, det_threshold=0.5)

    for det_idx, (x1, y1, x2, y2, det_score) in enumerate(detections):
        dog = {
            "det_bbox": [x1, y1, x2, y2],
            "det_score": float(det_score),
        }

        # Expand box 10%
        cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, 0.10, orig_w, orig_h)
        crop = image.crop((cx1, cy1, cx2, cy2))
        dog["expanded_crop"] = [cx1, cy1, cx2, cy2]

        # Species classification on unexpanded crop
        raw_crop = image.crop((x1, y1, x2, y2))
        species, specific, cls_conf = classify_species(
            CLASSIFIER_PATH, species_lookup, imagenet_categories, raw_crop
        )
        dog["species"] = species
        dog["species_class"] = specific
        dog["species_confidence"] = float(cls_conf)

        # Pose on expanded crop
        inp, scale, pad_x, pad_y = preprocess_letterbox(crop)
        if model_name == "hrnet":
            kps_256, confs = run_hrnet(model_paths["hrnet"], inp)
        else:
            kps_256, confs = run_rtmpose(model_paths["rtmpose"], inp)

        kps_orig = unletterbox_keypoints(kps_256, scale, pad_x, pad_y, cx1, cy1)

        # Body keypoints (15-38)
        body_kps = []
        for i in range(15, NUM_KEYPOINTS):
            body_kps.append({
                "index": i,
                "name": ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
                         "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
                         "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
                         "neck_base", "neck_end", "throat_base", "throat_end",
                         "back_base", "back_end", "back_middle", "tail_base", "tail_end",
                         "front_left_thai", "front_left_knee", "front_left_paw",
                         "front_right_thai", "front_right_knee", "front_right_paw",
                         "back_left_paw", "back_left_thai", "back_right_thai",
                         "back_left_knee", "back_right_knee", "back_right_paw",
                         "belly_bottom", "body_middle_right", "body_middle_left"][i],
                "x": round(float(kps_orig[i, 0]), 2),
                "y": round(float(kps_orig[i, 1]), 2),
                "confidence": round(float(confs[i]), 6),
            })
        dog["body_keypoints"] = body_kps

        # Face detection + landmarks
        face_bbox_orig, face_lm_orig = get_face_landmarks_in_image(
            FACE_LOCALIZER_PATH,
            FACE_MODEL_CONFIGS,
            crop,
            dog_crop_offset=(cx1, cy1),
            crop_margin=0.20,
        )
        if face_bbox_orig is not None:
            dog["face_bbox"] = [round(float(v), 2) for v in face_bbox_orig]
            dog["face_landmarks"] = [
                {"index": i, "x": round(float(face_lm_orig[i, 0]), 2), "y": round(float(face_lm_orig[i, 1]), 2)}
                for i in range(len(face_lm_orig))
            ]
        else:
            dog["face_bbox"] = None
            dog["face_landmarks"] = None

        result["dogs"].append(dog)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
