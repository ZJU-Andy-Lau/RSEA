import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model.encoder_dino_0927 import EncoderDino


@dataclass
class WarpParams:
    rotation_deg: float
    scale: float
    translate_x: float
    translate_y: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(path: str, size: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = np.stack([image] * 3,axis=-1)
    return image


def build_affine_matrix(params: WarpParams, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, params.rotation_deg, params.scale)
    matrix[0, 2] += params.translate_x
    matrix[1, 2] += params.translate_y
    return matrix


def warp_image(image: np.ndarray, matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image


def extract_features(encoder: EncoderDino, image: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        feat, conf = encoder(tensor)
    return feat.squeeze(0), conf.squeeze(0)


def compute_feature_stride(input_size: Tuple[int, int], feature_shape: Tuple[int, int]) -> Tuple[float, float]:
    h_in, w_in = input_size
    h_feat, w_feat = feature_shape
    stride_y = h_in / h_feat
    stride_x = w_in / w_feat
    return stride_y, stride_x


def map_coords_affine(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((coords.shape[0], 1), dtype=np.float32)
    coords_h = np.hstack([coords, ones])
    mapped = coords_h @ matrix.T
    return mapped


def sample_features(feat: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
    h, w = feat.shape[1:]
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    coords_x = np.clip(coords_x, 0, w - 1)
    coords_y = np.clip(coords_y, 0, h - 1)
    coords = np.stack([coords_y, coords_x], axis=1)
    coords_t = torch.from_numpy(coords).long()
    sampled = feat[:, coords_t[:, 0], coords_t[:, 1]].T
    return sampled


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return (a_norm * b_norm).sum(dim=1)


def build_negative_samples(
    coords: np.ndarray,
    mask_coords: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(mask_coords) == 0:
        raise ValueError("No valid mask coordinates for negative sampling.")
    negatives = []
    for _ in range(coords.shape[0]):
        idxs = rng.choice(len(mask_coords), size=k, replace=len(mask_coords) < k)
        negatives.append(mask_coords[idxs])
    return np.stack(negatives, axis=0)


def compute_auc(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    conf_a: torch.Tensor,
    conf_b: torch.Tensor,
    input_size: Tuple[int, int],
    affine_matrix: np.ndarray,
    conf_threshold: float,
    negative_k: int,
    seed: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    h_feat, w_feat = conf_a.shape
    mask = (conf_a > conf_threshold) & (conf_b > conf_threshold)
    mask_idx = mask.nonzero(as_tuple=False)
    if mask_idx.numel() == 0:
        raise ValueError("No high-confidence pixels after thresholding.")

    coords_feat = torch.stack([mask_idx[:, 1], mask_idx[:, 0]], dim=1).cpu().numpy().astype(np.float32)
    stride_y, stride_x = compute_feature_stride(input_size, (h_feat, w_feat))
    coords_img = np.stack([coords_feat[:, 0] * stride_x, coords_feat[:, 1] * stride_y], axis=1)
    mapped_img = map_coords_affine(coords_img, affine_matrix)
    mapped_feat = np.stack([mapped_img[:, 0] / stride_x, mapped_img[:, 1] / stride_y], axis=1)

    valid_mask = (
        (mapped_feat[:, 0] >= 0)
        & (mapped_feat[:, 0] < w_feat)
        & (mapped_feat[:, 1] >= 0)
        & (mapped_feat[:, 1] < h_feat)
    )
    coords_feat = coords_feat[valid_mask]
    mapped_feat = mapped_feat[valid_mask]

    if coords_feat.shape[0] == 0:
        raise ValueError("No valid mapped coordinates after affine transform.")

    pos_a = sample_features(feat_a, coords_feat)
    pos_b = sample_features(feat_b, mapped_feat)
    pos_sims = cosine_similarity(pos_a, pos_b).cpu().numpy()

    rng = np.random.default_rng(seed)
    mask_coords = coords_feat.astype(np.float32)
    neg_coords = build_negative_samples(coords_feat, mask_coords, negative_k, rng)
    neg_a = pos_a.repeat_interleave(negative_k, dim=0)
    neg_b_list = []
    for i in range(neg_coords.shape[0]):
        neg_b_list.append(sample_features(feat_b, neg_coords[i]))
    neg_b = torch.cat(neg_b_list, dim=0)
    neg_sims = cosine_similarity(neg_a, neg_b).cpu().numpy()

    labels = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
    scores = np.concatenate([pos_sims, neg_sims])
    auc = roc_auc_score(labels, scores)
    return auc, pos_sims, neg_sims


def build_encoder(args: argparse.Namespace, device: torch.device) -> EncoderDino:
    encoder = EncoderDino(
        dino_weight_path=args.dino_weight,
        upsample_times=args.upsample_times,
        use_adapter=args.use_adapter,
        use_conf=args.use_conf,
    )
    if args.adapter_path:
        encoder.load_adapter(args.adapter_path)
    encoder.to(device)
    encoder.eval()
    return encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-a", required=True, help="Path to the first image")
    parser.add_argument("--image-b", required=True, help="Path to the second image")
    parser.add_argument("--dino-weight", required=True, help="Path to DINOv3 weight file")
    parser.add_argument("--adapter-path", default=None, help="Path to adapter weights (optional)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--size", type=int, default=1024, help="Resize size (square)")
    parser.add_argument("--apply-warp", action="store_true", help="Apply affine warp to image B")
    parser.add_argument("--rotation-deg", type=float, default=0.0, help="Rotation in degrees")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
    parser.add_argument("--translate-x", type=float, default=0.0, help="Translation in pixels (x)")
    parser.add_argument("--translate-y", type=float, default=0.0, help="Translation in pixels (y)")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--negative-k", type=int, default=10, help="Number of negatives per anchor")
    parser.add_argument("--upsample-times", type=int, default=0, help="Upsample times in encoder")
    parser.add_argument("--use-adapter", action="store_true", default=True, help="Use adapter in encoder")
    parser.add_argument("--no-adapter", dest="use_adapter", action="store_false", help="Disable adapter in encoder")
    parser.add_argument("--use-conf", action="store_true", default=True, help="Use confidence head in encoder")
    parser.add_argument("--no-conf", dest="use_conf", action="store_false", help="Disable confidence head in encoder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(args.seed)

    size = (args.size, args.size)
    image_a = load_image(args.image_a, size)
    image_b = load_image(args.image_b, size)

    warp_params = WarpParams(
        rotation_deg=args.rotation_deg,
        scale=args.scale,
        translate_x=args.translate_x,
        translate_y=args.translate_y,
    )
    affine_matrix = np.eye(2, 3, dtype=np.float32)
    if args.apply_warp:
        affine_matrix = build_affine_matrix(warp_params, size)
        image_b = warp_image(image_b, affine_matrix, size)

    encoder = build_encoder(args, device)
    feat_a, conf_a = extract_features(encoder, image_a, device)
    feat_b, conf_b = extract_features(encoder, image_b, device)

    auc, pos_sims, neg_sims = compute_auc(
        feat_a,
        feat_b,
        conf_a.squeeze(0),
        conf_b.squeeze(0),
        (args.size, args.size),
        affine_matrix,
        args.conf_threshold,
        args.negative_k,
        args.seed,
    )

    print(f"AUC: {auc:.6f}")
    print(f"Positive samples: {len(pos_sims)}")
    print(f"Negative samples: {len(neg_sims)}")
    print(f"Positive similarity mean: {pos_sims.mean():.6f}")
    print(f"Negative similarity mean: {neg_sims.mean():.6f}")


if __name__ == "__main__":
    main()
