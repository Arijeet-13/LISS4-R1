#!/usr/bin/env python
"""
Visualization script for SegEarth-R2 predictions.

Generates a publication-quality figure with columns:
    QA  |  Image  |  Ground Truth (overlay)  |  Predicted Mask (overlay)

Supports datasets: LaSeRS, EarthReason, RefSegRS, RRSISD, RISBench, LISS4Reason

Usage:
    python segearth_r2/eval/visualize.py \
        --model_path  <path_to_merged_model> \
        --base_data_path <dataset_root> \
        --dataset_type LaSeRS \
        --num_samples 5 \
        --output_path output/visualization.png
"""

import os
import sys
import json
import copy
import argparse
import random
import textwrap

# -- project root on sys.path -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import transformers
from transformers import SiglipImageProcessor

from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.datasets.dataset import (
    DataCollatorForCOCODatasetV2,
    LaSeRSDataset,
    EarthReasonDataset,
    RefSegRSDataset,
    RRSISDDataset,
    LISS4ReasonDataset,
    RISBenchDataset,
)

# ==============================================================================
# Category lists per dataset (from the paper's table)
# ==============================================================================
DATASET_CATEGORIES = {
    "LaSeRS": [
        # General
        "airplane", "airport", "airport runway", "bare land",
        "baseball diamond", "baseball field", "basketball court", "beach",
        "bridge", "bridge road", "building", "bushes", "canal", "chimney",
        "cooling tower", "dam", "expressway service area",
        "expressway toll station", "farmland", "football field", "golf field",
        "grass", "green strip", "greenhouse", "ground track field", "harbor",
        "helicopter", "helipad", "intersection", "jet bridge", "lake",
        "large vehicle", "overpass", "parking lot", "path", "paved road",
        "paved square", "plane", "playground", "railway", "river", "road",
        "roundabout", "sea", "ship", "slide", "small car", "small vehicle",
        "soccer ball field", "solar panel", "sports field", "stadium",
        "storage tank", "substation", "swimming pool", "tennis court",
        "terminal", "train station", "tree", "unimproved road", "vehicle",
        "volleyball court", "water", "white smoke", "windmill",
        # Fine-grained Concept
        "B1-B boomer", "a220", "a321", "a330", "a350", "arj21",
        "boeing737", "boeing747", "boeing777", "boeing787", "bus", "c919",
        "cargo truck", "container crane", "dry cargo ship",
        "dockside warehouse", "dump truck", "driveway", "engineering ship",
        "excavator", "fishing boat", "hangar", "liquid cargo ship",
        "motorboat", "passenger ship", "tractor", "trailer",
        "truck tractor", "tugboat", "van", "warship",
        # Part
        "airplane engine", "bleachers", "bow of ship", "cargo hold",
        "center circle", "center line", "center service line",
        "cooling tower shell", "cooling tower top opening", "downstream",
        "football net", "fuselage", "horizontal stabilizer",
        "industrial pipeline", "net",
        "no man's land of tennis court", "riverbank", "service box",
        "shipping container", "stern of ship", "tennis net",
        "three-point line", "upstream", "wake", "wing", "zebra crossing",
    ],
    "RefSegRS": [
        "road", "vehicle", "car", "van", "building", "truck", "trailer",
        "bus", "road marking", "bikeway", "sidewalk", "tree",
        "low vegetation", "impervious surface",
    ],
    "RRSISD": [
        "airplane", "airport", "golf field", "expressway service area",
        "baseball field", "stadium", "ground track field", "storage tank",
        "basketball court", "chimney", "tennis court", "overpass",
        "train station", "ship", "express toll station", "dam", "harbor",
        "bridge", "vehicle", "windmill",
    ],
    "RISBench": [
        "expressive service area", "expressive toll station",
        "ground track field", "basketball court", "container crane",
        "roundabout", "windmill", "overpass", "stadium", "bridge",
        "soccer ball field", "baseball diamond", "train station",
        "golf field", "airport", "harbor", "dam", "ship", "helipad",
        "vehicle", "chimney", "airplane", "helicopter", "tennis court",
        "storage tank", "swimming pool",
    ],
    "EarthReason": [
        "storage tank", "bridge", "intersection", "tennis court",
        "baseball field", "substation", "pier", "viaduct", "wind turbine",
        "church", "airport runway", "swimming pool", "lake",
        "airport helipad", "dam", "railway", "basketball court", "beach",
        "greenhouse", "roundabout", "solar power plant",
        "ground track field", "wateraste plant", "river", "train station",
        "stadium", "island", "factory",
    ],
    "LISS4Reason": [
        "built", "water", "crops", "trees", "grass", "bare",
        "flooded vegetation", "shrub", "snow", "clouds",
    ],
}

# ==============================================================================
# 20 visually distinct colours  (RGB 0-1)
# ==============================================================================
PALETTE_RGB = [
    (0.90, 0.10, 0.10),   # red
    (0.10, 0.70, 0.10),   # green
    (0.15, 0.30, 0.90),   # blue
    (1.00, 0.75, 0.00),   # amber / gold
    (0.60, 0.10, 0.90),   # purple
    (0.00, 0.80, 0.80),   # cyan
    (1.00, 0.40, 0.70),   # pink
    (0.55, 0.35, 0.15),   # brown
    (0.40, 0.80, 0.20),   # lime
    (1.00, 0.50, 0.00),   # orange
    (0.00, 0.50, 0.70),   # teal
    (0.80, 0.00, 0.40),   # magenta-ish
    (0.50, 0.50, 0.00),   # olive
    (0.30, 0.70, 0.70),   # sea-green
    (0.70, 0.70, 0.10),   # yellow-green
    (0.85, 0.35, 0.35),   # salmon
    (0.40, 0.20, 0.60),   # dark purple
    (0.20, 0.60, 0.40),   # forest
    (0.95, 0.60, 0.50),   # peach
    (0.30, 0.30, 0.80),   # slate-blue
]


# ==============================================================================
# Helpers
# ==============================================================================

def extract_category(instruction, dataset_type):
    """
    Try to match a known category from the instruction text.
    Falls back to the first few words if no known category matches.
    """
    cats = DATASET_CATEGORIES.get(dataset_type, [])
    inst_lower = instruction.lower()

    # Sort categories longest first so "ground track field" beats "field"
    sorted_cats = sorted(cats, key=len, reverse=True)
    for cat in sorted_cats:
        if cat.lower() in inst_lower:
            return cat

    # Fallback: return first 3-4 meaningful words as a label
    words = instruction.strip().split()
    return " ".join(words[:4]) if words else "object"


class CategoryColorMap:
    """Assigns a unique colour from the palette to each unique category."""

    def __init__(self):
        self._map = {}
        self._idx = 0

    def __call__(self, category):
        key = category.lower()
        if key not in self._map:
            self._map[key] = PALETTE_RGB[self._idx % len(PALETTE_RGB)]
            self._idx += 1
        return self._map[key]

    @property
    def legend_items(self):
        """Return list of (label, colour) for the legend."""
        return [(k.title(), v) for k, v in self._map.items()]


def overlay_mask_on_image(image_rgb, mask_binary, color_rgb, alpha=0.5):
    """
    Overlay a coloured semi-transparent mask on an RGB image.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8
    mask_binary : (H, W) 0/1 or bool
    color_rgb : (r, g, b) floats in [0, 1]
    alpha : overlay opacity

    Returns
    -------
    blended : (H, W, 3) uint8
    """
    h, w = image_rgb.shape[:2]
    mask = mask_binary.astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    color_uint8 = np.array(
        [int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255)],
        dtype=np.uint8,
    )

    overlay = image_rgb.copy()
    overlay[mask == 1] = color_uint8
    blended = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

    # draw a thin contour for clarity
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color_uint8.tolist(), 2)

    return blended


# -- Per-dataset Q/A & metadata extractors -------------------------------------

def _get_qa_lasers(dataset, idx):
    info = dataset.reason_file[idx]
    question = info["description"]
    answer = info["answer"]
    image_path = os.path.join(dataset.LaSeRS_image_path, info["image_name"])
    return question, answer, image_path


def _get_qa_refsegrs(dataset, idx):
    question = dataset.refs[idx]
    answer = "Sure, it is [SEG]."
    image_path = dataset.images[idx]
    return question, answer, image_path


def _get_qa_rrsisd(dataset, idx):
    question = dataset.refs[idx]
    answer = "Sure, it is [SEG]."
    image_path = dataset.images[idx]
    return question, answer, image_path


def _get_qa_risbench(dataset, idx):
    sample = dataset.risbench[idx]
    question = sample["phrase"]
    answer = "Sure, it is [SEG]."
    image_path = os.path.join(dataset._img_cache_dir, f"{idx}.jpg")
    return question, answer, image_path


def _get_qa_earthreason(dataset, idx):
    image_path, _, qa_path = dataset.samples[idx]
    with open(qa_path, "r") as f:
        QAs = json.load(f)
    question = QAs["questions"][0]
    answer = QAs["answer"][0] if QAs["answer"] else "No target object."
    return question, answer, image_path


def _get_qa_liss4reason(dataset, idx):
    image_path, _, qa_path = dataset.samples[idx]
    with open(qa_path, "r") as f:
        QAs = json.load(f)
    question = QAs["questions"][0]
    answer = QAs["answer"][0] if QAs["answer"] else "No target object."
    return question, answer, image_path


QA_EXTRACTORS = {
    "LaSeRS": _get_qa_lasers,
    "RefSegRS": _get_qa_refsegrs,
    "RRSISD": _get_qa_rrsisd,
    "RISBench": _get_qa_risbench,
    "EarthReason": _get_qa_earthreason,
    "LISS4Reason": _get_qa_liss4reason,
}


def load_original_image(image_path):
    """Load an image from disk as RGB uint8 (H, W, 3)."""
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ==============================================================================
# Dataset & split creation (mirrors eval.py logic)
# ==============================================================================

def create_dataset(dataset_type, base_data_path, tokenizer, data_args, split):
    """Instantiate the correct dataset class."""
    if dataset_type == "LaSeRS":
        json_folders = os.path.join(
            base_data_path, "rs_reason_seg/LaSeRS/test/annotations"
        )
        if os.path.isdir(json_folders):
            # Use the first JSON split found (same as eval.py)
            split_file = sorted(os.listdir(json_folders))[0]
        else:
            split_file = split
        return LaSeRSDataset(
            base_data_path=base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split_file,
        )
    elif dataset_type == "EarthReason":
        return EarthReasonDataset(
            base_data_path=base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )
    elif dataset_type == "RefSegRS":
        return RefSegRSDataset(
            base_data_path=base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )
    elif dataset_type == "RRSISD":
        return RRSISDDataset(
            base_data_path=base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )
    elif dataset_type == "LISS4Reason":
        return LISS4ReasonDataset(
            base_data_path=base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )
    elif dataset_type == "RISBench":
        return RISBenchDataset(
            base_data_path=base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )
    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type!r}. "
            f"Expected one of: LaSeRS, EarthReason, RefSegRS, RRSISD, RISBench, LISS4Reason"
        )


# ==============================================================================
# Figure builder
# ==============================================================================

def build_figure(rows, cmap, dataset_type, output_path, alpha=0.5):
    """
    Build and save the final matplotlib figure.

    Parameters
    ----------
    rows : list of dicts, each with keys:
        question, answer, image_rgb, gt_mask, pred_mask, category
    cmap : CategoryColorMap instance (already populated)
    dataset_type : str
    output_path : path to save the figure
    alpha : overlay opacity
    """
    n_rows = len(rows)
    col_labels = ["QA", "Image", "Ground Truth", "Pred Mask"]

    # width ratios: QA column is wider for text
    fig_height = 5.0 * n_rows + 1.5
    fig = plt.figure(figsize=(22, fig_height))
    gs = gridspec.GridSpec(
        n_rows + 1,          # +1 for header row
        4,
        width_ratios=[1.4, 1, 1, 1],
        hspace=0.12,
        wspace=0.06,
        top=0.94,
        bottom=0.06,
        left=0.02,
        right=0.98,
    )

    # -- Column headers --------------------------------------------------------
    for col_idx, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(
            0.5, 0.5, label,
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            ha="center", va="center",
        )
        ax.axis("off")

    # -- Rows ------------------------------------------------------------------
    for row_idx, row_data in enumerate(rows):
        color_rgb = cmap(row_data["category"])

        # .. QA text ...........................................................
        ax_qa = fig.add_subplot(gs[row_idx + 1, 0])
        q_text = textwrap.fill(row_data["question"], width=40)
        a_text = textwrap.fill(row_data["answer"], width=40)

        # Use colored Q: and A: labels
        ax_qa.text(
            0.05, 0.95,
            "",
            transform=ax_qa.transAxes,
            fontsize=8, va="top", ha="left",
        )
        # Build rich text with Q in blue, A in green
        ax_qa.text(
            0.05, 0.96,
            "Q: ",
            transform=ax_qa.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left",
            color="#1565C0",
        )
        ax_qa.text(
            0.05, 0.94,
            q_text,
            transform=ax_qa.transAxes,
            fontsize=7.5, va="top", ha="left",
            color="#333333",
            linespacing=1.4,
        )

        # Calculate vertical position for answer based on question length
        q_lines = q_text.count("\n") + 1
        a_y = 0.94 - (q_lines * 0.08) - 0.06

        ax_qa.text(
            0.05, a_y,
            "A: ",
            transform=ax_qa.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left",
            color="#2E7D32",
        )
        ax_qa.text(
            0.05, a_y - 0.02,
            a_text,
            transform=ax_qa.transAxes,
            fontsize=7.5, va="top", ha="left",
            color="#555555",
            linespacing=1.4,
        )

        # Light background for QA cell
        ax_qa.set_facecolor("#FAFAFA")
        for spine in ax_qa.spines.values():
            spine.set_edgecolor("#DDDDDD")
            spine.set_linewidth(0.8)
        ax_qa.set_xticks([])
        ax_qa.set_yticks([])

        # .. Original Image ....................................................
        ax_img = fig.add_subplot(gs[row_idx + 1, 1])
        ax_img.imshow(row_data["image_rgb"])
        ax_img.axis("off")

        # .. Ground Truth overlay ..............................................
        ax_gt = fig.add_subplot(gs[row_idx + 1, 2])
        gt_overlay = overlay_mask_on_image(
            row_data["image_rgb"], row_data["gt_mask"], color_rgb, alpha
        )
        ax_gt.imshow(gt_overlay)
        ax_gt.axis("off")

        # .. Predicted Mask overlay ............................................
        ax_pred = fig.add_subplot(gs[row_idx + 1, 3])
        pred_overlay = overlay_mask_on_image(
            row_data["image_rgb"], row_data["pred_mask"], color_rgb, alpha
        )
        ax_pred.imshow(pred_overlay)
        ax_pred.axis("off")

    # -- Legend ----------------------------------------------------------------
    legend_items = cmap.legend_items
    if legend_items:
        handles = [
            Patch(facecolor=color, edgecolor="black", label=label)
            for label, color in legend_items
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            fontsize=10,
            frameon=True,
            title=f"Categories  ({dataset_type})",
            title_fontsize=11,
        )

    fig.suptitle(
        f"SegEarth-R2 Predictions \u2014 {dataset_type}",
        fontsize=18, fontweight="bold", y=0.98,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n>>> Visualization saved to: {os.path.abspath(output_path)}")


# ==============================================================================
# CLI & Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize SegEarth-R2 predictions (GT vs Pred overlay)."
    )
    p.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the merged SegEarth-R2 model.",
    )
    p.add_argument(
        "--base_data_path", type=str, required=True,
        help="Root directory of the dataset.",
    )
    p.add_argument(
        "--dataset_type", type=str, required=True,
        choices=["LaSeRS", "EarthReason", "RefSegRS", "RRSISD",
                 "RISBench", "LISS4Reason"],
        help="Which dataset to visualize.",
    )
    p.add_argument(
        "--data_split", type=str, default="test",
        help="Dataset split: train / val / test  (default: test).",
    )
    p.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to visualize. "
             "If omitted the script will ask interactively.",
    )
    p.add_argument(
        "--output_path", type=str, default="output/visualization.png",
        help="Where to save the output figure (default: output/visualization.png).",
    )
    p.add_argument(
        "--mask_config", type=str,
        default="segearth_r2/model/mask_decoder/mask_config/"
                "maskformer2_swin_base_384_bs16_50ep.yaml",
        help="Path to mask decoder config YAML.",
    )
    p.add_argument(
        "--vision_tower", type=str,
        default="pretrained_model/CLIP",
        help="Path to CLIP / SigLIP vision tower.",
    )
    p.add_argument(
        "--overlay_alpha", type=float, default=0.45,
        help="Mask overlay opacity (0 = transparent, 1 = opaque). Default 0.45.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sample selection.",
    )
    return p.parse_args()


class _ModelArgs:
    """Minimal namespace expected by load_pretrained_model."""
    def __init__(self, mask_config, vision_tower):
        self.mask_config = mask_config
        self.vision_tower = vision_tower
        self.image_aspect_ratio = "square"
        self.image_grid_pinpoints = None
        self.model_map_name = "segearth_r2"
        self.version = "llava_phi"


def main():
    args = parse_args()

    # -- Interactive prompt for num_samples ------------------------------------
    if args.num_samples is None:
        while True:
            try:
                n = int(input("How many samples to visualize? (e.g. 3, 5, 10): "))
                if n < 1:
                    raise ValueError
                args.num_samples = n
                break
            except ValueError:
                print("Please enter a positive integer.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Load model ------------------------------------------------------------
    print("\n---------- Loading Model ----------")
    model_args = _ModelArgs(args.mask_config, args.vision_tower)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        os.path.expanduser(args.model_path),
        model_args=model_args,
        mask_config=args.mask_config,
        device=str(device),
    )
    model.to(dtype=torch.float16, device=device)
    model.eval()
    print("---------- Model Loaded ----------\n")

    # -- Prepare data pipeline -------------------------------------------------
    data_args_ns = argparse.Namespace(
        base_data_path=args.base_data_path,
        is_multimodal=True,
        image_aspect_ratio="square",
        image_grid_pinpoints=None,
        version="llava_phi",
    )
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        "llava_phi"
    ]
    clip_image_processor = SiglipImageProcessor.from_pretrained(args.vision_tower)
    data_collator = DataCollatorForCOCODatasetV2(
        tokenizer=tokenizer, clip_image_processor=clip_image_processor
    )

    # -- Create dataset --------------------------------------------------------
    print(f"Loading dataset: {args.dataset_type} (split={args.data_split})")
    dataset = create_dataset(
        args.dataset_type, args.base_data_path,
        tokenizer, data_args_ns, args.data_split,
    )
    total = len(dataset)
    print(f"Dataset size: {total}")

    if args.num_samples > total:
        print(
            f"WARNING: requested {args.num_samples} samples but dataset only "
            f"has {total}. Using all {total}."
        )
        args.num_samples = total

    # -- Select random sample indices (that have valid masks) ------------------
    qa_extractor = QA_EXTRACTORS[args.dataset_type]

    candidate_indices = list(range(total))
    random.shuffle(candidate_indices)

    selected_rows = []
    cmap = CategoryColorMap()

    print(f"\nRunning inference on {args.num_samples} samples ...")
    tried = 0
    for idx in candidate_indices:
        if len(selected_rows) >= args.num_samples:
            break
        tried += 1

        # -- 1. Get Q/A text & image path BEFORE touching the data_dict --------
        try:
            question, answer, image_path = qa_extractor(dataset, idx)
        except Exception as e:
            print(f"  [skip idx={idx}] Q/A extraction error: {e}")
            continue

        # -- 2. Get data_dict from the dataset ---------------------------------
        try:
            data_dict = dataset[idx]
        except Exception as e:
            print(f"  [skip idx={idx}] Dataset __getitem__ error: {e}")
            continue

        mask_num = data_dict.get("mask_num", 0)
        if mask_num == 0:
            # no mask -> skip
            continue

        # -- 3. Collate into a batch of 1 --------------------------------------
        #   The collator mutates data_dict in-place (deletes input_ids, labels,
        #   image).  That's OK - we already captured what we need above.
        try:
            batch = data_collator([data_dict])
        except Exception as e:
            print(f"  [skip idx={idx}] Collation error: {e}")
            continue

        # -- 4. Run inference --------------------------------------------------
        with torch.no_grad():
            inputs_gpu = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            inputs_gpu["token_refer_id"] = [
                ids.to(device) for ids in batch["token_refer_id"]
            ]

            try:
                outputs = model.eval_seg(
                    input_ids=inputs_gpu["input_ids"],
                    attention_mask=inputs_gpu["attention_mask"],
                    images=inputs_gpu["images"].to(dtype=torch.float16),
                    images_clip=inputs_gpu["images_clip"].to(dtype=torch.float16),
                    seg_info=inputs_gpu["seg_info"],
                    token_refer_id=inputs_gpu["token_refer_id"],
                    SEG_token_embedding_indices=inputs_gpu[
                        "SEG_token_embedding_indices"
                    ],
                    labels=inputs_gpu["labels"],
                    mask_num=inputs_gpu["mask_num"],
                )
            except Exception as e:
                print(f"  [skip idx={idx}] Inference error: {e}")
                continue

        if not outputs:
            continue

        result = outputs[0]  # first (and only) output in the batch
        pred_np = result["pred"]
        gt_np = result["gt"]

        if gt_np is None:
            continue

        # Convert from 0/255 uint8 to 0/1 binary
        pred_bin = (pred_np > 0).astype(np.uint8).squeeze()
        gt_bin = (gt_np > 0).astype(np.uint8).squeeze()

        # -- 5. Load original image --------------------------------------------
        try:
            image_rgb = load_original_image(image_path)
        except Exception as e:
            print(f"  [skip idx={idx}] Image load error: {e}")
            continue

        # -- 6. Extract category -----------------------------------------------
        instruction_text = question  # the referring instruction
        category = extract_category(instruction_text, args.dataset_type)
        _ = cmap(category)  # register colour

        selected_rows.append({
            "question": question,
            "answer": answer,
            "image_rgb": image_rgb,
            "gt_mask": gt_bin,
            "pred_mask": pred_bin,
            "category": category,
        })

        print(
            f"  [{len(selected_rows)}/{args.num_samples}]  idx={idx}  "
            f"category=\"{category}\""
        )

    if not selected_rows:
        print("\nERROR: Could not find any valid samples with masks. Exiting.")
        sys.exit(1)

    # -- Build & save figure ---------------------------------------------------
    print(f"\nBuilding figure with {len(selected_rows)} rows ...")
    build_figure(
        rows=selected_rows,
        cmap=cmap,
        dataset_type=args.dataset_type,
        output_path=args.output_path,
        alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
