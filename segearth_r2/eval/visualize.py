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
import re

# -- project root on sys.path -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
# 20 visually distinct colours (RGB 0-1) - high-contrast on white & imagery
# ==============================================================================
PALETTE_RGB = [
    (0.86, 0.15, 0.15),   # red (#DC2626)
    (0.09, 0.64, 0.29),   # green (#16A34A)
    (0.15, 0.39, 0.92),   # blue (#2563EB)
    (0.85, 0.46, 0.02),   # amber / gold (#D97706)
    (0.58, 0.20, 0.92),   # purple (#9333EA)
    (0.05, 0.58, 0.53),   # teal (#0D9488)
    (0.88, 0.11, 0.28),   # rose (#E11D48)
    (0.92, 0.35, 0.05),   # orange (#EA580C)
    (0.31, 0.27, 0.90),   # indigo (#4F46E5)
    (0.75, 0.07, 0.24),   # crimson (#BE123C)
    (0.40, 0.64, 0.05),   # lime (#65A30D)
    (0.71, 0.33, 0.04),   # brown (#B45309)
    (0.49, 0.23, 0.93),   # violet (#7C3AED)
    (0.03, 0.57, 0.70),   # dark cyan (#0891B2)
    (0.75, 0.15, 0.83),   # magenta (#C026D3)
    (0.52, 0.30, 0.05),   # olive (#854D0E)
    (0.28, 0.33, 0.41),   # slate (#475569)
    (0.96, 0.25, 0.37),   # coral (#F43F5E)
    (0.02, 0.59, 0.41),   # emerald (#059669)
    (0.76, 0.25, 0.05),   # deep orange (#C2410C)
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

# ==============================================================================
# QA Formatting & Typography (matches paper layout)
# ==============================================================================

def format_qa_for_display(raw_question, raw_answer, category, sample_idx):
    """
    Format Question and Answer to match the publication figure style:
    - If question is a short phrase ('large harbor'), format as 'Can you locate the large harbor?'
    - If answer doesn't have '<p> ... </p>', format with '<p> {category} </p> [SEG]'.
    """
    q = raw_question.strip()
    q_lower = q.lower()
    is_already_sentence = any(
        q_lower.startswith(prefix)
        for prefix in [
            "can you", "could you", "please", "what", "where", "how",
            "which", "find", "locate", "segment", "is there", "identify",
        ]
    ) or (len(q) > 40 and q.endswith((".", "?", "!")))

    if not is_already_sentence:
        # It's a short referring expression like "large harbor" or "vehicle on the lower right"
        if q_lower.startswith(("the ", "a ", "an ")):
            q_formatted = f"Can you locate {q}."
        else:
            q_formatted = f"Can you locate the {q}."
    else:
        q_formatted = q

    # Normalize answer
    if "<p>" in raw_answer and "</p>" in raw_answer:
        a_formatted = raw_answer.strip()
    else:
        templates = [
            "Of course! The <p> {cat} </p> [SEG] segmentation completed.",
            "Sure! The <p> {cat} </p> [SEG] area is here.",
            "Sure, I have segmented the <p> {cat} </p> [SEG] area.",
        ]
        a_formatted = templates[sample_idx % len(templates)].format(cat=category)

    return q_formatted, a_formatted


def _get_font_bundle(size=24):
    """
    Load serif fonts (regular, bold, bold-italic) bundled with matplotlib
    so this works out-of-the-box on Linux (Kaggle/Colab), Windows, Mac.
    """
    import matplotlib
    font_dir = os.path.join(
        os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"
    )

    cand_reg = os.path.join(font_dir, "DejaVuSerif.ttf")
    cand_bold = os.path.join(font_dir, "DejaVuSerif-Bold.ttf")
    cand_italic = os.path.join(font_dir, "DejaVuSerif-BoldItalic.ttf")

    if os.path.isfile(cand_reg) and os.path.isfile(cand_bold):
        f_reg = ImageFont.truetype(cand_reg, size)
        f_bold = ImageFont.truetype(cand_bold, size)
        f_italic = ImageFont.truetype(
            cand_italic if os.path.isfile(cand_italic) else cand_bold, size
        )
        return f_reg, f_bold, f_italic

    # Fallback: DejaVuSans
    cand_reg = os.path.join(font_dir, "DejaVuSans.ttf")
    cand_bold = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
    cand_italic = os.path.join(font_dir, "DejaVuSans-BoldOblique.ttf")
    if os.path.isfile(cand_reg) and os.path.isfile(cand_bold):
        f_reg = ImageFont.truetype(cand_reg, size)
        f_bold = ImageFont.truetype(cand_bold, size)
        f_italic = ImageFont.truetype(
            cand_italic if os.path.isfile(cand_italic) else cand_bold, size
        )
        return f_reg, f_bold, f_italic

    f = ImageFont.load_default()
    return f, f, f


def render_qa_cell(
    question_text,
    answer_text,
    category_color_rgb,
    width=560,
    height=512,
    base_font_size=24,
):
    """
    Renders the QA column as a clean, publication-ready image:
    - Pure white background with no outer border box
    - 'Q:' in bold serif
    - Question text line-wrapped
    - 'A:' in bold serif
    - '<p> category </p>' and '[SEG]' in bold italic serif with class color
    - Text centered vertically relative to the image row
    """
    color_uint8 = (
        int(category_color_rgb[0] * 255),
        int(category_color_rgb[1] * 255),
        int(category_color_rgb[2] * 255),
    )
    black_color = (15, 15, 15)

    for font_size in [base_font_size, base_font_size - 3, base_font_size - 6]:
        f_reg, f_bold, f_italic = _get_font_bundle(font_size)
        pad_x = 24
        max_w = width - pad_x * 2
        line_h = int(font_size * 1.45)

        tokens = [
            ("Q: ", f_bold, (0, 0, 0)),
            (question_text, f_reg, black_color),
            ("\n\n", None, None),
            ("A: ", f_bold, (0, 0, 0)),
        ]

        parts = re.split(r"(<p>.*?</p>|\[SEG\])", answer_text)
        for p in parts:
            if not p:
                continue
            if p.startswith("<p>") or p == "[SEG]":
                tokens.append((p + " ", f_italic, color_uint8))
            else:
                tokens.append((p.strip() + " ", f_reg, black_color))

        test_img = Image.new("RGB", (width, height), (255, 255, 255))
        test_draw = ImageDraw.Draw(test_img)

        lines = []
        cur_line = []
        cur_w = 0

        for text, font, color in tokens:
            if text == "\n\n":
                if cur_line:
                    lines.append((cur_line, False))
                    cur_line = []
                    cur_w = 0
                lines.append(([], True))
                continue

            words = text.split(" ")
            for i, word in enumerate(words):
                if not word and i > 0:
                    continue
                word_sp = word + " " if i < len(words) - 1 else word
                bbox = test_draw.textbbox((0, 0), word_sp, font=font)
                w = bbox[2] - bbox[0]
                if cur_w + w > max_w and cur_line:
                    lines.append((cur_line, False))
                    cur_line = []
                    cur_w = 0
                cur_line.append((word_sp, font, color, w))
                cur_w += w

        if cur_line:
            lines.append((cur_line, False))

        total_h = sum(
            line_h if not is_blank else int(line_h * 0.7)
            for _, is_blank in lines
        )
        if total_h <= height - 30:
            break

    out_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(out_img)

    start_y = max(24, int((height - total_h) * 0.45))
    y = start_y
    for cur_line, is_blank in lines:
        if is_blank:
            y += int(line_h * 0.7)
            continue
        x = pad_x
        for word_sp, font, color, w in cur_line:
            draw.text((x, y), word_sp, font=font, fill=color)
            x += w
        y += line_h

    return np.array(out_img)


# ==============================================================================
# Figure builder
# ==============================================================================

def build_figure(rows, cmap, dataset_type, output_path, alpha=0.45):
    """
    Build and save the final publication-quality figure.

    Columns:
        QA  |  Image  |  Ground Truth  |  Pred Mask

    No suptitle at top, no legend at bottom.
    """
    n_rows = len(rows)
    col_labels = ["QA", "Image", "Ground Truth", "Pred Mask"]

    # Target height per row in figure: 4.4 inches
    fig_height = 4.4 * n_rows + 0.8
    fig = plt.figure(figsize=(19, fig_height))
    gs = gridspec.GridSpec(
        n_rows + 1,
        4,
        height_ratios=[0.22] + [1.0] * n_rows,
        width_ratios=[1.15, 1.0, 1.0, 1.0],
        hspace=0.08,
        wspace=0.04,
        top=0.96,
        bottom=0.04,
        left=0.02,
        right=0.98,
    )

    # -- Column headers (very top of figure, NO SUPTITLE) ----------------------
    for col_idx, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(
            0.5, 0.5, label,
            transform=ax.transAxes,
            fontsize=22, fontweight="bold",
            ha="center", va="center",
            family="sans-serif",
        )
        ax.axis("off")

    # -- Rows ------------------------------------------------------------------
    for row_idx, row_data in enumerate(rows):
        color_rgb = cmap(row_data["category"])
        img_rgb = row_data["image_rgb"]

        cell_h = 512
        cell_w = int(cell_h * 1.15)

        # 1. QA Cell
        ax_qa = fig.add_subplot(gs[row_idx + 1, 0])
        qa_cell = render_qa_cell(
            question_text=row_data["question"],
            answer_text=row_data["answer"],
            category_color_rgb=color_rgb,
            width=cell_w,
            height=cell_h,
            base_font_size=24,
        )
        ax_qa.imshow(qa_cell)
        ax_qa.axis("off")

        # 2. Original Image
        ax_img = fig.add_subplot(gs[row_idx + 1, 1])
        ax_img.imshow(img_rgb)
        ax_img.axis("off")

        # 3. Ground Truth overlay
        ax_gt = fig.add_subplot(gs[row_idx + 1, 2])
        gt_overlay = overlay_mask_on_image(
            img_rgb, row_data["gt_mask"], color_rgb, alpha
        )
        ax_gt.imshow(gt_overlay)
        ax_gt.axis("off")

        # 4. Predicted Mask overlay
        ax_pred = fig.add_subplot(gs[row_idx + 1, 3])
        pred_overlay = overlay_mask_on_image(
            img_rgb, row_data["pred_mask"], color_rgb, alpha
        )
        ax_pred.imshow(pred_overlay)
        ax_pred.axis("off")

    # Clean figure ending (NO LEGEND, NO SUPTITLE)
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
        "--version", type=str, default="v0",
        help="Conversation template version (default: v0). "
             "Must match the version used during training.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sample selection.",
    )
    return p.parse_args()


class _ModelArgs:
    """Minimal namespace expected by load_pretrained_model."""
    def __init__(self, mask_config, vision_tower, version="v0"):
        self.mask_config = mask_config
        self.vision_tower = vision_tower
        self.image_aspect_ratio = "square"
        self.image_grid_pinpoints = None
        self.model_map_name = "segearth_r2"
        self.version = version


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
    model_args = _ModelArgs(args.mask_config, args.vision_tower, args.version)
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
        version=args.version,
    )
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.version
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

        # -- 6. Extract category & format QA for publication display -----------
        instruction_text = question  # the referring instruction
        category = extract_category(instruction_text, args.dataset_type)
        _ = cmap(category)  # register colour

        question_disp, answer_disp = format_qa_for_display(
            question, answer, category, len(selected_rows)
        )

        selected_rows.append({
            "question": question_disp,
            "answer": answer_disp,
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
