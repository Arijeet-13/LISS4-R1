import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import cv2
import torch
import numpy as np
from enum import Enum
from tqdm import tqdm
import transformers
from typing import Optional
from dataclasses import dataclass, field
from transformers import SiglipImageProcessor
import torch.distributed as distributed
import zipfile

from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset, EarthReasonDataset, RefSegRSDataset, RRSISDDataset, LISS4ReasonDataset

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def compute_metric(intersection_meter, union_meter, acc_iou_meter, pr_meters,
                    pred_mask, gt_mask, ignore_index=255):
    intersection, union, _ = intersectionAndUnionGPU(
        pred_mask, gt_mask, K=2, ignore_index=ignore_index
    )
    intersection, union = intersection.cpu().numpy(), union.cpu().numpy()

    acc_iou = intersection / (union + 1e-5)
    acc_iou[union == 0] = 1.0
    foreground_iou = acc_iou[1]

    intersection_meter.update(intersection)
    union_meter.update(union)
    acc_iou_meter.update(acc_iou, n=1)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        pr_meters[threshold].update(1.0 if foreground_iou > threshold else 0.0, n=1)
    

_DENORM_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_DENORM_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _load_original_image(seg, images_clip_tensor=None):
    candidate_keys = ["image_path", "img_path", "file_name", "image_file", "path"]
    for key in candidate_keys:
        if key in seg and seg[key]:
            img_path = seg[key]
            if os.path.isfile(img_path):
                img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    return img_bgr

    if images_clip_tensor is not None:
        img = images_clip_tensor.detach().cpu().float().numpy()
        if img.ndim == 4:  # (B, C, H, W) -> take first
            img = img[0]
        img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C
        img = (img * _DENORM_STD) + _DENORM_MEAN
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr

    return None


def save_overlay_visualization(image_bgr, pred_bin, save_path,
                                color_bgr=(0, 0, 255), alpha=0.5):
    if image_bgr is None:
        return False

    mask = pred_bin.astype(np.uint8)
    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    overlay = image_bgr.copy()
    overlay[mask == 1] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    cv2.imwrite(save_path, blended)
    return True


@dataclass
class Arguments:
    local_rank: int = 0

    vision_tower: str = "pretrained_model/CLIP"
    vision_tower_mask: str = "pretrained_model/mask2former/model_final_54b88a.pkl"

    lazy_preprocess: bool = False
    base_data_path: Optional[str] = field(default='your_data_path')

    model_path: Optional[str] = field(default="SegEarthR2_LaSeRS/hfweights-50000")
    mask_config: Optional[str] = field(default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'segearth_r2'
    version: str = 'llava_phi'

    temperature: float = 0.2
    num_beams: int = 1
    max_new_tokens: int = 128
    do_sample: bool = True

    output_dir: str = 'save_folder'
    dataloader_num_workers: int = 8
    dataset_type: str = field(default='LaSeRS')
    data_split: str = field(default='test')
    ignore_index: int = 255

    save_masks: bool = True
    save_overlay: bool = True          # toggle overlay visualization
    overlay_alpha: float = 0.5         # overlay opacity


def zip_folder(folder_path):
    folder_path = os.path.abspath(folder_path)
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))


def main():
    parser = transformers.HfArgumentParser(Arguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    model_path = os.path.expanduser(data_args.model_path)

    print("---------- Initializing Model ----------")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_args=data_args,
        mask_config=data_args.mask_config,
        device="cuda"
    )

    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu")
    model.to(dtype=torch.float16, device=device)
    model.eval()

    print("---------- Model Initialization Complete ----------")


    # --- DETERMINISM CHECK ---
    not_in_eval = [name for name, module in model.named_modules() if module.training]
    if not_in_eval:
        print(f"WARNING: {len(not_in_eval)} submodules still in TRAIN mode:")
        for n in not_in_eval[:30]:
            print(f"  - {n}")
    else:
        print("All submodules confirmed in eval mode.")

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d,
                                torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            status = "TRAIN (ACTIVE!)" if module.training else "eval"
            print(f"{name}: {type(module).__name__} -> {status}")
    # --- END DETERMINISM CHECK (part 1) ---

    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)

    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    if data_args.dataset_type == 'LaSeRS': #ADDED Setup for EarthReason
        json_folders = os.path.join(data_args.base_data_path, 'rs_reason_seg/LaSeRS/test/annotations')
        splits = os.listdir(json_folders)
    elif data_args.dataset_type == 'EarthReason':
        splits = [data_args.data_split]
    elif data_args.dataset_type == 'RefSegRS':
        splits = [data_args.data_split]
    elif data_args.dataset_type == 'RRSISD':
        splits = [data_args.data_split]
    elif data_args.dataset_type == 'LISS4Reason':
        splits = [data_args.data_split]
    else:
        raise ValueError(f"Unknown dataset_type: {data_args.dataset_type!r} (expected 'LaSeRS', 'EarthReason', 'RefSegRS', 'LISS4Reason' or 'RRSISD')")
        # save_folder = data_args.output_dir
    for split in splits:
        if data_args.dataset_type == 'LaSeRS':
            eval_dataset = LaSeRSDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=split)
        elif data_args.dataset_type == 'RefSegRS': #RefSegRS
            eval_dataset = RefSegRSDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=split)
        elif data_args.dataset_type == 'RRSISD': #RRSISD
            eval_dataset = RRSISDDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=split)
        elif data_args.dataset_type == 'LISS4Reason': #LISS4
            eval_dataset = LISS4ReasonDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=split)
        else:  # 'EarthReason'
            eval_dataset = EarthReasonDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=split)

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=data_args.dataloader_num_workers,
            pin_memory=False,
            sampler=None,
            collate_fn=data_collator)

        # --- DETERMINISM CHECK (part 2): run one sample twice ---
        # if split == splits[0]:  # only do this once, on the first split
        #     sample_inputs = next(iter(eval_dataloader))
        #     sample_inputs_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_inputs.items()}
        #     sample_inputs_gpu['token_refer_id'] = [ids.to(device) for ids in sample_inputs['token_refer_id']]
        if split == splits[0]:
            sample_inputs = None
            for candidate in eval_dataloader:
                if candidate['mask_num'][0] > 0 and len(candidate['seg_info']) > 0:
                    sample_inputs = candidate
                    break
            if sample_inputs is None:
                    print("No valid (non-empty) sample found for determinism check.")
            else:
                    sample_inputs_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_inputs.items()}
                    sample_inputs_gpu['token_refer_id'] = [ids.to(device) for ids in sample_inputs['token_refer_id']]

            with torch.no_grad():
                out1 = model.eval_seg(
                    input_ids=sample_inputs_gpu['input_ids'],
                    attention_mask=sample_inputs_gpu['attention_mask'],
                    images=sample_inputs_gpu['images'].to(dtype=torch.float16),
                    images_clip=sample_inputs_gpu['images_clip'].to(dtype=torch.float16),
                    seg_info=sample_inputs_gpu['seg_info'],
                    token_refer_id=sample_inputs_gpu['token_refer_id'],
                    SEG_token_embedding_indices=sample_inputs_gpu['SEG_token_embedding_indices'],
                    labels=sample_inputs_gpu['labels'],
                    mask_num=sample_inputs_gpu['mask_num'],
                )
                out2 = model.eval_seg(
                    input_ids=sample_inputs_gpu['input_ids'],
                    attention_mask=sample_inputs_gpu['attention_mask'],
                    images=sample_inputs_gpu['images'].to(dtype=torch.float16),
                    images_clip=sample_inputs_gpu['images_clip'].to(dtype=torch.float16),
                    seg_info=sample_inputs_gpu['seg_info'],
                    token_refer_id=sample_inputs_gpu['token_refer_id'],
                    SEG_token_embedding_indices=sample_inputs_gpu['SEG_token_embedding_indices'],
                    labels=sample_inputs_gpu['labels'],
                    mask_num=sample_inputs_gpu['mask_num'],
                )
            pred1, pred2 = out1[0]['pred'], out2[0]['pred']
            print("Determinism check — identical outputs:", np.array_equal(pred1, pred2))
            print("Differing pixels:", np.sum(pred1 != pred2), "/", pred1.size)
        # --- END DETERMINISM CHECK (part 2) ---


        do_eval(model, eval_dataloader, split, data_args, device)


def do_eval(model, eval_dataloader, split, data_args, device):
    model.eval()

    save_dir = os.path.join(data_args.output_dir, split)
    if data_args.save_masks:
        os.makedirs(save_dir, exist_ok=True)

    overlay_dir = os.path.join(data_args.output_dir, split, "overlays")
    if data_args.save_overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    pr_meters = {t: AverageMeter(f"Pr@{t}", ":6.3f", Summary.AVERAGE) for t in thresholds}

    n_skipped_empty = 0
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            mask_num = inputs['mask_num'][0]
            if mask_num == 0 or len(inputs['seg_info']) == 0:
                n_skipped_empty += 1
                continue

            inputs_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            inputs_gpu['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]

            outputs = model.eval_seg(
                input_ids=inputs_gpu['input_ids'],
                attention_mask=inputs_gpu['attention_mask'],
                images=inputs_gpu['images'].to(dtype=torch.float16),
                images_clip=inputs_gpu['images_clip'].to(dtype=torch.float16),
                seg_info=inputs_gpu['seg_info'],
                token_refer_id=inputs_gpu['token_refer_id'],
                SEG_token_embedding_indices=inputs_gpu['SEG_token_embedding_indices'],
                labels=inputs_gpu['labels'],
                mask_num=inputs_gpu['mask_num'],
            )

            for output, seg in zip(outputs, inputs['seg_info']):
                pred_np = output['pred']
                gt_np = output['gt']

                pred_bin = (pred_np > 0).squeeze() #.astype(np.int64)
                gt_bin = (gt_np > 0).squeeze() #.astype(np.int64)
                if pred_bin.ndim != 2 or gt_bin.ndim != 2:
                    print(f"WARNING: unexpected shape — pred {pred_bin.shape}, gt {gt_bin.shape}, image {seg.get('image_id')}")
                pred_t = torch.from_numpy(pred_bin).to(device)
                gt_t = torch.from_numpy(gt_bin).to(device)

                compute_metric(
                    intersection_meter, union_meter, acc_iou_meter, pr_meters,
                    pred_t, gt_t, ignore_index=data_args.ignore_index
                )
                print(f"pred_sum={pred_t.sum().item()}, gt_sum={gt_t.sum().item()}, overlap={(pred_t & gt_t).sum().item()}")

                image_id = seg['image_id']
                mask_id = seg['mask_id']

                if data_args.save_masks:
                    pred_path = os.path.join(save_dir, f"{image_id}_mask{mask_id}_pred.png")
                    gt_path = os.path.join(save_dir, f"{image_id}_mask{mask_id}_gt.png")
                    cv2.imwrite(pred_path, pred_bin.astype(np.uint8) * 255)
                    cv2.imwrite(gt_path, gt_bin.astype(np.uint8) * 255)

                if data_args.save_overlay:
                    original_img = _load_original_image(
                        seg, images_clip_tensor=inputs_gpu['images_clip']
                    )
                    overlay_path = os.path.join(overlay_dir, f"{image_id}_mask{mask_id}_overlay.png")
                    save_overlay_visualization(
                        original_img, pred_bin, overlay_path,
                        color_bgr=(0, 0, 255),  # red in BGR
                        alpha=data_args.overlay_alpha,
                    )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    cIoU = iou_class[1]
    gIoU = acc_iou_meter.avg[1]

    print(f"\n===== {split} =====")
    print(f"Instances evaluated: {acc_iou_meter.count}")
    if n_skipped_empty:
        print(f"Skipped (empty-target) samples: {n_skipped_empty}")
    print(f"gIoU (mean per-instance IoU): {gIoU:.4f}")
    print(f"cIoU (cumulative I/U):        {cIoU:.4f}")
    print("IoU Thresholds: " + ", ".join([f"@{t}: {m.avg:.4f}" for t, m in pr_meters.items()]))

    if data_args.save_masks:
        zip_folder(save_dir)
        print(f"Saved masks zipped to {save_dir}.zip")

    if data_args.save_overlay:
        zip_folder(overlay_dir)
        print(f"Saved overlay visualizations zipped to {overlay_dir}.zip")


if __name__ == "__main__":
    main()