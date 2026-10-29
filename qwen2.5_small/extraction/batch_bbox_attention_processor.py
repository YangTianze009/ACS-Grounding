# -*- coding: utf-8 -*-
import os
import re
import math
import json
import pandas as pd
import time
import argparse
from PIL import Image, ImageDraw
from datetime import datetime
import traceback
# Import the bbox attention extractor function
from bbox_attention_extractor import extract_bbox_attention_for_evaluation
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
import pandas as pd
from PIL import Image
import json
import pandas as pd
import numpy as np
import random


def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多张GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process images with Qwen2.5-VL model")
    
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=100,
                       help="Top-k sampling parameter (default: 100)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter (default: 0.9)")
    parser.add_argument("--num_responses", type=int, default=10,
                       help="Number of responses per image (default: 10)")
    parser.add_argument("--device", type=str, default="0",
                       help="CUDA device ID (default: '0')")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed in this experiment")
    parser.add_argument("--object_num", type=str, default="single",
                    help="select single or multiple objects to do experiments")
    parser.add_argument("--data_type", type=str, default="testing",
                       help="select training or testing data")
    parser.add_argument("--dataset", type=str, default="COCO",
                    help="select dataset (COCO or Objects365)")

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="HF hub id or local path of the LVLM")
    parser.add_argument("--data_dir", type=str, default="../../dataset_splits",
                       help="folder with the split CSVs (dataset_splits/) and the images (see README)")
    
    return parser.parse_args()

# Parse arguments first to set device
args = parse_args()

# Set GPU device based on argument
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# Load model and processor
print("Loading Qwen2.5-VL model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(args.model_path)
print("Model loaded successfully!")

# =========================
# Configuration
# =========================
def setup_config(args):
    """Setup configuration based on command line arguments"""
    config = {
        'object_size': "small",  # "small" or "large"
        'num_responses': args.num_responses,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'seed': args.seed,
        'object_num': args.object_num,
        'dataset': args.dataset
    }
    if config['dataset'] == "COCO":
        if args.data_type == "training":
            config['csv_path'] = f"{args.data_dir}/COCO/coco_small_{args.object_num}_object_training.csv"
            config['image_dir'] = f"{args.data_dir}/COCO/train2017"
        elif args.data_type == "testing":
            config['csv_path'] = f"{args.data_dir}/COCO/coco_small_{args.object_num}_object_testing.csv"
            config['image_dir'] = f"{args.data_dir}/COCO/val2017"
    
    elif config['dataset'] == "Objects365":
            config['csv_path'] = f"{args.data_dir}/Objects365/object365_single_object_testing.csv"
            config['image_dir'] = f"{args.data_dir}/Objects365/images"
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Must be 'COCO' or 'Objects365'")
    
    # Output directory for attention pkl files
    if args.temperature > 0:
        config['output_root'] = f"./model_response/{args.dataset}/{args.data_type}/{config['object_num']}_object/bbox_attention_temp_{config['temperature']}_top_p_{config['top_p']}_response_{config['num_responses']}"
    elif args.temperature <= 0:
        config['output_root'] = f"./model_response/{args.dataset}/{args.data_type}/{config['object_num']}_object/bbox_attention_greedy_response_1"
    os.makedirs(config['output_root'], exist_ok=True)
    
    # Log file
    config['log_file'] = os.path.join(config['output_root'], f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    return config

# =========================
# Utilities
# =========================
def generate_prompt_2(object_name):
    prompt = f"""Detect all instances of {object_name} in the image:

    Your task is to detect any {object_name} objects that is visible in the image.
    
    For each detected object, provide:
    The bounding box coordinates [x1, y1, x2, y2]
    
    Output format (as a JSON array):
    [
      {{"bbox_2d": [x1, y1, x2, y2], "label": "{object_name}"}},
      {{"bbox_2d": [x1, y1, x2, y2], "label": "{object_name}"}}
    ]
    
    Important: Focus on any {object_name} that might be present in the image, especially those that are still visible and NOT masked out.
    
    Example output:
    [
      {{"bbox_2d": [200, 150, 280, 220], "label": "{object_name}"}},
      {{"bbox_2d": [350, 180, 420, 270], "label": "{object_name}"}}
    ]
    """
    return prompt

def _to_int(x):
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return int(round(val))
    except Exception:
        return None

def extract_xyxy_all(text):
    """
    从任意文本中提取所有形如 [x1, y1, x2, y2] 的框坐标（预测框）。
    """
    pattern = re.compile(
        r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]?'
    )
    boxes = []
    for m in pattern.finditer(text):
        x1 = _to_int(m.group(1))
        y1 = _to_int(m.group(2))
        x2 = _to_int(m.group(3))
        y2 = _to_int(m.group(4))
        if None not in (x1, y1, x2, y2):
            boxes.append([x1, y1, x2, y2])
    return boxes

def clip_and_validate(boxes, W, H, verbose_prefix=""):
    """
    坐标裁剪到图像范围；过滤 x2<=x1 或 y2<=y1 的无效框。
    """
    valid = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cx1 = max(0, min(int(x1), W - 1))
        cy1 = max(0, min(int(y1), H - 1))
        cx2 = max(0, min(int(x2), W - 1))
        cy2 = max(0, min(int(y2), H - 1))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        valid.append([cx1, cy1, cx2, cy2])
    return valid

def xywh_list_from_row(row):
    """
    从该行中找出形如 [[x,y,w,h], ...] 的字符串列并解析为 list。
    """
    for col in row.index:
        val = row[col]
        if isinstance(val, str) and val.strip().startswith('[['):
            try:
                return json.loads(val)
            except Exception:
                pass
    return []

def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]

def boxes_xyxy_from_xywh_list(xywh_list):
    return [xywh_to_xyxy(b) for b in xywh_list]

def log_message(message, log_file, print_to_console=True):
    """Log message to both file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if print_to_console:
        print(message)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

# =========================
# Main Processing
# =========================
def process_single_image(row, model, processor, process_vision_info, config, response_id=1):
    """Process a single image and extract attention data for one response"""
    image_filename = row['image_filename']
    category_name = row['category_name'] if 'category_name' in row.index else str(row.iloc[1])
    
    # Create output directory and filename structure
    stem, _ = os.path.splitext(image_filename)
    safe_cat = str(category_name).replace(" ", "_").replace("/", "_")
    
    # Create directory: basename_category
    output_dir = os.path.join(config['output_root'], f"{stem}_{safe_cat}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename: response_X.pkl
    output_filename = f"response_{response_id}.pkl"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        log_message(f"Skipping {image_filename} response {response_id} (already processed)", config['log_file'])
        return True
    
    log_message(f"Processing {image_filename} - {category_name} - Response {response_id}", config['log_file'])
    
    # Parse GT boxes
    xywh_list = xywh_list_from_row(row)
    if not xywh_list:
        log_message(f"No GT boxes found for {image_filename}", config['log_file'])
        return False
    
    # Filter: only process images with exactly 1 GT bbox
    # if len(xywh_list) != 1:
    #     log_message(f"Skipping {image_filename} - has {len(xywh_list)} GT boxes (need exactly 1)", config['log_file'])
    #     return None  # Use None to indicate filtered out
    
    gt_xyxy_raw = boxes_xyxy_from_xywh_list(xywh_list)
    
    # Load image
    image_path = os.path.join(config['image_dir'], image_filename)
    if not os.path.exists(image_path):
        log_message(f"Image not found: {image_path}", config['log_file'])
        return False
    
    try:
        base_image = Image.open(image_path).convert("RGB")
        W, H = base_image.size
        
        # Clip GT boxes
        gt_boxes = gt_xyxy_raw
        # gt_boxes = clip_and_validate(gt_xyxy_raw, W, H)
        
        # Generate prompt
        prompt = generate_prompt_2(category_name)
        print(prompt)
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate response
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            do_sample=config['temperature'] > 0,
            temperature=config['temperature'],
            top_k=config['top_k'],
            top_p=config['top_p'],
            pad_token_id=processor.tokenizer.eos_token_id,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Extract generated text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse predicted boxes
        print(f"output response is {output_text}")
        pred_boxes = extract_xyxy_all(output_text)
        # pred_boxes = clip_and_validate(raw_pred, W, H)
        
        log_message(f"  Found {len(pred_boxes)} valid predicted boxes", config['log_file'])
        
        # Extract attention data
        success = extract_bbox_attention_for_evaluation(
            generated_text=output_text,
            outputs=generated_ids,
            inputs=inputs,
            processor=processor,
            save_path=output_path,
            image_inputs=image_inputs,
            original_prompt=prompt,
            image_filename=image_filename,
            category_name=category_name,
            gt_boxes=gt_boxes,
            gt_boxes_xywh=xywh_list,
            base_image=base_image
        )
        
        if success:
            log_message(f"  Successfully saved attention data: {output_dir}/{output_filename}", config['log_file'])
            return True
        else:
            log_message(f"  Failed to extract attention data for {image_filename} response {response_id}", config['log_file'])
            return False
            
    except Exception as e:
        log_message(f"  Error processing {image_filename}: {str(e)}", config['log_file'])
        log_message(f"  Traceback: {traceback.format_exc()}", config['log_file'])
        return False

def main():
    """Main processing function"""
    # Setup configuration with parsed args
    config = setup_config(args)
    
    # Print configuration
    print(f"Configuration:")
    print(f"random seed is: {config['seed']}")
    print(f"  Device: {args.device}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Top-k: {config['top_k']}")
    print(f"  Top-p: {config['top_p']}")
    print(f"  Number of responses: {config['num_responses']}")
    print(f"  Output directory: {config['output_root']}")

    set_seed(config['seed'])
    # Check if required objects are available
    try:
        model
        processor  
        process_vision_info
        log_message("Model and processor are available", config['log_file'])
    except NameError as e:
        log_message(f"Required objects not found: {e}", config['log_file'])
        log_message("Please make sure model, processor, and process_vision_info are loaded", config['log_file'])
        return
    
    # Load CSV data
    log_message(f"Loading CSV data from {config['csv_path']}", config['log_file'])
    if not os.path.exists(config['csv_path']):
        log_message(f"CSV file not found: {config['csv_path']}", config['log_file'])
        return
    
    df = pd.read_csv(config['csv_path'])
    total_images = len(df)
    log_message(f"Found {total_images} images to process", config['log_file'])
    
    # Process each image
    processed = 0
    failed = 0
    skipped = 0
    filtered_out = 0  # Images with multiple GT boxes
    
    start_time = time.time()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        log_message(f"\n=== Processing {idx+1}/{total_images} ===", config['log_file'])
        
        try:
            image_filename = row['image_filename']
            category_name = row['category_name'] if 'category_name' in row.index else str(row.iloc[1])
            stem, _ = os.path.splitext(image_filename)
            safe_cat = str(category_name).replace(" ", "_").replace("/", "_")
            
            # Check if image should be filtered out (multi-GT) - do this once per image
            xywh_list = xywh_list_from_row(row)
            if not xywh_list:
                log_message(f"No GT boxes found for {image_filename}", config['log_file'])
                failed += 1
                continue
            
            # if len(xywh_list) != 1:
            #     log_message(f"Filtering out {image_filename} - has {len(xywh_list)} GT boxes (need exactly 1)", config['log_file'])
            #     filtered_out += 1
            #     continue
            
            # Process multiple responses for this image
            image_processed = 0
            image_failed = 0
            image_skipped = 0
            
            for response_id in range(1, config['num_responses'] + 1):
                # Check if this response already exists
                output_dir = os.path.join(config['output_root'], f"{stem}_{safe_cat}")
                output_filename = f"response_{response_id}.pkl"
                output_path = os.path.join(output_dir, output_filename)
                
                if os.path.exists(output_path):
                    image_skipped += 1
                    continue
                
                success = process_single_image(row, model, processor, process_vision_info, config, response_id)
                
                if success is True:
                    image_processed += 1
                elif success is False:
                    image_failed += 1
                # Note: success is None shouldn't happen here since we pre-filter
            
            # Update overall stats
            if image_processed > 0:
                processed += 1
            if image_failed > 0:
                failed += 1
            skipped += image_skipped
            
            log_message(f"Image {image_filename}: {image_processed} processed, {image_failed} failed, {image_skipped} skipped", config['log_file'])
                
        except KeyboardInterrupt:
            log_message("Processing interrupted by user", config['log_file'])
            break
        except Exception as e:
            log_message(f"Unexpected error: {e}", config['log_file'])
            failed += 1
        
        # Progress update every 10 images
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = (total_images - idx - 1) * avg_time
            log_message(f"Progress: {idx+1}/{total_images}, "
                       f"Processed: {processed}, Failed: {failed}, Skipped: {skipped}, "
                       f"Filtered out (multi-GT): {filtered_out}, "
                       f"Est. remaining: {remaining/60:.1f} min", config['log_file'])
    
    # Final summary
    elapsed = time.time() - start_time
    log_message(f"\n=== Processing Complete ===", config['log_file'])
    log_message(f"Total images in CSV: {total_images}", config['log_file'])
    log_message(f"Images with single GT: {total_images - filtered_out}", config['log_file'])
    log_message(f"Images successfully processed: {processed}", config['log_file'])
    log_message(f"Images failed: {failed}", config['log_file'])
    log_message(f"Response files skipped (already exists): {skipped}", config['log_file'])
    log_message(f"Images filtered out (multi-GT): {filtered_out}", config['log_file'])
    log_message(f"Responses per image: {config['num_responses']}", config['log_file'])
    log_message(f"Total response files generated: {processed * config['num_responses']}", config['log_file'])
    log_message(f"Success rate: {processed/(total_images - filtered_out)*100:.1f}% (of eligible images)", config['log_file'])
    log_message(f"Total time: {elapsed/60:.1f} minutes", config['log_file'])
    log_message(f"Output directory: {config['output_root']}", config['log_file'])

if __name__ == "__main__":
    main()