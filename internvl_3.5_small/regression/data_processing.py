import os
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import glob
from collections import defaultdict
import random
import re
import logging
from tqdm import tqdm

def setup_logging(output_dir: str, data_type: str):
    """
    Setup logging with both console and file output
    
    Args:
        output_dir: Output directory for log file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging configuration
    log_file = os.path.join(output_dir, f'data_preparation_{data_type}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger

def load_pickle_file(file_path: str):
    """
    Load pickle file, supporting both .pkl and .pkl.gz formats
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded data
    """
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def compute_iou_xyxy(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes in xyxy format
    
    Args:
        box1, box2: [x1, y1, x2, y2] format bboxes
    
    Returns:
        IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

def process_bbox_refcoco_style(pred_bbox: List[float], original_size: Tuple[int, int]) -> List[float]:
    """
    Process bbox using RefCOCO-style normalization and scaling
    
    Args:
        pred_bbox: Predicted bbox [x1, y1, x2, y2] (may be in [0, 1000] or [0, 1] range)
        original_size: (width, height) of original image
    
    Returns:
        Processed bbox [x1, y1, x2, y2] in absolute coordinates (pixels)
    """
    original_w, original_h = original_size
    
    # Step 1: Normalize from [0, 1000] to [0, 1] if needed
    if sum(pred_bbox) >= 4:
        normalized_box = [x / 1000.0 for x in pred_bbox]
    else:
        normalized_box = pred_bbox  # Already in [0, 1] range
    
    # Step 2: Convert to absolute coordinates (pixels)
    processed_bbox = [
        normalized_box[0] * original_w,  # x1
        normalized_box[1] * original_h,  # y1
        normalized_box[2] * original_w,  # x2
        normalized_box[3] * original_h   # y2
    ]
    
    return processed_bbox

def calculate_max_iou(pred_box: List[float], gt_boxes: List[List[float]]) -> float:
    """
    Calculate maximum IoU between prediction box and all GT boxes
    
    Args:
        pred_box: Prediction box [x1, y1, x2, y2]
        gt_boxes: GT boxes list [[x1, y1, x2, y2], ...]
    
    Returns:
        Maximum IoU value
    """
    if not gt_boxes:
        return 0.0
    
    max_iou = 0.0
    for gt_box in gt_boxes:
        iou = compute_iou_xyxy(pred_box, gt_box)
        max_iou = max(max_iou, iou)
    
    return max_iou

def is_valid_bbox(bbox_values: List[float]) -> bool:
    """
    Check if bbox coordinates order is valid
    
    Args:
        bbox_values: [x1, y1, x2, y2] format bbox
    
    Returns:
        bool: Whether it's a valid bbox (only check coordinate order)
    """
    if len(bbox_values) != 4:
        return False
    
    x1, y1, x2, y2 = bbox_values
    
    # Only check coordinate order
    if x2 <= x1 or y2 <= y1:
        return False
    
    return True

def parse_image_folder_name(folder_name: str) -> Tuple[str, str]:
    """
    Parse image folder name to extract image ID and category
    
    Args:
        folder_name: Folder name like "000000000064_stop_sign"
    
    Returns:
        (image_id, category)
    """
    parts = folder_name.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return folder_name, "unknown"

def save_single_sample(sample: Dict, split_dir: str, logger) -> Optional[Dict]:
    """
    Save a single sample as pkl.gz file and return CSV record
    
    Args:
        sample: Sample data
        split_dir: Split directory path
        logger: Logger instance
    
    Returns:
        CSV record dict or None if filtered out
    """
    bbox_info = sample['bbox_info']
    processed_bbox = bbox_info['processed_pred_bbox']
    
    # Check bbox coordinate order
    if not is_valid_bbox(processed_bbox):
        logger.warning(f"Filtered invalid bbox coordinates: {bbox_info['image_folder']}_response_{bbox_info['response_id']}_bbox_{bbox_info['bbox_id']}")
        return None
    
    # Generate filename with .pkl.gz extension
    filename = (f"{bbox_info['image_folder']}_response_{bbox_info['response_id']}_bbox_{bbox_info['bbox_id']}.pkl")
    
    # Save pkl.gz file
    # file_path = os.path.join(split_dir, filename)
    # with gzip.open(file_path, 'wb') as f:
    #     pickle.dump(sample, f)
    file_path = os.path.join(split_dir, filename)
    with open(file_path, 'wb') as f:  # 去掉 gzip.
        pickle.dump(sample, f)
    
    # Calculate bbox area
    x1, y1, x2, y2 = processed_bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    
    # Generate CSV record
    csv_record = {
        'pkl_file': filename,  # Saved pkl.gz filename
        'image_folder': bbox_info['image_folder'],
        'response_id': bbox_info['response_id'],
        'bbox_id': bbox_info['bbox_id'],
        'pred_bbox': str(bbox_info['pred_bbox']),
        'processed_pred_bbox': str(bbox_info['processed_pred_bbox']),
        'gt_box': str(bbox_info['gt_bbox']),
        'bbox_area': round(bbox_area, 2),
        'iou': round(bbox_info['iou'], 4),
        'label': bbox_info['label'],
        'verification_passed': sample['metadata']['verification_passed']
    }
    
    return csv_record

def determine_split(image_folder: str, val_image_folders: set) -> str:
    """
    Determine which split an image folder belongs to
    
    Args:
        image_folder: Image folder name
        val_image_folders: Set of validation image folders
    
    Returns:
        Split name ('train' or 'val')
    """
    if image_folder in val_image_folders:
        return 'val'
    else:
        return 'train'

def process_single_response_pkl_streaming(pkl_path: str, image_folder: str, response_id: int,
                                         output_dir: str, val_image_folders: set, is_test: bool, logger) -> List[Dict]:
    """
    Process single response pkl file and save samples immediately, return CSV records
    
    Args:
        pkl_path: Response pkl file path
        image_folder: Image folder name (like "000000000064_stop_sign")
        response_id: Response number
        output_dir: Output directory
        val_image_folders: Set of validation image folders
        is_test: Whether this is test data
        logger: Logger instance
    
    Returns:
        CSV record list
    """
    try:
        data = load_pickle_file(pkl_path)
    except Exception as e:
        logger.error(f"Error loading {pkl_path}: {e}")
        return []
    
    csv_records = []
    
    # Check data integrity
    if not data.get('extraction_success', False):
        logger.warning(f"Extraction failed for {pkl_path}, reason: {data.get('failure_reason', 'Unknown')}")
        return []
    
    # Get GT boxes
    gt_boxes = data['metadata'].get('gt_boxes', [])
    if not gt_boxes:
        logger.warning(f"No GT boxes found in {pkl_path}")
        return []
    
    # Get token grid information
    token_grid = data.get('token_grid', {})
    original_height = token_grid.get('height', 0)
    original_width = token_grid.get('width', 0)
    
    if original_height == 0 or original_width == 0:
        logger.error(f"Invalid token grid in {pkl_path}")
        return []
    
    # Get image size information for bbox processing
    original_image_size = token_grid.get('original_image_size', (0, 0))  # (width, height)
    
    if original_image_size == (0, 0):
        logger.error(f"Missing original_image_size in {pkl_path}")
        return []
    
    # Parse image information
    image_id, category = parse_image_folder_name(image_folder)
    
    # Determine split and create directory
    if is_test:
        split_name = 'test'
    else:
        split_name = determine_split(image_folder, val_image_folders)
    
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Process each bbox
    for bbox_key, bbox_data in data.get('bboxes', {}).items():
        try:
            # Get prediction box and verification status
            pred_box = bbox_data.get('values', [])
            verification_passed = bbox_data.get('verification_passed', False)
            
            if len(pred_box) != 4:
                logger.warning(f"Invalid pred_box in {pkl_path} {bbox_key}")
                continue
            
            # Process bbox using RefCOCO-style normalization
            processed_pred_box = process_bbox_refcoco_style(pred_box, original_image_size)
            
            logger.debug(f"Processed bbox from {pred_box} to {processed_pred_box} "
                        f"(original size: {original_image_size})")
            
            # Pre-check bbox coordinate order
            if not is_valid_bbox(processed_pred_box):
                logger.warning(f"Skipped invalid bbox coordinates: {image_folder}_response_{response_id}_{bbox_key}")
                continue
            
            # Calculate IoU using processed bbox
            max_iou = calculate_max_iou(processed_pred_box, gt_boxes)
            label = 1 if max_iou >= 0.5 else 0
            
            # Get attention data for four coordinates (keep original format)
            coord_attention_maps = {}
            valid_coords = 0
            
            for coord in ['x1', 'y1', 'x2', 'y2']:
                attention_key = f'{coord}_attention'
                if attention_key in bbox_data:
                    attention_map = bbox_data[attention_key]  # [num_layers, vision_tokens]
                    
                    # Verify attention data integrity
                    expected_tokens = original_height * original_width
                    if attention_map.shape[2] == expected_tokens:
                        # coord_attention_maps[coord] = attention_map
                        coord_attention_maps[coord] = attention_map.astype(np.float16)
                        valid_coords += 1
                    else:
                        logger.error(f"Attention shape mismatch in {pkl_path} {bbox_key} {coord}: "
                                   f"expected {expected_tokens}, got {attention_map.shape[1]}")
                        break
            
            # Only save sample when all 4 coordinates' attention are successfully processed
            if valid_coords == 4:
                # Extract bbox number
                bbox_match = re.search(r'bbox_(\d+)', bbox_key)
                bbox_id = int(bbox_match.group(1)) if bbox_match else 1
                
                sample = {
                    'bbox_info': {
                        'image_folder': image_folder,
                        'image_id': image_id,
                        'category': category,
                        'response_id': response_id,
                        'bbox_id': bbox_id,
                        'pred_bbox': pred_box,  # Original predicted bbox
                        'processed_pred_bbox': processed_pred_box,  # Processed bbox in absolute coordinates
                        'gt_bbox': gt_boxes,
                        'iou': max_iou,  # IoU calculated using processed bbox
                        'label': label
                    },
                    'attention_data': {
                        'x1_attention_map': coord_attention_maps['x1'],  # [num_layers, vision_tokens]
                        'y1_attention_map': coord_attention_maps['y1'],
                        'x2_attention_map': coord_attention_maps['x2'],
                        'y2_attention_map': coord_attention_maps['y2'],
                    },
                    'metadata': {
                        'original_grid_shape': (original_height, original_width),
                        'verification_passed': verification_passed,
                        'source_pkl': pkl_path,
                        'original_image_size': original_image_size
                    }
                }
                
                # Save sample immediately and get CSV record
                csv_record = save_single_sample(sample, split_dir, logger)
                if csv_record:
                    csv_records.append(csv_record)
            
            else:
                logger.warning(f"Skipped {bbox_key} in {pkl_path}: only {valid_coords}/4 valid coordinates")
        
        except Exception as e:
            logger.error(f"Error processing {bbox_key} in {pkl_path}: {e}")
            continue
    
    return csv_records

def process_image_folder_streaming(image_folder_path: str, output_dir: str,
                                  val_image_folders: set, is_test: bool, logger) -> List[Dict]:
    """
    Process all response pkl files in a single image folder with streaming save
    
    Args:
        image_folder_path: Image folder path
        output_dir: Output directory
        val_image_folders: Set of validation image folders
        is_test: Whether this is test data
        logger: Logger instance
    
    Returns:
        CSV record list for all responses of this image
    """
    image_folder_name = os.path.basename(image_folder_path)
    all_csv_records = []
    
    # Find all response pkl files (support both .pkl and .pkl.gz)
    response_files = glob.glob(os.path.join(image_folder_path, "response_*.pkl"))
    response_files += glob.glob(os.path.join(image_folder_path, "response_*.pkl.gz"))
    
    if not response_files:
        logger.warning(f"No response pkl files found in {image_folder_path}")
        return []
    
    for response_file in sorted(response_files):
        # Extract response number
        filename = os.path.basename(response_file)
        response_match = re.search(r'response_(\d+)\.pkl', filename)
        if response_match:
            response_id = int(response_match.group(1))
        else:
            logger.error(f"Cannot parse response ID from {filename}")
            continue
        
        # Process this response file with streaming save
        csv_records = process_single_response_pkl_streaming(
            response_file, image_folder_name, response_id, output_dir, val_image_folders, is_test, logger
        )
        all_csv_records.extend(csv_records)
    
    return all_csv_records

def get_all_image_folders(directory_path: str) -> List[str]:
    """
    Get all image folder names from directory
    
    Args:
        directory_path: Directory path
    
    Returns:
        List of image folder names
    """
    return [d for d in os.listdir(directory_path) 
            if os.path.isdir(os.path.join(directory_path, d))]

def split_image_folders(image_folders: List[str], val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[set, set]:
    """
    Split image folders into train and validation sets
    
    Args:
        image_folders: List of image folder names
        val_ratio: Validation set ratio
        random_seed: Random seed
    
    Returns:
        (train_image_folders, val_image_folders) as sets
    """
    random.seed(random_seed)
    shuffled_folders = image_folders.copy()
    random.shuffle(shuffled_folders)
    
    val_count = int(len(shuffled_folders) * val_ratio)
    val_image_folders = set(shuffled_folders[:val_count])
    train_image_folders = set(shuffled_folders[val_count:])
    
    return train_image_folders, val_image_folders

def process_directory_streaming(directory_path: str, output_dir: str, is_test: bool,
                               val_image_folders: set, logger) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Process all image folders in directory with streaming save
    
    Args:
        directory_path: Directory path
        output_dir: Output directory
        is_test: Whether this is test data
        val_image_folders: Set of validation image folders (only used for train/val data)
        logger: Logger instance
    
    Returns:
        (train_csv_records, val_csv_records, test_csv_records)
    """
    train_csv_records = []
    val_csv_records = []
    test_csv_records = []
    
    # Find all image folders
    image_folders = get_all_image_folders(directory_path)
    
    logger.info(f"Found {len(image_folders)} image folders in {directory_path}")
    
    # Process with progress bar
    for image_folder in tqdm(sorted(image_folders), desc="Processing image folders"):
        image_folder_path = os.path.join(directory_path, image_folder)
        csv_records = process_image_folder_streaming(
            image_folder_path, output_dir, val_image_folders or set(), is_test, logger
        )
        
        if is_test:
            test_csv_records.extend(csv_records)
        else:
            # Determine split for this image folder
            if image_folder in val_image_folders:
                val_csv_records.extend(csv_records)
            else:
                train_csv_records.extend(csv_records)
    
    return train_csv_records, val_csv_records, test_csv_records

def save_csv_list(csv_records: List[Dict], csv_path: str, logger):
    """Save CSV list"""
    df = pd.DataFrame(csv_records)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV list to {csv_path}")

def print_csv_statistics(csv_records: List[Dict], split_name: str, logger):
    """Print dataset statistics from CSV records"""
    if not csv_records:
        logger.info(f"{split_name}: No samples")
        return
    
    # Basic statistics
    total_count = len(csv_records)
    pos_count = sum(1 for r in csv_records if r['label'] == 1)
    neg_count = total_count - pos_count
    
    # Image and category statistics
    image_folders = set(r['image_folder'] for r in csv_records)
    responses = set(r['response_id'] for r in csv_records)
    
    # IoU distribution
    ious = [r['iou'] for r in csv_records]
    avg_iou = np.mean(ious)
    
    # Categories
    categories = set()
    for r in csv_records:
        image_folder = r['image_folder']
        _, category = parse_image_folder_name(image_folder)
        categories.add(category)
    
    logger.info(f"\n{split_name} Statistics:")
    logger.info(f"  Total samples: {total_count}")
    logger.info(f"  Positive samples: {pos_count} ({pos_count/total_count:.3f})")
    logger.info(f"  Negative samples: {neg_count} ({neg_count/total_count:.3f})")
    logger.info(f"  Unique image folders: {len(image_folders)}")
    logger.info(f"  Unique categories: {len(categories)}")
    logger.info(f"  Response IDs: {sorted(responses)}")
    logger.info(f"  Average IoU: {avg_iou:.4f}")

def main_train_only(train_val_dir: str, output_dir: str, data_type: str):
    """
    Process only train/val data
    
    Args:
        train_val_dir: Directory containing train/val data
        output_dir: Output directory for processed data
    """
    # Setup logging
    logger = setup_logging(output_dir, data_type)
    
    logger.info("Processing ONLY train/val data...")
    logger.info("Processing structure: image_folder/response_X.pkl or response_X.pkl.gz")
    logger.info("Attention maps will be kept in original format [num_layers, vision_tokens]")
    logger.info("Filtering bbox with invalid coordinates (x2 <= x1 or y2 <= y1)")
    logger.info("Using RefCOCO-style bbox processing for IoU calculation")
    logger.info(f"Output files will be saved as .pkl.gz format")
    
    # Check if train_val_dir exists
    if not os.path.exists(train_val_dir):
        logger.error(f"Train/val directory does not exist: {train_val_dir}")
        return
    
    # Get all image folders and split them
    logger.info("Preparing train/val split...")
    train_val_image_folders = get_all_image_folders(train_val_dir)
    
    if not train_val_image_folders:
        logger.error(f"No image folders found in: {train_val_dir}")
        return
    
    train_image_folders, val_image_folders = split_image_folders(train_val_image_folders, val_ratio=0.2)
    logger.info(f"Train image folders: {len(train_image_folders)}")
    logger.info(f"Val image folders: {len(val_image_folders)}")
    
    # Process train/val data
    logger.info("Processing train/val data with streaming...")
    train_csv, val_csv, _ = process_directory_streaming(train_val_dir, output_dir, is_test=False,
                                                        val_image_folders=val_image_folders, logger=logger)
    
    if not train_csv and not val_csv:
        logger.error("No valid samples found!")
        return
    
    # Save CSV lists
    logger.info("Saving CSV lists...")
    save_csv_list(train_csv, os.path.join(output_dir, "train_list.csv"), logger)
    save_csv_list(val_csv, os.path.join(output_dir, "val_list.csv"), logger)
    
    # Print statistics
    logger.info("Train/Val data preparation completed!")
    print_csv_statistics(train_csv, "Train", logger)
    print_csv_statistics(val_csv, "Val", logger)

def main_test_only(test_dir: str, output_dir: str, data_type: str):
    """
    Process only test data
    
    Args:
        test_dir: Directory containing test data
        output_dir: Output directory for processed data
    """
    # Setup logging
    logger = setup_logging(output_dir, data_type)
    
    logger.info("Processing ONLY test data...")
    logger.info("Processing structure: image_folder/response_X.pkl or response_X.pkl.gz")
    logger.info("Attention maps will be kept in original format [num_layers, vision_tokens]")
    logger.info("Filtering bbox with invalid coordinates (x2 <= x1 or y2 <= y1)")
    logger.info("Using RefCOCO-style bbox processing for IoU calculation")
    logger.info(f"Output files will be saved as .pkl.gz format")
    
    # Check if test_dir exists
    if not os.path.exists(test_dir):
        logger.error(f"Test directory does not exist: {test_dir}")
        return
    
    # Get test image folders
    test_image_folders = get_all_image_folders(test_dir)
    if not test_image_folders:
        logger.error(f"No image folders found in: {test_dir}")
        return
    
    logger.info(f"Test image folders: {len(test_image_folders)}")
    
    # Process test data
    logger.info("Processing test data with streaming...")
    _, _, test_csv = process_directory_streaming(test_dir, output_dir, is_test=True,
                                                val_image_folders=set(), logger=logger)
    
    if not test_csv:
        logger.error("No valid samples found!")
        return
    
    # Save CSV list
    logger.info("Saving CSV list...")
    save_csv_list(test_csv, os.path.join(output_dir, "test_list.csv"), logger)
    
    # Print statistics
    logger.info("Test data preparation completed!")
    print_csv_statistics(test_csv, "Test", logger)

if __name__ == "__main__":
    # Configure paths here
    data_type = "testing"
    dataset_name = "Objects365"
    temperature = 0.5

    response_dir = f"bbox_attention_temp_{temperature}_top_p_0.9_response_10"
    output_dir = f"./processed_data/{dataset_name}/temp_{temperature}_top_p_0.9_response_10"

    # Example 1: Process train/val data
    if data_type == "training":
        train_val_dir = f"/scratch/ty45972/MLLM_object_detection/object_detection_InternVL/model_response/{dataset_name}/training/{response_dir}"
        main_train_only(train_val_dir, output_dir, data_type)  # Process train/val data
    # Example 2: Process test data
    if data_type == "testing":
        test_dir = f"/scratch/ty45972/MLLM_object_detection/object_detection_InternVL/model_response/{dataset_name}/testing/{response_dir}"
        main_test_only(test_dir, output_dir, data_type)  # Process test data
        