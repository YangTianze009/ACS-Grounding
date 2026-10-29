import os
import pickle
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
import argparse
import gzip 

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

def setup_logging(output_dir: str):
    """
    Setup logging with both console and file output
    
    Args:
        output_dir: Output directory for log file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging configuration
    log_file = os.path.join(output_dir, 'data_preparation.log')
    
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

def map_bbox_to_original_size(pred_bbox: List[float], processed_size: Tuple[int, int], 
                             original_size: Tuple[int, int]) -> List[float]:
    """
    Map bbox from processed image size back to original image size
    
    Args:
        pred_bbox: Predicted bbox [x1, y1, x2, y2] in processed image coordinates
        processed_size: (width, height) of processed image
        original_size: (width, height) of original image
    
    Returns:
        Mapped bbox [x1, y1, x2, y2] in original image coordinates
    """
    processed_w, processed_h = processed_size
    original_w, original_h = original_size
    
    if processed_w == 0 or processed_h == 0:
        return pred_bbox
    
    # Convert to numpy array for easier manipulation
    bbox_array = np.array(pred_bbox)
    
    # Scale factors
    scale_x = original_w / processed_w
    scale_y = original_h / processed_h
    
    # Apply inverse transformation: bbox / processed_size * original_size
    mapped_bbox = bbox_array * np.array([scale_x, scale_y, scale_x, scale_y])
    
    return mapped_bbox.tolist()

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
    # print(f"gt boxes are {len(gt_boxes)}")
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

def save_attention_maps(attention_map: np.ndarray, output_path: str, precision: str = 'fp32'):
    """
    Save attention map with specified precision
    
    Args:
        attention_map: The attention map to save
        output_path: Output file path
        precision: 'fp16' or 'fp32'
    """
    if precision == 'fp16':
        # Convert to fp16 to save storage
        attention_map = attention_map.astype(np.float16)
    # For fp32, keep the original dtype
    
    np.save(output_path, attention_map)

def save_single_sample(sample: Dict, split_dir: str, logger, precision: str = 'fp32') -> Optional[Dict]:
    """
    Save a single sample as pkl file and return CSV record
    
    Args:
        sample: Sample data
        split_dir: Split directory path
        logger: Logger instance
        precision: Precision for saving attention maps ('fp16' or 'fp32')
    
    Returns:
        CSV record dict or None if filtered out
    """
    bbox_info = sample['bbox_info']
    bbox_values = bbox_info['pred_bbox']
    
    # Check bbox coordinate order
    if not is_valid_bbox(bbox_values):
        logger.warning(f"Filtered invalid bbox coordinates: {bbox_info['image_folder']}_response_{bbox_info['response_id']}_bbox_{bbox_info['bbox_id']}")
        return None
    
    # Generate filename
    filename = (f"{bbox_info['image_folder']}_response_{bbox_info['response_id']}_bbox_{bbox_info['bbox_id']}.pkl")
    
    # Convert attention maps to specified precision before saving
    if precision == 'fp16':
        for key in ['x1_attention_map', 'y1_attention_map', 'x2_attention_map', 'y2_attention_map']:
            if key in sample['attention_data']:
                sample['attention_data'][key] = sample['attention_data'][key].astype(np.float16)
    
    # Save pkl file
    file_path = os.path.join(split_dir, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(sample, f)
    
    # Calculate bbox area
    x1, y1, x2, y2 = bbox_values
    bbox_area = (x2 - x1) * (y2 - y1)
    
    # Generate CSV record
    csv_record = {
        'pkl_file': filename,  # Saved pkl filename
        'image_folder': bbox_info['image_folder'],
        'response_id': bbox_info['response_id'],
        'bbox_id': bbox_info['bbox_id'],
        'pred_bbox': str(bbox_info['pred_bbox']),
        'mapped_pred_bbox': str(bbox_info['mapped_pred_bbox']),
        'gt_box': str(bbox_info['gt_bbox']),
        'bbox_area': round(bbox_area, 2),
        'iou': round(bbox_info['iou'], 4),
        'label': bbox_info['label'],
        'verification_passed': sample['metadata']['verification_passed'],
        'attention_precision': precision  # Add precision info to CSV
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
                                        output_dir: str, val_image_folders: set, is_test: bool, 
                                        logger, precision: str = 'fp32') -> List[Dict]:
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
        precision: Precision for saving attention maps
    
    Returns:
        CSV record list
    """
    try:
        data = load_pickle_file(pkl_path)  # 直接调用新函数
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
    
    # Get image size information for bbox mapping
    original_image_size = token_grid.get('original_image_size', (0, 0))  # (width, height)
    processed_image_size = token_grid.get('processed_image_size', (0, 0))  # (width, height)
    
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
            
            # Pre-check bbox coordinate order
            if not is_valid_bbox(pred_box):
                logger.warning(f"Skipped invalid bbox coordinates: {image_folder}_response_{response_id}_{bbox_key}")
                continue
            
            # Map prediction box from processed image size to original image size
            mapped_pred_box = pred_box  # Default to original pred_box
            if (original_image_size != (0, 0) and processed_image_size != (0, 0) and 
                original_image_size != processed_image_size):
                
                mapped_pred_box = map_bbox_to_original_size(
                    pred_box, processed_image_size, original_image_size
                )
                logger.debug(f"Mapped bbox from {pred_box} to {mapped_pred_box} "
                           f"(processed: {processed_image_size} -> original: {original_image_size})")
            else:
                if original_image_size == (0, 0) or processed_image_size == (0, 0):
                    logger.warning(f"Missing image size info in {pkl_path}: "
                                 f"original={original_image_size}, processed={processed_image_size}")
                else:
                    logger.debug(f"No bbox mapping needed (same size): {original_image_size}")
            
            # Calculate IoU using mapped bbox
            max_iou = calculate_max_iou(mapped_pred_box, gt_boxes)
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
                        coord_attention_maps[coord] = attention_map
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
                        'pred_bbox': pred_box,  # Keep original pred_box in processed coordinates
                        'mapped_pred_bbox': mapped_pred_box,  # Add mapped bbox for reference
                        'gt_bbox': gt_boxes,
                        'iou': max_iou,  # IoU calculated using mapped bbox
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
                        'original_image_size': original_image_size,  # Add size info to metadata
                        'processed_image_size': processed_image_size
                    }
                }
                
                # Save sample immediately and get CSV record with precision
                csv_record = save_single_sample(sample, split_dir, logger, precision)
                if csv_record:
                    csv_records.append(csv_record)
                
            else:
                logger.warning(f"Skipped {bbox_key} in {pkl_path}: only {valid_coords}/4 valid coordinates")
                
        except Exception as e:
            logger.error(f"Error processing {bbox_key} in {pkl_path}: {e}")
            continue
    
    return csv_records

def process_image_folder_streaming(image_folder_path: str, output_dir: str, 
                                 val_image_folders: set, is_test: bool, logger, 
                                 precision: str = 'fp32') -> List[Dict]:
    """
    Process all response pkl files in a single image folder with streaming save
    
    Args:
        image_folder_path: Image folder path
        output_dir: Output directory
        val_image_folders: Set of validation image folders
        is_test: Whether this is test data
        logger: Logger instance
        precision: Precision for saving attention maps
    
    Returns:
        CSV record list for all responses of this image
    """
    image_folder_name = os.path.basename(image_folder_path)
    all_csv_records = []
    
    # Find all response pkl files
    response_files = glob.glob(os.path.join(image_folder_path, "response_*.pkl"))
    response_files += glob.glob(os.path.join(image_folder_path, "response_*.pkl.gz"))
    
    if not response_files:
        logger.warning(f"No response pkl files found in {image_folder_path}")
        return []
    
    for response_file in sorted(response_files):
        # Extract response number
        filename = os.path.basename(response_file)
        response_match = re.search(r'response_(\d+)\.pkl(?:\.gz)?', filename)
        if response_match:
            response_id = int(response_match.group(1))
        else:
            logger.error(f"Cannot parse response ID from {filename}")
            continue
        
        # Process this response file with streaming save
        csv_records = process_single_response_pkl_streaming(
            response_file, image_folder_name, response_id, output_dir, 
            val_image_folders, is_test, logger, precision
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
                              val_image_folders: set, logger, precision: str = 'fp32') -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Process all image folders in directory with streaming save
    
    Args:
        directory_path: Directory path
        output_dir: Output directory
        is_test: Whether this is test data
        val_image_folders: Set of validation image folders (only used for train/val data)
        logger: Logger instance
        precision: Precision for saving attention maps
    
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
            image_folder_path, output_dir, val_image_folders or set(), 
            is_test, logger, precision
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

def main_train_only(train_val_dir: str, output_dir: str, data_type: str, precision: str = 'fp32'):
    """
    Process only train/val data
    
    Args:
        train_val_dir: Directory containing train/val data
        output_dir: Output directory for processed data
        data_type: Type of data being processed ('training')
        precision: Precision for saving attention maps
    """
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("Processing ONLY train/val data...")
    logger.info(f"Input directory: {train_val_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Attention map precision: {precision}")
    logger.info("Processing structure: image_folder/response_X.pkl")
    logger.info("Attention maps will be kept in original format [num_layers, vision_tokens]")
    logger.info("Filtering bbox with invalid coordinates (x2 <= x1 or y2 <= y1)")
    logger.info("NEW: Mapping predicted bboxes from processed image size to original image size for IoU calculation")
    
    if precision == 'fp16':
        logger.info("Using FP16 precision for attention maps (50% storage reduction)")
    else:
        logger.info("Using FP32 precision for attention maps (full precision)")
    
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
    train_csv, val_csv, _ = process_directory_streaming(
        train_val_dir, output_dir, is_test=False, 
        val_image_folders=val_image_folders, logger=logger, precision=precision
    )
    
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

def main_test_only(test_dir: str, output_dir: str, data_type: str, precision: str = 'fp32'):
    """
    Process only test data
    
    Args:
        test_dir: Directory containing test data
        output_dir: Output directory for processed data
        data_type: Type of data being processed ('testing')
        precision: Precision for saving attention maps
    """
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("Processing ONLY test data...")
    logger.info(f"Input directory: {test_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Attention map precision: {precision}")
    logger.info("Processing structure: image_folder/response_X.pkl")
    logger.info("Attention maps will be kept in original format [num_layers, vision_tokens]")
    logger.info("Filtering bbox with invalid coordinates (x2 <= x1 or y2 <= y1)")
    logger.info("NEW: Mapping predicted bboxes from processed image size to original image size for IoU calculation")
    
    if precision == 'fp16':
        logger.info("Using FP16 precision for attention maps (50% storage reduction)")
    else:
        logger.info("Using FP32 precision for attention maps (full precision)")
    
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
    _, _, test_csv = process_directory_streaming(
        test_dir, output_dir, is_test=True, 
        val_image_folders=set(), logger=logger, precision=precision
    )
    
    if not test_csv:
        logger.error("No valid samples found!")
        return
    
    # Save CSV list
    logger.info("Saving CSV list...")
    save_csv_list(test_csv, os.path.join(output_dir, "test_list.csv"), logger)
    
    # Print statistics
    logger.info("Test data preparation completed!")
    print_csv_statistics(test_csv, "Test", logger)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process attention data for object detection')
    
    # Data configuration
    parser.add_argument(
        '--data_type',
        type=str,
        default='testing',
        choices=['training', 'testing'],
        help='Type of data to process (default: testing)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='Objects365',
        choices=['Objects365', 'COCO', 'RefCOCO'],  # 你可以根据需要添加更多数据集
        help='Dataset name (default: Objects365)'
    )
    
    parser.add_argument(
        '--object_num',
        type=str,
        default='single',
        choices=['single', 'multiple'],
        help='Number of objects (default: single)'
    )
    
    # Model response configuration
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='Temperature for model response (default: 0.5)'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p for model response (default: 0.9)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Top-k for model response (optional)'
    )
    
    parser.add_argument(
        '--response_num',
        type=int,
        default=10,
        help='Number of responses (default: 10)'
    )
    
    # Processing configuration
    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp16', 'fp32'],
        help='Precision for saving attention maps (fp16 saves ~50%% storage, default: fp16)'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Validation split ratio for train/val data (default: 0.2)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for train/val split (default: 42)'
    )
    
    # Path configuration (optional overrides)
    parser.add_argument(
        '--base_dir',
        type=str,
        default=os.path.join(os.environ.get('DATA_ROOT', '../../data'), 'object_detection_Qwen', 'model_response'),
        help='Base directory for model responses'
    )
    
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='./processed_data',
        help='Base directory for output (default: ./processed_data)'
    )
    
    return parser.parse_args()

def main():
    """Main function with path configuration from command line"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Construct response directory name based on parameters
    response_dir_parts = [
        "bbox_attention",
        f"temp_{args.temperature}",
        f"top_p_{args.top_p}"
    ]
    
    # Add optional parameters

    response_dir_parts.append(f"response_{args.response_num}")
    
    
    response_dir = "_".join(response_dir_parts)
    
    # Construct output directory
    output_dir_parts = [
        args.output_base_dir,
        args.dataset,
        f"{args.object_num}_object",
        f"temp_{args.temperature}_top_p_{args.top_p}"
    ]
    
    if args.top_k is not None:
        output_dir_parts[-1] += f"_top_k_{args.top_k}"
   
    output_dir_parts[-1] += f"_response_{args.response_num}"
    
    output_dir = os.path.join(*output_dir_parts)
    
    # Construct base directories
    base_training_dir = os.path.join(
        args.base_dir,
        args.dataset,
        "training",
        f"{args.object_num}_object"
    )
    
    base_testing_dir = os.path.join(
        args.base_dir,
        args.dataset,
        "testing",
        f"{args.object_num}_object"
    )
    
    # Print configuration
    print("=" * 80)
    print("Starting data preparation with following configuration:")
    print("=" * 80)
    print(f"  Dataset: {args.dataset}")
    print(f"  Object number: {args.object_num}")
    print(f"  Data type: {args.data_type}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    if args.top_k:
        print(f"  Top-k: {args.top_k}")
    print(f"  Response number: {args.response_num}")
    print(f"  Response directory: {response_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Attention precision: {args.precision}")
    print("=" * 80)
    
    if args.data_type == "training":
        train_val_dir = os.path.join(base_training_dir, response_dir)
        print(f"  Input directory: {train_val_dir}")
        print(f"  Validation ratio: {args.val_ratio}")
        print(f"  Random seed: {args.random_seed}")
        
        if not os.path.exists(train_val_dir):
            print(f"ERROR: Input directory does not exist: {train_val_dir}")
            return
            
        main_train_only(train_val_dir, output_dir, args.data_type, args.precision)
    
    elif args.data_type == "testing":
        test_dir = os.path.join(base_testing_dir, response_dir)
        print(f"  Input directory: {test_dir}")
        
        if not os.path.exists(test_dir):
            print(f"ERROR: Input directory does not exist: {test_dir}")
            return
            
        main_test_only(test_dir, output_dir, args.data_type, args.precision)

if __name__ == "__main__":
    main()