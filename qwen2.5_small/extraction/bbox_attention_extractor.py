import torch
import numpy as np
import re
import json
import math
import pickle
import os

def _to_int(x):
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return int(round(val))
    except Exception:
        return None

def extract_xyxy_all(text):
    """Extract all [x1, y1, x2, y2] patterns from text - 保持原有功能"""
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

def _compact_ws(s: str) -> str:
    """Remove whitespace for comparison"""
    return re.sub(r"\s+", "", s)

def find_candidate_json_spans(text: str):
    """Find potential JSON array spans in text - 保持原有功能但不再使用"""
    spans = []
    stack = []
    for i, ch in enumerate(text):
        if ch == '[':
            stack.append(i)
        elif ch == ']':
            if stack:
                st = stack.pop()
                if i - st >= 10:
                    spans.append((st, i+1))
    return spans

def extract_xyxy_matches(text_sub: str):
    """Extract bbox patterns with position info - 保持原有功能但不再使用"""
    num = r"([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)"
    pat = re.compile(r"\[\s*"+num+r"\s*,\s*"+num+r"\s*,\s*"+num+r"\s*,\s*"+num+r"\s*\]?")
    out = []
    for m in pat.finditer(text_sub):
        groups = []
        for gi in range(1,5):
            gs, ge = m.span(gi)
            groups.append((gs, ge, m.group(gi)))
        out.append({
            "span": m.span(0),
            "groups": groups,
            "values": [m.group(i) for i in range(1,5)],
            "raw_text": m.group(0)
        })
    return out

def map_char_span_to_token_span(offsets, char_start: int, char_end: int):
    """Map character span to token span"""
    covered = []
    for i, (s, e) in enumerate(offsets):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        covered.append(i)
    if not covered:
        return None
    return covered[0], covered[-1], covered

def calculate_single_token_entropy(step_logits):
    """
    计算单个token位置的entropy
    Args:
        step_logits: 该步骤的logits [batch_size, vocab_size] 或 [vocab_size]
    Returns:
        float: entropy值
    """
    print(f"step logits shape is {step_logits.shape}")
    # 处理batch维度
    if step_logits.dim() == 2:
        step_logits = step_logits[0]  # 取第一个batch
    
    # 计算概率分布
    step_probs = torch.softmax(step_logits, dim=-1)
    
    # 计算entropy: -sum(p * log(p))
    entropy = -(step_probs * torch.log(step_probs + 1e-10)).sum()
    
    return entropy.item()

def locate_bbox_number_tokens(generated_text: str, tokenizer, input_length: int = None):
    """
    改进版：直接匹配bbox数字，不依赖JSON结构完整性
    修复了原版本只能找到部分bbox的问题
    """
    print(f"Analyzing generated text length: {len(generated_text)}")
    
    # 直接使用正则表达式找到所有的bbox模式
    bbox_matches = []
    
    # 定义数字的正则表达式  
    num = r"([-+]?\d+(?:\.\d+)?)"
    
    # 多种bbox模式，使用前瞻断言确保数字后面有合适的分隔符
    patterns = [
        # bbox_2d键值对格式（优先级最高）
        (re.compile(r'"bbox_2d"\s*:\s*\[\s*' + num + r'\s*,\s*' + num + r'\s*,\s*' + num + r'\s*,\s*' + num + r'(?=\s*[,\]\}\s]|$)'), "bbox_2d"),
        # 纯数组格式
        (re.compile(r'\[\s*' + num + r'\s*,\s*' + num + r'\s*,\s*' + num + r'\s*,\s*' + num + r'(?=\s*[,\]\}\s]|$)'), "array"),
        # 其他bbox键值对格式
        (re.compile(r'"bbox"\s*:\s*\[\s*' + num + r'\s*,\s*' + num + r'\s*,\s*' + num + r'\s*,\s*' + num + r'(?=\s*[,\]\}\s]|$)'), "bbox"),
    ]
    
    for pattern, pattern_name in patterns:
        for match in pattern.finditer(generated_text):
            try:
                # 提取四个数字及其在原文中的精确位置
                groups = []
                values = []
                for i in range(1, 5):
                    value_str = match.group(i)
                    values.append(float(value_str))
                    # 获取数字在原文中的字符位置
                    char_start, char_end = match.span(i)
                    groups.append((char_start, char_end, value_str))
                
                match_info = {
                    "values": values,
                    "groups": groups,
                    "span": match.span(0),
                    "raw_text": match.group(0),
                    "pattern_type": pattern_name
                }
                bbox_matches.append(match_info)
                print(f"Found {pattern_name}: {values} at position {match.span(0)}")
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing match: {e}")
                continue
    
    if not bbox_matches:
        print("Warning: No bbox patterns found")
        return None
    
    # 按位置排序
    bbox_matches.sort(key=lambda x: x["span"][0])
    
    # 去重：移除数值相同且位置重叠的匹配
    unique_matches = []
    for match in bbox_matches:
        is_duplicate = False
        match_values = tuple(match["values"])
        match_start, match_end = match["span"]
        
        for existing in unique_matches:
            existing_values = tuple(existing["values"])
            existing_start, existing_end = existing["span"]
            
            if (match_values == existing_values and 
                not (match_end <= existing_start or match_start >= existing_end)):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_matches.append(match)
        else:
            print(f"Removed duplicate: {match_values}")
    
    print(f"Found {len(unique_matches)} unique bbox patterns")

    # Encode text to tokens
    try:
        enc = tokenizer(
            generated_text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]
        input_ids = enc["input_ids"]
                # 🔎 print all token list
        print("\n=== Token check ===")
        for idx, tok_id in enumerate(input_ids):
            tok_text = tokenizer.decode([tok_id],
                                        skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
            print(f"{idx:>4}: {tok_id:>6} | {tok_text!r} | span={offsets[idx]}")
        print("=====================\n")
    except Exception as e:
        print(f"Error encoding text: {e}")
        return None

    bbox_token_info = []
    
    for bi, match in enumerate(unique_matches, start=1):
        coords_info = []
        verification_passed = True
        
        print(f"Processing bbox {bi}: {match['values']}")
        
        for name, (char_start, char_end, txt) in zip(["x1","y1","x2","y2"], match["groups"]):
            mapped = map_char_span_to_token_span(offsets, char_start, char_end)
            
            if mapped is None:
                print(f"  Warning: Failed to map {name}={txt} to tokens")
                verification_passed = False
                coord_info = {
                    "name": name, "text": txt,
                    "char_span": (char_start, char_end),
                    "first_token_step": None,
                    "all_token_steps": [],  # 新增
                    "verification_passed": False
                }
            else:
                tok_s, tok_e, covered = mapped
                first_token_step = covered[0]
                
                # Verify token decoding
                try:
                    first_token_decoded = tokenizer.decode([input_ids[covered[0]]], 
                                                        skip_special_tokens=True, 
                                                        clean_up_tokenization_spaces=False)
                    all_tokens_decoded = tokenizer.decode(input_ids[covered[0]:covered[-1]+1], 
                                                        skip_special_tokens=True, 
                                                        clean_up_tokenization_spaces=False)
                    
                    txt_clean = _compact_ws(txt)
                    decoded_clean = _compact_ws(all_tokens_decoded)
                    tokens_match = txt_clean == decoded_clean
                    
                    if not tokens_match:
                        print(f"  Warning: Token mismatch for {name}: '{txt_clean}' vs '{decoded_clean}'")
                        # 注意：这里不设置verification_passed = False，因为轻微的token差异是正常的
                
                except Exception as e:
                    print(f"  Error verifying tokens for {name}: {e}")
                    tokens_match = False

                coord_info = {
                    "name": name, "text": txt,
                    "char_span": (char_start, char_end),
                    "first_token_step": first_token_step,
                    "all_token_steps": covered,  
                    "first_token_decoded": first_token_decoded if 'first_token_decoded' in locals() else None,
                    "all_tokens_decoded": all_tokens_decoded if 'all_tokens_decoded' in locals() else None,
                    "verification_passed": tokens_match
                }
            
            print(f"  {name}: first_token_step = {first_token_step}, all_token_steps = {covered if mapped else []}")
            coords_info.append(coord_info)
        
        bbox_info = {
            "bbox_id": bi,
            "coordinates": coords_info,
            "bbox_values": [float(match["values"][i]) for i in range(4)],
            "verification_passed": verification_passed
        }
        bbox_token_info.append(bbox_info)
    
    return bbox_token_info

def identify_vision_tokens(input_ids, tokenizer):
    """Identify positions of vision tokens in the input sequence"""
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    input_ids_flat = input_ids.flatten()
    
    vision_positions = {
        'start': torch.where(input_ids_flat == vision_start_id)[0].tolist(),
        'end': torch.where(input_ids_flat == vision_end_id)[0].tolist(),
        'image_pads': torch.where(input_ids_flat == image_pad_id)[0].tolist()
    }
    
    return vision_positions

def get_token_grid_info(image_inputs):
    """
    根据处理后的图像尺寸计算正确的token网格
    Args:
        image_inputs: PIL图像列表，例如 [<PIL.Image.Image image mode=RGB size=644x420>]
    Returns:
        tuple: (token_height, token_width, processed_image_size)
    """
    if not image_inputs or len(image_inputs) == 0:
        return 0, 0, (0, 0)
    
    # 获取第一张图像的尺寸
    image = image_inputs[0]  # PIL Image
    width, height = image.size  # PIL的size是 (width, height)
    
    # 根据qwen_vl_utils.py，IMAGE_FACTOR = 28
    IMAGE_FACTOR = 28
    token_width = width // IMAGE_FACTOR   # 注意：PIL的width对应token_width
    token_height = height // IMAGE_FACTOR  # PIL的height对应token_height
    
    print(f"PIL Image size: {width} x {height}")
    print(f"Calculated token grid: {token_height} x {token_width} = {token_height * token_width}")
    
    return token_height, token_width, (width, height)

def extract_bbox_attention_for_evaluation(generated_text, outputs, inputs, processor, save_path, image_inputs=None, original_prompt=None, image_filename=None, category_name=None, gt_boxes=None, gt_boxes_xywh=None, base_image=None):
    """
    Extract bbox attention data for evaluation purposes
    
    Args:
        generated_text: Generated text containing bboxes
        outputs: Model outputs containing .attentions and .scores
        inputs: Model inputs for vision token positions
        processor: Model processor
        save_path: Path to save the pkl file
        image_inputs: PIL图像列表，用于计算正确的token网格 (新增)
        original_prompt: The prompt used for generation (optional)
        image_filename: Name of the image file (optional)
        category_name: Target category name (optional)
        gt_boxes: Ground truth boxes in xyxy format (optional)
        gt_boxes_xywh: Ground truth boxes in xywh format (optional)
    
    Returns:
        bool: Success status (True if pkl saved, False if error occurred)
    """
    
    print(f"Starting bbox attention extraction for {image_filename or 'unknown image'}...")
    
    # Initialize result structure - always save this regardless of success/failure
    result_data = {
        'bboxes': {},
        'extraction_success': False,  # 标识提取是否成功
        'failure_reason': None,       # 失败原因
        'raw_generated_text': generated_text,  # 始终保存原始生成文本
        'token_grid': {},
        'metadata': {
            'image_filename': image_filename,
            'category_name': category_name,
            "gt_boxes": gt_boxes,
            "gt_boxes_xywh": gt_boxes_xywh,
            'extracted_boxes': [],  # 先初始化为空
            'num_layers': 0,
            'num_bboxes': 0,
            'generation_steps': {},
            'generated_text': generated_text,  # 保持原有字段兼容性
            'verification_summary': {
                'total_bboxes': 0,
                'failed_bboxes': 0,
                'success_rate': 0.0
            }
        }
    }
    
    try:
        # Extract bounding boxes using the original function
        extracted_boxes = extract_xyxy_all(generated_text)
        result_data['metadata']['extracted_boxes'] = extracted_boxes
        
        if not extracted_boxes:
            result_data['failure_reason'] = "No bounding boxes found in generated text"
            print("No bounding boxes found in generated text")
        else:
            print(f"extract_xyxy_all found {len(extracted_boxes)} boxes: {extracted_boxes}")
            
            # Locate bbox token positions using the improved function
            bbox_token_info = locate_bbox_number_tokens(
                generated_text=generated_text,
                tokenizer=processor.tokenizer,
                input_length=inputs.input_ids.shape[1]
            )
            
            if not bbox_token_info:
                result_data['failure_reason'] = "Failed to locate bbox tokens"
                print("Failed to locate bbox tokens")
            else:
                print(f"locate_bbox_number_tokens found {len(bbox_token_info)} bboxes")
                
                # Check verification
                failed_bboxes = [bbox for bbox in bbox_token_info if not bbox["verification_passed"]]
                if failed_bboxes:
                    print(f"Token verification failed for {len(failed_bboxes)} bbox(es)")
                
                # Identify vision tokens
                vision_positions = identify_vision_tokens(inputs['input_ids'], processor.tokenizer)
                image_pad_positions = vision_positions['image_pads']
                
                if not image_pad_positions:
                    result_data['failure_reason'] = "No vision tokens found"
                    print("No vision tokens found")
                else:
                    # 使用新的token网格计算方法
                    if image_inputs is not None:
                        token_height, token_width, processed_image_size = get_token_grid_info(image_inputs)
                        
                        # 验证token数量是否匹配
                        expected_tokens = token_height * token_width
                        actual_tokens = len(image_pad_positions)
                        print(f"Token grid validation:")
                        print(f"  Expected tokens (from image size): {expected_tokens}")
                        print(f"  Actual vision tokens: {actual_tokens}")
                        
                        if expected_tokens != actual_tokens:
                            print(f"  Warning: Token count mismatch!")
                            # 仍然继续处理，但记录警告
                        
                        result_data['token_grid'] = {
                            'height': token_height,
                            'width': token_width,
                            'total_tokens': len(image_pad_positions),
                            "original_image_size": base_image.size if base_image is not None else (0, 0),
                            'processed_image_size': processed_image_size,
                            'expected_tokens': expected_tokens,
                            'token_count_match': expected_tokens == actual_tokens
                        }
                    else:
                        # 如果没有image_inputs，使用旧方法作为fallback
                        print("Warning: image_inputs not provided, using fallback method")
                        total_vision_tokens = len(image_pad_positions)
                        sqrt_tokens = int(np.sqrt(total_vision_tokens))
                        
                        for height in range(sqrt_tokens, 0, -1):
                            if total_vision_tokens % height == 0:
                                width = total_vision_tokens // height
                                token_height, token_width = height, width
                                break
                        else:
                            token_height = token_width = sqrt_tokens
                        
                        result_data['token_grid'] = {
                            'height': token_height,
                            'width': token_width,
                            'total_tokens': len(image_pad_positions),
                            'processed_image_size': (0, 0),
                            'expected_tokens': token_height * token_width,
                            'token_count_match': True  # fallback方法总是匹配
                        }
                    
                    # Get number of layers and validate data availability
                    if not outputs.attentions:
                        result_data['failure_reason'] = "No attention data found in outputs"
                        print("No attention data found in outputs")
                    elif not outputs.scores:
                        result_data['failure_reason'] = "No scores data found in outputs"
                        print("No scores data found in outputs")
                    else:
                        # 添加验证：检查attentions和scores长度是否匹配
                        print(f"Attention steps: {len(outputs.attentions)}")
                        print(f"Score steps: {len(outputs.scores)}")
                        
                        if len(outputs.attentions) != len(outputs.scores):
                            print("Warning: Attention and scores lengths don't match!")
                            print(f"This may cause indexing issues in entropy calculation")
                        
                        # Check if we have enough attention steps
                        valid_coords = [
                            coord for bbox in bbox_token_info 
                            for coord in bbox["coordinates"] 
                            if coord["first_token_step"] is not None
                        ]
                        
                        if not valid_coords:
                            result_data['failure_reason'] = "No valid coordinate tokens found"
                            print("No valid coordinate tokens found")
                        else:
                            max_step = max(coord["first_token_step"] for coord in valid_coords)
                            
                            # 检查所有token步骤是否在有效范围内
                            all_token_steps = [
                                step for bbox in bbox_token_info 
                                for coord in bbox["coordinates"] 
                                for step in coord.get("all_token_steps", [])
                            ]
                            max_entropy_step = max(all_token_steps) if all_token_steps else 0
                            
                            if max_step >= len(outputs.attentions):
                                result_data['failure_reason'] = f"Required attention step {max_step} exceeds available steps {len(outputs.attentions)}"
                                print(f"Required attention step {max_step} exceeds available steps {len(outputs.attentions)}")
                            elif max_entropy_step >= len(outputs.scores):
                                result_data['failure_reason'] = f"Required entropy step {max_entropy_step} exceeds available scores {len(outputs.scores)}"
                                print(f"Required entropy step {max_entropy_step} exceeds available scores {len(outputs.scores)}")
                            else:
                                num_layers = len(outputs.attentions[0])  # Number of layers in first step
                                result_data['metadata']['num_layers'] = num_layers
                                result_data['metadata']['num_bboxes'] = len(bbox_token_info)
                                
                                print(f"Processing {len(bbox_token_info)} bboxes with {num_layers} layers")
                                print(f"Token grid: {token_height} x {token_width} = {token_height * token_width} tokens")
                                
                                # Process each bbox
                                processed_bboxes = 0
                                
                                for bbox_info in bbox_token_info:
                                    bbox_id = bbox_info["bbox_id"]
                                    bbox_key = f'bbox_{bbox_id}'
                                    
                                    bbox_data = {
                                        'values': bbox_info["bbox_values"],
                                        'verification_passed': bbox_info["verification_passed"]
                                    }
                                    
                                    generation_steps = {}
                                    
                                    # Process each coordinate
                                    for coord_info in bbox_info["coordinates"]:
                                        coord_name = coord_info["name"]
                                        first_token_step = coord_info["first_token_step"]
                                        all_token_steps = coord_info.get("all_token_steps", [])
                                        print(f"\n=== Processing {bbox_key} {coord_name} ===")
                                        print(f"  Text: '{coord_info['text']}'")
                                        print(f"  First token step: {first_token_step}")
                                        print(f"  All token steps: {all_token_steps}")
                                        if first_token_step is None:
                                            print(f"Warning: No valid token step for {bbox_key} {coord_name}")
                                            continue
                                        
                                        generation_steps[coord_name] = first_token_step
                                        
                                        # === ATTENTION EXTRACTION (修改：保存所有head) ===
                                        step_attentions = outputs.attentions[first_token_step]
                                        coord_attentions = []
                                        
                                        for layer_idx in range(num_layers):
                                            layer_attn = step_attentions[layer_idx]  # [batch, heads, seq, seq]
                                            generated_token_attn = layer_attn[0, :, -1, :]  # [heads, seq]
                                            # 修改：不再对head取平均，保持[heads, vision_tokens]结构
                                            vision_only = generated_token_attn[:, image_pad_positions]     # [heads, vision_tokens]
                                            coord_attentions.append(vision_only.float().cpu().numpy())
                                        
                                        # Stack to [num_layers, heads, vision_tokens]
                                        bbox_data[f'{coord_name}_attention'] = np.stack(coord_attentions)
                                        
                                        print(f"  Extracted {coord_name} attention: shape {bbox_data[f'{coord_name}_attention'].shape}")
                                        
                                        # === ENTROPY EXTRACTION (新增) ===
                                        if all_token_steps:
                                            coord_entropies = []
                                            for token_step in all_token_steps:
                                                if token_step < len(outputs.scores):
                                                    step_logits = outputs.scores[token_step]
                                                    entropy = calculate_single_token_entropy(step_logits)
                                                    coord_entropies.append(entropy)
                                                else:
                                                    print(f"  Warning: Token step {token_step} exceeds scores length for {coord_name}")
                                            
                                            bbox_data[f'{coord_name}_entropy'] = np.array(coord_entropies)
                                            print(f"  Extracted {coord_name} entropy: {len(coord_entropies)} values {coord_entropies}")
                                        else:
                                            bbox_data[f'{coord_name}_entropy'] = np.array([])
                                            print(f"  No token steps found for {coord_name} entropy")
                                    
                                    if generation_steps:  # Only add if we have at least some valid coordinates
                                        result_data['bboxes'][bbox_key] = bbox_data
                                        result_data['metadata']['generation_steps'][bbox_key] = generation_steps
                                        processed_bboxes += 1
                                
                                # Update verification summary
                                result_data['metadata']['verification_summary'] = {
                                    'total_bboxes': len(bbox_token_info),
                                    'failed_bboxes': len(failed_bboxes),
                                    'success_rate': (len(bbox_token_info) - len(failed_bboxes)) / len(bbox_token_info) if bbox_token_info else 0
                                }
                                
                                if processed_bboxes > 0:
                                    result_data['extraction_success'] = True
                                    print(f"Successfully processed {processed_bboxes}/{len(bbox_token_info)} bboxes")
                                else:
                                    result_data['failure_reason'] = "No bboxes could be processed successfully"
                                    print("No bboxes could be processed successfully")

    except Exception as e:
        result_data['failure_reason'] = f"Exception during processing: {str(e)}"
        print(f"Exception during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # try:
    #     os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(result_data, f)
        
    #     if result_data['extraction_success']:
    #         print(f"Successfully saved attention data to: {save_path}")
    #         print(f"Data structure: {len(result_data['bboxes'])} bboxes processed, {result_data['metadata']['num_layers']} layers, {result_data['token_grid'].get('total_tokens', 0)} vision tokens")
    #         if 'processed_image_size' in result_data['token_grid']:
    #             print(f"Processed image size: {result_data['token_grid']['processed_image_size']}")
    #     else:
    #         print(f"Saved failed extraction data to: {save_path}")
    #         print(f"Failure reason: {result_data['failure_reason']}")
    #         print(f"Generated text length: {len(generated_text)} chars")
        
    #     return True  # 文件保存成功就返回True
        
    # except Exception as e:
    #     print(f"Failed to save data to {save_path}: {str(e)}")
    #     return False  # 只有文件保存失败才返回False
    import gzip

    try:
        # 自动创建输出目录
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # 统一使用 .pkl.gz 结尾，防止误判为未压缩文件
        compressed_path = save_path if save_path.endswith(".gz") else save_path + ".gz"

        # ✅ 使用 gzip 压缩保存
        with gzip.open(compressed_path, 'wb') as f:
            pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if result_data['extraction_success']:
            print(f"✅ Successfully saved compressed attention data to: {compressed_path}")
            print(f"Data structure: {len(result_data['bboxes'])} bboxes processed, "
                  f"{result_data['metadata']['num_layers']} layers, "
                  f"{result_data['token_grid'].get('total_tokens', 0)} vision tokens")
            if 'original_image_size' in result_data['token_grid']:
                print(f"Original image size: {result_data['token_grid']['original_image_size']}")
        else:
            print(f"⚠️ Saved failed extraction data to: {compressed_path}")
            print(f"Failure reason: {result_data['failure_reason']}")
            print(f"Generated text length: {len(generated_text)} chars")

        return True  # 文件保存成功就返回True

    except Exception as e:
        print(f"❌ Failed to save compressed data to {save_path}: {str(e)}")
        return False  # 只有文件保存失败才返回False