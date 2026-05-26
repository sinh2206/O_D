import torch
import json
from pathlib import Path
from tqdm import tqdm
from .process import imread_unicode, letterbox_preprocess, enhance_low_light_bgr
from .nms import postprocess_batch
from .config import IMG_SIZE, CLASS_NAMES

@torch.no_grad()
def run_inference(model, image_paths, device, batch_size=8, conf_thresh=0.3):
    model.eval()
    all_predictions = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inference"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        batch_metas = []
        valid_indices = []
        
        for idx, path in enumerate(batch_paths):
            image = imread_unicode(str(path))
            if image is None:
                continue
            
            image = enhance_low_light_bgr(image)
            tensor, meta = letterbox_preprocess(image, IMG_SIZE)
            batch_tensors.append(tensor)
            batch_metas.append(meta)
            valid_indices.append(idx)
            
        if not batch_tensors:
            continue
            
        input_tensor = torch.stack(batch_tensors).to(device)
        outputs = model(input_tensor)
        
        results = postprocess_batch(outputs, batch_metas, conf_thresh=conf_thresh)
        
        for idx, res in enumerate(results):
            original_idx = valid_indices[idx]
            image_id = Path(batch_paths[original_idx]).stem
            all_predictions.append({
                "image_id": image_id,
                "predictions": res
            })
            
    return all_predictions

def save_predictions_json(predictions, output_path):
    # Standard format for typical challenges: list of dicts with image_id and boxes
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

def apply_class_thresholds(predictions, class_conf_thresh):
    """
    class_conf_thresh: dict {class_name: threshold} or list of thresholds in CLASS_NAMES order
    """
    if isinstance(class_conf_thresh, list):
        thresh_dict = {name: t for name, t in zip(CLASS_NAMES, class_conf_thresh)}
    else:
        thresh_dict = class_conf_thresh
        
    filtered = []
    for item in predictions:
        new_preds = []
        for p in item["predictions"]:
            if p["score"] >= thresh_dict.get(p["class"], 0.0):
                new_preds.append(p)
        filtered.append({
            "image_id": item["image_id"],
            "predictions": new_preds
        })
    return filtered
