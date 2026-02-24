import re
import os
import cv2
from .config import CLASS_ID_MAP

NAME_TO_ID = {v['name'].lower(): k for k, v in CLASS_ID_MAP.items()}
NAME_TO_ID.update({
    "red glia": 0, "green glia": 1, "yellow glia": 2,
    "red neuron": 3, "green neuron": 4, "yellow neuron": 5
})

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def parse_class_string(raw_cls):
    clean_name = str(raw_cls).strip().lower()
    if clean_name in NAME_TO_ID: return NAME_TO_ID[clean_name]
    try: return int(float(raw_cls))
    except: return 0 

def export_yolo_patch(image_stack, shapes_layer_data, properties, patch_size, save_root, base_name):
    """将 Napari 里的框转换为 YOLO txt，并保存中心层图像"""
    img_dir = os.path.join(save_root, 'images')
    lbl_dir = os.path.join(save_root, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    # 提取中间目标层 (Z=1)
    target_img = image_stack[1]
    cv2.imwrite(os.path.join(img_dir, base_name + '.jpg'), cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR))
    
    lines = []
    if shapes_layer_data is not None:
        for i, box in enumerate(shapes_layer_data):
            z_coord = int(box[0][0])
            if z_coord != 1: continue # 仅保存中间层
            
            ys, xs = box[:, 1], box[:, 2] 
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            
            cx = (x1 + x2) / 2 / patch_size
            cy = (y1 + y2) / 2 / patch_size
            w = (x2 - x1) / patch_size
            h = (y2 - y1) / patch_size
            
            cid = properties['class_id'][i]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
    with open(os.path.join(lbl_dir, base_name + '.txt'), 'w') as f:
        f.write("\n".join(lines))
    return True