# config.py
import numpy as np

# --- Parameters ---
PATCH_SIZE = 256
TILE_SHAPE_PX = (2048, 2048) 

TYPE_SHORTCUTS = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U']
COLOR_SHORTCUTS = ['1', '2', '3', '4']

# --- Classes Configuration ---
# 因子 1: 颜色 (0-2)
COLOR_FACTOR = {
    0: {'name': 'Red',    'color': 'red'},
    1: {'name': 'Green',  'color': 'green'},
    2: {'name': 'Yellow', 'color': 'yellow'}
}

# 因子 2: 细胞类型 (基数)
# ID = Type_Base + Color_Index
TYPE_FACTOR = {
    'Glia':   {'base': 0,  'name': 'Glia'},
    'Neuron': {'base': 3,  'name': 'Neuron'},
    'TypeA':  {'base': 6,  'name': 'Type A'},
    'TypeB':  {'base': 9,  'name': 'Type B'},
    'TypeC':  {'base': 12, 'name': 'Type C'}
}

# 自动生成所有 YOLO ID 的映射 (用于显示)
CLASS_ID_MAP = {}
for t_key, t_val in TYPE_FACTOR.items():
    for c_key, c_val in COLOR_FACTOR.items():
        cid = t_val['base'] + c_key
        name = f"{c_val['name']} {t_val['name']}"
        CLASS_ID_MAP[cid] = {'name': name, 'color': c_val['color'], 'type': t_key, 'color_idx': c_key}

# 颜色映射 (用于 Napari)
NAPARI_COLOR_MAP = {cid: info['color'] for cid, info in CLASS_ID_MAP.items()}