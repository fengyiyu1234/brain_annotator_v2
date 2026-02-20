import os
import re
import numpy as np
import tifffile
import pandas as pd
import cv2
import napari
import copy
from napari.components import ViewerModel
from napari.qt import QtViewer
from functools import partial

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSplitter, QProgressDialog, 
                             QApplication, QGroupBox, QShortcut)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from .config import PATCH_SIZE, COLOR_FACTOR, TYPE_FACTOR, CLASS_ID_MAP, NAPARI_COLOR_MAP
from .utils import normalize_percentile
from .widgets import OntologyTreeWidget

TYPE_SHORTCUTS = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U']
COLOR_SHORTCUTS = ['1', '2', '3', '4']

NAME_TO_ID = {v['name'].lower(): k for k, v in CLASS_ID_MAP.items()}
NAME_TO_ID.update({
    "red glia": 0, "green glia": 1, "yellow glia": 2,
    "red neuron": 3, "green neuron": 4, "yellow neuron": 5
})

def parse_class_string(raw_cls):
    clean_name = str(raw_cls).strip().lower()
    if clean_name in NAME_TO_ID: return NAME_TO_ID[clean_name]
    try: return int(float(raw_cls))
    except: return 0 

def natural_sort_key(s):
    """解决 10.tif 排在 2.tif 前面的问题"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# =========================================================
# Thread 1: Registration Loader
# =========================================================
class RegistrationLoaderThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict) 
    
    def __init__(self, reg_dir):
        super().__init__()
        self.reg_dir = reg_dir
        
    def run(self):
        anchor_map = {}
        if not os.path.exists(self.reg_dir):
            self.finished_signal.emit(anchor_map)
            return
            
        csv_files = []
        for root, dirs, files in os.walk(self.reg_dir):
            for f in files:
                if f.endswith('.csv'): csv_files.append(os.path.join(root, f))
                
        for i, path in enumerate(csv_files):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',', 7)
                        if len(parts) >= 7:
                            gx = float(parts[0])
                            gy = float(parts[1])
                            gz = int(float(parts[2]))
                            go = int(parts[6])
                            if go not in anchor_map: anchor_map[go] = []
                            anchor_map[go].append((gx, gy, gz))
            except Exception as e:
                pass
            self.progress_signal.emit(int((i+1)/len(csv_files)*100))
            
        self.finished_signal.emit(anchor_map)

# =========================================================
# Thread 2: Patch Loader (完全更新的切图坐标逻辑)
# =========================================================
class PatchLoaderThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(list)
    
    def __init__(self, anchors, coord_sys, root_red, root_green, det_dir):
        super().__init__()
        self.anchors = anchors 
        self.tiles = coord_sys.tiles
        self.z_max = coord_sys.z_max  # 获取全脑的齐平起跳点
        self.root_red = root_red
        self.root_green = root_green
        self.det_dir = det_dir
        
    def run(self):
        print(f"\n[CHECKPOINT] Allocating {len(self.anchors)} anchors to Tiles...")
        tasks_by_tile_z = {}
        for (gx, gy, gz) in self.anchors:
            target_tile = None
            for t in self.tiles:
                if t['abs_x'] <= gx < t['abs_x'] + t['w'] and \
                   t['abs_y'] <= gy < t['abs_y'] + t['h']:
                    target_tile = t
                    break
            
            if not target_tile: 
                print(f"[Warning] Global XY ({gx}, {gy}) out of bounds. Skipped.")
                continue
            
            # 【核心 Z 轴换算】
            gz_0based = gz - 1
            z0 = self.z_max - target_tile['abs_z']
            local_z = gz_0based + z0
            
            if local_z < 0:
                continue
            
            lx = gx - target_tile['abs_x']
            ly = gy - target_tile['abs_y']
            key = (target_tile['dir'], local_z)
            
            if key not in tasks_by_tile_z:
                tasks_by_tile_z[key] = {'tile': target_tile, 'pts': []}
            # 记录下 global_z，方便保存文件时追溯
            tasks_by_tile_z[key]['pts'].append({'lx': lx, 'ly': ly, 'gx': gx, 'gy': gy, 'gz': gz})
            
        total_tasks = len(tasks_by_tile_z)
        
        patches = []
        det_cache = {} 
        processed = 0
        
        for (t_dir, local_z), data in tasks_by_tile_z.items():
            tile = data['tile']
            t_prefix = t_dir.replace('\\', '/').split('/')[-1] 
            
            # 读取 Detection CSV 缓存
            if t_prefix not in det_cache:
                det_path = None
                for f in os.listdir(self.det_dir):
                    if t_prefix in f and f.endswith('.csv'):
                        det_path = os.path.join(self.det_dir, f)
                        if "_result" in f: break 
                
                if det_path:
                    try:
                        df = pd.read_csv(det_path, header=None,
                                         names=['fname', 'x1', 'y1', 'x2', 'y2', 'cls', 'conf', 'intensity', 'z'])
                        det_cache[t_prefix] = df
                    except: det_cache[t_prefix] = pd.DataFrame()
                else:
                    det_cache[t_prefix] = pd.DataFrame()

            df_tile = det_cache[t_prefix]
            
            # 【核心修复】原始 YOLO 结果 CSV 里的 Z 是 1-based (等于 local_z + 1)
            raw_csv_z = local_z + 1
            df_z = df_tile[df_tile['z'] == raw_csv_z].copy() if not df_tile.empty else pd.DataFrame()
            
            r_dir = os.path.join(self.root_red, t_dir)
            g_dir = os.path.join(self.root_green, t_dir)
            
            try:
                # 使用自然排序提取文件
                fr = sorted(os.listdir(r_dir), key=natural_sort_key) if os.path.exists(r_dir) else []
                fg = sorted(os.listdir(g_dir), key=natural_sort_key) if os.path.exists(g_dir) else []
                
                if local_z < 0 or local_z >= len(fr): raise IndexError
                
                img_r = tifffile.imread(os.path.join(r_dir, fr[local_z]))
                img_g = tifffile.imread(os.path.join(g_dir, fg[local_z])) if local_z < len(fg) else np.zeros_like(img_r)
                
                r8 = normalize_percentile(img_r)
                g8 = normalize_percentile(img_g)
                rgb = np.dstack((r8, g8, np.zeros_like(r8)))
            except Exception as e:
                processed += 1
                continue 

            for pt in data['pts']:
                lx, ly = pt['lx'], pt['ly']
                cx, cy = int(lx), int(ly)
                half = PATCH_SIZE // 2
                x1, y1 = cx - half, cy - half
                x2, y2 = cx + half, cy + half
                
                pad_l = max(0, -x1); pad_t = max(0, -y1)
                pad_r = max(0, x2 - rgb.shape[1]); pad_b = max(0, y2 - rgb.shape[0])
                sx1 = max(0, x1); sy1 = max(0, y1)
                sx2 = min(x2, rgb.shape[1]); sy2 = min(y2, rgb.shape[0])
                
                crop = rgb[sy1:sy2, sx1:sx2]
                if any([pad_l, pad_t, pad_r, pad_b]):
                    crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                
                boxes = []
                if not df_z.empty:
                    df_z['cx'] = (df_z['x1'] + df_z['x2']) / 2
                    df_z['cy'] = (df_z['y1'] + df_z['y2']) / 2
                    mask = (df_z['cx'] >= x1) & (df_z['cx'] <= x2) & (df_z['cy'] >= y1) & (df_z['cy'] <= y2)
                    for _, row in df_z[mask].iterrows():
                        cls_id = parse_class_string(row['cls'])
                        boxes.append({
                            'cls': cls_id,
                            'x1': row['x1'] - x1, 'y1': row['y1'] - y1,
                            'x2': row['x2'] - x1, 'y2': row['y2'] - y1
                        })
                
                # 命名带上 local_z，方便你核对和保存
                patches.append({
                    'name': f"{t_prefix}_GZ{int(pt['gz'])}_LZ{local_z}_GX{int(pt['gx'])}_GY{int(pt['gy'])}",
                    'image': crop,
                    'boxes': boxes,
                    'tile_prefix': t_prefix,
                    'z': local_z,
                    'gx': pt['gx'], 'gy': pt['gy']
                })
                
            processed += 1
            self.progress_signal.emit(int(processed/total_tasks*100))
            
        print(f"[CHECKPOINT] Patch Generation Complete. Yielded {len(patches)} image patches.")
        self.finished_signal.emit(patches)

# =========================================================
# Main GUI
# =========================================================
class SimpleAnnotator(QWidget):
    def __init__(self, coord_sys, ontology, det_dir, reg_dir, save_root, root_red, root_green):
        super().__init__()
        self.coord_sys = coord_sys
        self.ontology = ontology
        self.det_dir = det_dir
        self.reg_dir = reg_dir
        self.save_root = save_root
        self.root_red = root_red
        self.root_green = root_green
        
        self.anchor_map = {} 
        self.current_patches = []
        self.current_idx = 0
        self.undo_stack = [] 
        
        self.init_ui()
        QTimer.singleShot(100, self.start_indexing)

    def init_ui(self):
        self.setWindowTitle("Brain Annotator - Anchor Cropping Mode")
        self.resize(1600, 900)
        main_layout = QHBoxLayout(self)
        
        left_panel = QWidget()
        l_lay = QVBoxLayout(left_panel)
        self.lbl_status = QLabel("Initializing Registration Anchors...")
        self.progress = QProgressDialog("Loading...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.hide()
        
        l_lay.addWidget(self.lbl_status)
        self.tree_widget = OntologyTreeWidget(self.ontology)
        self.tree_widget.tree.itemClicked.connect(self.on_tree_click)
        l_lay.addWidget(self.tree_widget)
        
        mid_panel = QWidget()
        m_lay = QVBoxLayout(mid_panel)
        self.viewer = ViewerModel()
        self.qt_viewer = QtViewer(self.viewer)
        self.viewer.dims.ndisplay = 2
        m_lay.addWidget(self.qt_viewer)
        
        bot_bar = QHBoxLayout()
        self.btn_prev = QPushButton("<< Prev (A)"); self.btn_prev.clicked.connect(lambda: self.change_patch(-1))
        self.btn_next = QPushButton("Next (D) >>"); self.btn_next.clicked.connect(lambda: self.change_patch(1))
        self.btn_save = QPushButton("Save Patch (S)"); self.btn_save.clicked.connect(self.save_current_patch)
        self.lbl_counter = QLabel("0 / 0")
        bot_bar.addWidget(self.btn_prev); bot_bar.addWidget(self.lbl_counter)
        bot_bar.addWidget(self.btn_next); bot_bar.addWidget(self.btn_save)
        m_lay.addLayout(bot_bar)
        
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        r_lay = QVBoxLayout(right_panel)
        
        short_grp = QGroupBox("Modify Class (Shortcuts)")
        s_lay = QVBoxLayout()
        color_txt = ", ".join([f"{i+1}={v['name']}" for i, v in enumerate(COLOR_FACTOR.values())])
        type_txt = "<br>".join([f"<b>{TYPE_SHORTCUTS[i]}</b>: {k}" for i, k in enumerate(TYPE_FACTOR.keys())])
        s_text = f"<b>Color:</b> {color_txt}<hr><b>Type:</b><br>{type_txt}"
        s_lay.addWidget(QLabel(s_text))
        short_grp.setLayout(s_lay)
        r_lay.addWidget(short_grp)
        r_lay.addStretch()
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel); splitter.addWidget(mid_panel); splitter.addWidget(right_panel)
        splitter.setSizes([250, 1050, 320])
        main_layout.addWidget(splitter)
        
        self.bind_keys()
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_action)

    def bind_keys(self):
        def next_cb(v): self.change_patch(1)
        def prev_cb(v): self.change_patch(-1)
        def save_cb(v): self.save_current_patch()
        next_cb.__name__ = "next_cb"; prev_cb.__name__ = "prev_cb"; save_cb.__name__ = "save_cb"
        self.viewer.bind_key('d', next_cb)
        self.viewer.bind_key('a', prev_cb)
        self.viewer.bind_key('s', save_cb)
        
        for i, (k, v) in enumerate(TYPE_FACTOR.items()):
            if i < len(TYPE_SHORTCUTS):
                QShortcut(QKeySequence(TYPE_SHORTCUTS[i]), self).activated.connect(partial(self.modify_selection, 'type', v['base']))
                
        for i, (k, v) in enumerate(COLOR_FACTOR.items()):
            if i < len(COLOR_SHORTCUTS):
                QShortcut(QKeySequence(COLOR_SHORTCUTS[i]), self).activated.connect(partial(self.modify_selection, 'color', i))

    def start_indexing(self):
        self.progress.setLabelText("Loading Registration Anchors...")
        self.progress.show()
        self.loader = RegistrationLoaderThread(self.reg_dir)
        self.loader.progress_signal.connect(self.progress.setValue)
        self.loader.finished_signal.connect(self.indexing_done)
        self.loader.start()

    def indexing_done(self, anchor_map):
        self.progress.close()
        self.anchor_map = anchor_map
        total = sum(len(v) for v in anchor_map.values())
        self.lbl_status.setText(f"Loaded {total} Anchors in Space.")

    def on_tree_click(self, item, col):
        nid = item.data(0, Qt.UserRole)
        region_name = item.text(0)
        print(f"\n{'='*50}\n[UI ACTION] Selected Brain Region: {region_name}")
        
        desc_ids = self.ontology.get_all_descendant_ids(nid)
        target_gos = [self.ontology.id_to_go[i] for i in desc_ids if i in self.ontology.id_to_go]
        
        anchors = []
        for go in target_gos:
            if go in self.anchor_map:
                anchors.extend(self.anchor_map[go])
                
        if not anchors:
            print(f"[CHECKPOINT] No cells found in Registration data for {region_name}.")
            self.lbl_status.setText(f"No cells found for {region_name}.")
            return
            
        print(f"[CHECKPOINT] Extracted {len(anchors)} Registration Anchors (Cells) for {region_name}.")
        self.lbl_status.setText(f"Cropping {len(anchors)} patches...")
        self.progress.setLabelText("Cropping Tiles...")
        self.progress.show()
        
        # 【核心修正】把整个 coord_sys 传给线程，以便使用 z_max
        self.patch_loader = PatchLoaderThread(anchors, self.coord_sys, self.root_red, self.root_green, self.det_dir)
        self.patch_loader.progress_signal.connect(self.progress.setValue)
        self.patch_loader.finished_signal.connect(self.patches_loaded)
        self.patch_loader.start()

    def patches_loaded(self, patches):
        self.progress.close()
        self.current_patches = patches
        self.current_idx = 0
        self.undo_stack = []
        self.lbl_status.setText(f"Loaded {len(patches)} Patches.")
        self.show_patch()

    def show_patch(self):
        if not self.current_patches: 
            self.viewer.layers.clear(); return
        data = self.current_patches[self.current_idx]
        self.viewer.layers.clear()
        self.viewer.add_image(data['image'], name='image')
        
        shapes, classes, edges, labels = [], [], [], []
        for b in data['boxes']:
            cid = b['cls']
            cfg = CLASS_ID_MAP.get(cid, {'name': f'UNK {cid}', 'color': 'white'})
            shapes.append([[b['y1'], b['x1']], [b['y2'], b['x2']]])
            classes.append(cid)
            edges.append(NAPARI_COLOR_MAP.get(cid, 'white'))
            labels.append(cfg['name'])
            
        if shapes:
            self.shapes_layer = self.viewer.add_shapes(
                shapes, shape_type='rectangle', edge_width=2, edge_color=edges,
                face_color='transparent', name='boxes',
                text={'string': '{label}', 'color': 'white', 'anchor': 'upper_left', 'size': 8},
                properties={'label': labels, 'class_id': classes}
            )
            self.shapes_layer.events.data.connect(self.push_undo)
            
        self.lbl_counter.setText(f"{self.current_idx + 1} / {len(self.current_patches)}")
        self.viewer.text_overlay.text = data['name']

    def modify_selection(self, factor, value):
        if not hasattr(self, 'shapes_layer'): return
        sel_idx = list(self.shapes_layer.selected_data)
        if not sel_idx: return
        self.push_undo(None)
        
        props = self.shapes_layer.properties
        edge_c = self.shapes_layer.edge_color
        
        for idx in sel_idx:
            old_id = int(props['class_id'][idx])
            c_base = (old_id // 3) * 3
            c_off = old_id % 3
            
            new_id = (c_base + value) if factor == 'color' else (value + c_off)
            cfg = CLASS_ID_MAP.get(new_id)
            if not cfg: continue
            
            props['class_id'][idx] = new_id
            props['label'][idx] = cfg['name']
            
            from napari.utils.colormaps.standardize_color import transform_color
            edge_c[idx] = transform_color(cfg['color'])[0]
            
            self.current_patches[self.current_idx]['boxes'][idx]['cls'] = new_id
            
        self.shapes_layer.properties = props
        self.shapes_layer.edge_color = edge_c
        self.shapes_layer.refresh()
        self.shapes_layer.refresh_text()

    def push_undo(self, event):
        if not self.current_patches: return
        data = self.current_patches[self.current_idx]
        self.undo_stack.append(copy.deepcopy(data['boxes']))
        if len(self.undo_stack) > 10: self.undo_stack.pop(0)

    def undo_action(self):
        if not self.undo_stack: return
        self.current_patches[self.current_idx]['boxes'] = self.undo_stack.pop()
        self.show_patch()

    def change_patch(self, delta):
        if not self.current_patches: return
        self.current_idx = (self.current_idx + delta) % len(self.current_patches)
        self.undo_stack = []
        self.show_patch()

    def save_current_patch(self):
        if not self.current_patches: return
        data = self.current_patches[self.current_idx]
        base = data['name']
        
        img_dir = os.path.join(self.save_root, 'images')
        lbl_dir = os.path.join(self.save_root, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(img_dir, base+'.jpg'), cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
        
        lines = []
        if hasattr(self, 'shapes_layer'):
            for i, box in enumerate(self.shapes_layer.data):
                ys, xs = box[:, 0], box[:, 1]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                cx = (x1 + x2) / 2 / PATCH_SIZE
                cy = (y1 + y2) / 2 / PATCH_SIZE
                w = (x2 - x1) / PATCH_SIZE
                h = (y2 - y1) / PATCH_SIZE
                
                cid = self.shapes_layer.properties['class_id'][i]
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
        with open(os.path.join(lbl_dir, base+'.txt'), 'w') as f:
            f.write("\n".join(lines))
            
        self.viewer.text_overlay.text = f"Saved {base}!"