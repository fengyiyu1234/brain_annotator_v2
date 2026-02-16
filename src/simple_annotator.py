import os
import numpy as np
import tifffile
import pandas as pd
import cv2
import napari
import copy
from napari.components import ViewerModel
from napari.qt import QtViewer
from functools import partial
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import SimpleITK as sitk
import traceback
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSplitter, QMessageBox, QProgressDialog, 
                             QApplication, QTableWidget, QTableWidgetItem, QComboBox, QGroupBox, QShortcut)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from .config import (PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE, TILE_SHAPE_PX)
from .utils import normalize_percentile
from .widgets import OntologyTreeWidget

# --- 核心配置 ---
PATCH_SIZE = 256 

# --- 1. 双因子类别配置 (Two-Factor Configuration) ---

# 颜色因子 (偏移量)
COLOR_FACTORS = {
    0: {'name': 'Red',    'color': 'red',    'key': '1'},
    1: {'name': 'Green',  'color': 'green',  'key': '2'},
    2: {'name': 'Yellow', 'color': 'yellow', 'key': '3'}
}

# 细胞类型因子 (基准索引)
# 按照你的需求: q=Neuron, w=Glia, e=TypeA...
# 注意: Glia是0-2, 所以Base=0; Neuron是3-5, 所以Base=3
TYPE_FACTORS = {
    'Glia':   {'base': 0,  'key': 'w', 'code': 'Glia'},
    'Neuron': {'base': 3,  'key': 'q', 'code': 'Neuron'},
    'TypeA':  {'base': 6,  'key': 'e', 'code': 'TypeA'},
    'TypeB':  {'base': 9,  'key': 'r', 'code': 'TypeB'},
    'TypeC':  {'base': 12, 'key': 't', 'code': 'TypeC'},
}

NAME_TO_ID = {
    "red glia": 0,
    "green glia": 1,
    "yellow glia": 2,
    "red neuron": 3,
    "green neuron": 4,
    "yellow neuron": 5
}
# 动态生成最终的 CLASS_CONFIG 字典
# 结构: ID -> {'name': 'Red Glia', 'color': 'red', 'type': 'Glia'}
CLASS_CONFIG = {}
for t_name, t_info in TYPE_FACTORS.items():
    for c_offset, c_info in COLOR_FACTORS.items():
        final_id = t_info['base'] + c_offset
        final_name = f"{c_info['name']} {t_name}"
        CLASS_CONFIG[final_id] = {
            'name': final_name,
            'color': c_info['color'], # Visual color
            'type': t_info['code']    # For filtering
        }

# Napari 颜色映射
COLOR_DICT = {
    'red':    [1, 0, 0],
    'green':  [0, 1, 0],
    'yellow': [1, 1, 0],
    'white':  [1, 1, 1] # Fallback
}

# --- 全局变量 (用于多进程共享 Atlas) ---
global_atlas = None

def init_worker(atlas_array):
    global global_atlas
    global_atlas = atlas_array

# --- Worker 1: 索引 CSV ---

def process_single_csv(task_data):
    csv_path, t_idx, origin, res_raw, res_nav = task_data
    results = {} 
    
    if global_atlas is None: return {}
    if not os.path.exists(csv_path): return {}

    try:
        # 读取 CSV
        df = pd.read_csv(csv_path, header=None, usecols=[1,2,3,4,5,8],
                         names=['x1', 'y1', 'x2', 'y2', 'cls', 'z'])
        df = df.dropna(subset=['x1', 'y1', 'z'])
    except Exception:
        return {}

    # --- 坐标转换 ---
    z_local = df['z'].values - 1
    cx_local = (df['x1'].values + df['x2'].values) / 2
    cy_local = (df['y1'].values + df['y2'].values) / 2
    
    gz_um = origin[0] + z_local * res_raw[0]
    gy_um = origin[1] + cy_local * res_raw[1]
    gx_um = origin[2] + cx_local * res_raw[2]
    
    az = (gz_um / res_nav[0]).astype(int)
    ay = (gy_um / res_nav[1]).astype(int)
    ax = (gx_um / res_nav[2]).astype(int)
    
    max_z, max_y, max_x = global_atlas.shape
    np.clip(az, 0, max_z-1, out=az)
    np.clip(ay, 0, max_y-1, out=ay)
    np.clip(ax, 0, max_x-1, out=ax)
    
    atlas_ids = global_atlas[az, ay, ax]
    valid_mask = atlas_ids > 0
    
    if not np.any(valid_mask): return {}
        
    valid_df = df[valid_mask]
    valid_ids = atlas_ids[valid_mask]
    valid_z = z_local[valid_mask]
    
    # --- 结果收集 ---
    for i, (idx, row) in enumerate(valid_df.iterrows()):
        aid = int(valid_ids[i])
        
        # [核心修正] 智能解析 Class
        raw_cls = row['cls']
        cls_val = 0 # 默认值
        
        # 1. 先尝试查字典 (处理 "yellow neuron")
        # 去掉可能的空格并转小写，增加匹配成功率
        clean_name = str(raw_cls).strip() 
        
        if clean_name in NAME_TO_ID:
            cls_val = NAME_TO_ID[clean_name]
        else:
            # 2. 如果字典里没有，尝试转数字 (处理 "5.0" 或 5)
            try:
                cls_val = int(float(raw_cls))
            except (ValueError, TypeError):
                # 如果既不是已知名字，也不是数字，跳过
                print(f"Unknown class: {raw_cls}")
                continue 

        if aid not in results: results[aid] = []
        
        results[aid].append({
            'tile_idx': t_idx,
            'z': int(valid_z[i]),
            'x1': row['x1'], 'y1': row['y1'],
            'x2': row['x2'], 'y2': row['y2'],
            'cls': cls_val
        })
            
    return results

# --- Worker 2: 加载并切片 ---
def load_patches_worker(task_data):
    r_path, g_path, tile_idx, z, cell_list = task_data
    patches = []
    
    try:
        img_r = tifffile.imread(r_path)
        img_g = tifffile.imread(g_path)
        
        r8 = normalize_percentile(img_r)
        g8 = normalize_percentile(img_g)
        
        # Red Channel, Green Channel, Blue=0
        rgb = np.zeros((*r8.shape, 3), dtype=np.uint8)
        rgb[..., 0] = r8 
        rgb[..., 1] = g8 
        
        for c in cell_list:
            cx, cy = (c['x1']+c['x2'])//2, (c['y1']+c['y2'])//2
            half = PATCH_SIZE // 2
            y1, y2 = int(cy - half), int(cy + half)
            x1, x2 = int(cx - half), int(cx + half)
            
            pad_t, pad_l = max(0, -y1), max(0, -x1)
            pad_b, pad_r = max(0, y2 - rgb.shape[0]), max(0, x2 - rgb.shape[1])
            
            crop = rgb[max(0,y1):min(y2, rgb.shape[0]), max(0,x1):min(x2, rgb.shape[1])]
            
            if any([pad_t, pad_l, pad_b, pad_r]):
                crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            
            box_loc = {
                'cls': c['cls'],
                'x1': c['x1'] - x1, 'x2': c['x2'] - x1,
                'y1': c['y1'] - y1, 'y2': c['y2'] - y1
            }
            
            patches.append({
                'image': crop,
                'boxes': [box_loc],
                'tile_idx': tile_idx,
                'z': z
            })
    except Exception as e:
        print(f"Error cutting patches: {e}")
        
    return patches

class IndexerThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict, int)
    
    def __init__(self, tiles, csv_root, atlas_array):
        super().__init__()
        self.tiles = tiles
        self.csv_root = csv_root
        self.atlas = atlas_array
        
    def run(self):
        tasks = []
        # 确保 csv_root 存在
        if not os.path.exists(self.csv_root):
            print(f"[FATAL] CSV Root does not exist: {self.csv_root}")
            self.finished_signal.emit({}, 0)
            return

        files = [f for f in os.listdir(self.csv_root) if f.endswith('.csv')]
        print(f"[DEBUG] Found {len(files)} CSV files in {self.csv_root}")

        for tile in self.tiles:
            t_idx = tile['list_index']
            base_name = tile['dir'].split('/')[-1]
            
            # 模糊匹配逻辑
            candidates = [f for f in files if base_name in f]
            if not candidates: 
                # print(f"[DEBUG] No CSV found for tile {base_name}")
                continue
            
            target_csv = candidates[0]
            # 优先选择带 _result 的文件
            for c in candidates:
                if "_result" in c: target_csv = c; break
            
            tasks.append((
                os.path.join(self.csv_root, target_csv), 
                t_idx, tile['origin_um'], PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE
            ))
            
        print(f"[DEBUG] Created {len(tasks)} indexing tasks.")
        
        region_index = {}
        total_cells = 0
        processed = 0
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(self.atlas,)) as ex:
                # 使用 future 对象来捕获子进程的异常
                futures = {ex.submit(process_single_csv, t): t for t in tasks}
                
                for future in futures:
                    try:
                        res = future.result() # 这里会重新抛出子进程的异常
                        processed += 1
                        self.progress_signal.emit(processed)
                        
                        if res:
                            for aid, lst in res.items():
                                if aid not in region_index: region_index[aid] = []
                                region_index[aid].extend(lst)
                                total_cells += len(lst)
                    except Exception as e:
                        print(f"[CRITICAL ERROR in Worker] {e}")
                        import traceback
                        traceback.print_exc()
                        
        except Exception as main_e:
            print(f"[CRITICAL ERROR in Main Thread] {main_e}")
            
        print(f"[DEBUG] Indexing finished. Total cells: {total_cells}")
        self.finished_signal.emit(region_index, total_cells)

class PatchLoaderThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(list)
    
    def __init__(self, cells, tiles, root_red, root_green):
        super().__init__()
        self.cells = cells
        self.tiles = tiles
        self.root_red = root_red
        self.root_green = root_green
        
    def run(self):
        tasks_map = {}
        for c in self.cells:
            key = (c['tile_idx'], c['z'])
            if key not in tasks_map: tasks_map[key] = []
            tasks_map[key].append(c)
            
        load_tasks = []
        for (t_idx, z), cell_list in tasks_map.items():
            tile_meta = self.tiles[t_idx]
            r_dir = os.path.join(self.root_red, tile_meta['dir'])
            g_dir = os.path.join(self.root_green, tile_meta['dir'])
            fr = sorted(os.listdir(r_dir))
            fg = sorted(os.listdir(g_dir))
            if 0 <= z < len(fr):
                load_tasks.append((
                    os.path.join(r_dir, fr[z]),
                    os.path.join(g_dir, fg[z]),
                    t_idx, z, cell_list
                ))
                
        final_patches = []
        processed = 0
        num_workers = max(1, multiprocessing.cpu_count() - 4)
        
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for batch_patches in ex.map(load_patches_worker, load_tasks):
                processed += 1
                self.progress_signal.emit(processed)
                final_patches.extend(batch_patches)
                
        for i, p in enumerate(final_patches):
            p['name'] = f"P_{i}_T{p['tile_idx']}_Z{p['z']}"
            
        self.finished_signal.emit(final_patches)

class SimpleAnnotator(QWidget):
    def __init__(self, coord_sys, ontology, root_red, root_green, csv_root, nav_atlas_path, save_root):
        super().__init__()
        self.coord_sys = coord_sys
        self.ontology = ontology
        self.root_red = root_red
        self.root_green = root_green
        self.csv_root = csv_root
        self.save_root = save_root
        
        print("Loading Atlas...")
        if nav_atlas_path.endswith('.mhd'):
            self.atlas = sitk.GetArrayFromImage(sitk.ReadImage(nav_atlas_path))
        else:
            self.atlas = tifffile.imread(nav_atlas_path)
            
        self.region_index = {}
        self.current_patches = []
        self.current_idx = 0
        self.undo_stack = [] 
        
        self.init_ui()
        QTimer.singleShot(100, self.start_indexing)

    def init_ui(self):
        self.setWindowTitle("Brain Annotator - Factor Mode")
        self.resize(1600, 900)
        main_layout = QHBoxLayout(self)
        
        # Left Panel
        left_panel = QWidget()
        l_lay = QVBoxLayout(left_panel)
        l_lay.addWidget(QLabel("<b>Brain Hierarchy</b>"))
        self.tree_widget = OntologyTreeWidget(self.ontology)
        self.tree_widget.region_selected.connect(self.on_region_selected)
        l_lay.addWidget(self.tree_widget)
        
        # Middle Panel (Viewer)
        mid_panel = QWidget()
        m_lay = QVBoxLayout(mid_panel)
        self.lbl_info = QLabel("Initializing...")
        m_lay.addWidget(self.lbl_info)
        
        self.viewer = ViewerModel()
        self.qt_viewer = QtViewer(self.viewer)
        self.viewer.dims.ndisplay = 2
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.font_size = 12
        m_lay.addWidget(self.qt_viewer)
        
        bot_bar = QHBoxLayout()
        self.btn_prev = QPushButton("<< Prev (A)"); self.btn_prev.clicked.connect(lambda: self.change_patch(-1))
        self.btn_next = QPushButton("Next (D) >>"); self.btn_next.clicked.connect(lambda: self.change_patch(1))
        self.btn_save = QPushButton("Save (S)"); self.btn_save.clicked.connect(self.save_current_patch)
        self.lbl_counter = QLabel("0 / 0")
        bot_bar.addWidget(self.btn_prev); bot_bar.addWidget(self.lbl_counter)
        bot_bar.addWidget(self.btn_next); bot_bar.addWidget(self.btn_save)
        m_lay.addLayout(bot_bar)
        
        # Right Panel (Controls)
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        r_lay = QVBoxLayout(right_panel)
        
        # 1. Filter
        filter_grp = QGroupBox("Filter View")
        f_lay = QVBoxLayout()
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("Show All")
        
        # Add dynamic filters
        unique_types = sorted(list(TYPE_FACTORS.keys()))
        for t in unique_types:
            self.combo_filter.addItem(f"All {t}s") # All Neurons
            # Also add specific "Red Neuron", "Green Neuron" if needed, but simple is better
        
        self.combo_filter.currentIndexChanged.connect(self.refresh_layer_view)
        f_lay.addWidget(self.combo_filter)
        filter_grp.setLayout(f_lay)
        r_lay.addWidget(filter_grp)
        
        # 2. Stats
        r_lay.addWidget(QLabel("<b>Region Statistics</b>"))
        self.table_stats = QTableWidget()
        self.table_stats.setColumnCount(2)
        self.table_stats.setHorizontalHeaderLabels(["Class", "Count"])
        r_lay.addWidget(self.table_stats)
        
        # 3. Shortcuts
        short_grp = QGroupBox("Controls")
        s_lay = QVBoxLayout()
        
        # Build shortcut text dynamically
        color_txt = ", ".join([f"{v['key']}={v['name']}" for v in COLOR_FACTORS.values()])
        type_txt = "<br>".join([f"<b>{v['key'].upper()}</b>: {k}" for k,v in TYPE_FACTORS.items()])
        
        s_text = f"""
        <b>Nav:</b> A / D<br>
        <b>Save:</b> S<br>
        <b>Undo:</b> Ctrl+Z<br>
        <b>Draw:</b> P | <b>Select:</b> M<br>
        <hr>
        <b>Set Color (Factor 1):</b><br>
        {color_txt}<br>
        <hr>
        <b>Set Type (Factor 2):</b><br>
        {type_txt}
        """
        lbl_short = QLabel(s_text)
        lbl_short.setTextFormat(Qt.RichText)
        s_lay.addWidget(lbl_short)
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
        """
        修复版：显式定义回调函数，避免 Napari 报错 'no attribute __name__'
        """
        
        # 1. 定义具体的包装函数 (Wrapper Functions)
        # Napari 会把 viewer 作为参数传进来，所以要接收它 (哪怕不用)
        def next_patch_callback(viewer):
            self.change_patch(1)
        
        def prev_patch_callback(viewer):
            self.change_patch(-1)
            
        def save_patch_callback(viewer):
            self.save_current_patch()

        # 2. 确保它们有名字 (虽然 def 定义的本身就有，加这一步双重保险)
        next_patch_callback.__name__ = "next_patch_callback"
        prev_patch_callback.__name__ = "prev_patch_callback"
        save_patch_callback.__name__ = "save_patch_callback"

        # 3. 绑定按键
        # 注意：这里直接传函数名，不要加括号 ()
        self.viewer.bind_key('d', next_patch_callback)
        self.viewer.bind_key('a', prev_patch_callback)
        self.viewer.bind_key('s', save_patch_callback)

    def modify_selection(self, viewer, factor, value):
        """
        核心逻辑：根据 Factor 修改选中 Box 的 ID
        factor: 'color' (value=0,1,2) 或 'type' (value=0,3,6...)
        """
        if not hasattr(self, 'shapes_layer'): return
        sel_idx = list(self.shapes_layer.selected_data)
        if not sel_idx:
            self.viewer.text_overlay.text = "Select a box first!"
            return

        self.push_undo(None)
        
        current_props = self.shapes_layer.properties
        current_edge = self.shapes_layer.edge_color
        
        count = 0
        for idx in sel_idx:
            old_id = int(current_props['class_id'][idx])
            
            # 反解当前状态 (ID = Base + Offset)
            # Base = (ID // 3) * 3
            # Offset = ID % 3
            cur_base = (old_id // 3) * 3
            cur_offset = old_id % 3
            
            new_id = old_id
            
            if factor == 'color':
                # 只改 offset，保持 base 不变
                new_id = cur_base + value
            elif factor == 'type':
                # 只改 base，保持 offset 不变
                new_id = value + cur_offset
                
            # 获取新配置
            cfg = CLASS_CONFIG.get(new_id)
            if not cfg:
                print(f"Warning: Calculated ID {new_id} not in config.")
                continue
                
            # Update Layer Properties
            current_props['class_id'][idx] = new_id
            current_props['label'][idx] = cfg['name']
            current_edge[idx] = list(np.array(COLOR_DICT.get(cfg['color'])))+[1.0]
            
            # Update Internal Data
            self.current_patches[self.current_idx]['boxes'][idx]['cls'] = new_id
            count += 1
            
        self.shapes_layer.properties = current_props
        self.shapes_layer.edge_color = current_edge
        self.shapes_layer.refresh()
        self.shapes_layer.refresh_text()
        
        self.viewer.text_overlay.text = f"Updated {count} boxes."

    # --- Standard Methods (Same as before) ---
    def start_indexing(self):
        self.pd = QProgressDialog("Indexing...", "Cancel", 0, len(self.coord_sys.tiles), self)
        self.pd.setWindowModality(Qt.WindowModal)
        self.pd.show()
        self.idx_worker = IndexerThread(self.coord_sys.tiles, self.csv_root, self.atlas)
        self.idx_worker.progress_signal.connect(self.pd.setValue)
        self.idx_worker.finished_signal.connect(self.indexing_done)
        self.idx_worker.start()

    def indexing_done(self, idx_data, count):
        self.pd.close()
        self.region_index = idx_data
        self.lbl_info.setText(f"Ready. Indexed {count} cells.")

    def on_region_selected(self, rid, rname):
        self.lbl_info.setText(f"Querying {rname}...")
        QApplication.processEvents()
        ids = self.ontology.get_all_descendant_ids(rid)
        cells = []
        for i in ids:
            if i in self.region_index: cells.extend(self.region_index[i])
        if not cells:
            self.lbl_info.setText(f"{rname}: No cells found.")
            return
        self.load_patches_from_cells(cells)

    def load_patches_from_cells(self, cells):
        counts = {}
        for c in cells:
            cls = c['cls']
            cfg = CLASS_CONFIG.get(cls, {'name': f'Unknown({cls})'})
            counts[cfg['name']] = counts.get(cfg['name'], 0) + 1
        self.update_stats_table(counts)
        
        unique = set((c['tile_idx'], c['z']) for c in cells)
        self.pd = QProgressDialog("Loading...", "Cancel", 0, len(unique), self)
        self.pd.setWindowModality(Qt.WindowModal)
        self.pd.show()
        self.loader = PatchLoaderThread(cells, self.coord_sys.tiles, self.root_red, self.root_green)
        self.loader.progress_signal.connect(self.pd.setValue)
        self.loader.finished_signal.connect(self.patches_loaded)
        self.loader.start()

    def patches_loaded(self, patches):
        self.pd.close()
        self.current_patches = patches
        self.current_idx = 0
        self.undo_stack = []
        self.lbl_info.setText(f"Loaded {len(patches)} patches.")
        self.show_patch()

    def update_stats_table(self, counts):
        self.table_stats.setRowCount(0)
        row = 0
        for k in sorted(counts.keys()):
            self.table_stats.insertRow(row)
            self.table_stats.setItem(row, 0, QTableWidgetItem(str(k)))
            self.table_stats.setItem(row, 1, QTableWidgetItem(str(counts[k])))
            row += 1

    def show_patch(self):
        if not self.current_patches: 
            self.viewer.layers.clear(); return
        data = self.current_patches[self.current_idx]
        self.viewer.layers.clear()
        self.viewer.add_image(data['image'], name='image')
        self.refresh_layer_view()
        self.lbl_counter.setText(f"{self.current_idx + 1} / {len(self.current_patches)}")
        self.viewer.text_overlay.text = data['name']

    def refresh_layer_view(self):
        if not self.current_patches: return
        data = self.current_patches[self.current_idx]
        filter_txt = self.combo_filter.currentText()
        
        shapes, classes, edges, labels = [], [], [], []
        
        for b in data['boxes']:
            cls_id = b['cls']
            cfg = CLASS_CONFIG.get(cls_id, CLASS_CONFIG[0]) # Default safe
            
            # Filtering
            if filter_txt != "Show All":
                if filter_txt.startswith("All ") and filter_txt.endswith("s"):
                    # e.g., "All Neurons" -> check cfg['type'] == "Neuron"
                    target_type = filter_txt[4:-1] 
                    if cfg['type'] != target_type: continue
                else:
                    # Exact match logic if needed, currently only All X supported
                    pass

            shapes.append([[b['y1'], b['x1']], [b['y2'], b['x2']]])
            classes.append(cls_id)
            edges.append(COLOR_DICT.get(cfg['color'], [1,1,1]))
            labels.append(cfg['name'])
            
        try: self.viewer.layers.remove('boxes')
        except: pass
        if not shapes: return

        self.shapes_layer = self.viewer.add_shapes(
            shapes, shape_type='rectangle', edge_width=2, edge_color=edges,
            face_color='transparent', name='boxes',
            text={'string': '{label}', 'color': 'white', 'anchor': 'upper_left', 'translation': [-5, 0], 'size': 8},
            properties={'label': labels, 'class_id': classes}
        )
        self.shapes_layer.events.data.connect(self.push_undo)

    def push_undo(self, event):
        if not self.current_patches: return
        data = self.current_patches[self.current_idx]
        self.undo_stack.append(copy.deepcopy(data['boxes']))
        if len(self.undo_stack) > 20: self.undo_stack.pop(0)

    def undo_action(self):
        if not self.undo_stack: return
        self.current_patches[self.current_idx]['boxes'] = self.undo_stack.pop()
        self.refresh_layer_view()
        self.viewer.text_overlay.text = "Undo."

    def change_patch(self, delta):
        if not self.current_patches: return
        self.current_idx = (self.current_idx + delta) % len(self.current_patches)
        self.undo_stack = []
        self.show_patch()

    def save_current_patch(self):
        if not self.current_patches: return
        data = self.current_patches[self.current_idx]
        base = data['name']
        
        # Sync from layer
        new_boxes = []
        if hasattr(self, 'shapes_layer'):
            for i, box in enumerate(self.shapes_layer.data):
                ys, xs = box[:, 0], box[:, 1]
                new_boxes.append({
                    'cls': self.shapes_layer.properties['class_id'][i],
                    'x1': min(xs), 'y1': min(ys), 'x2': max(xs), 'y2': max(ys)
                })
        data['boxes'] = new_boxes
        
        cv2.imwrite(os.path.join(self.save_root, 'images', base+'.jpg'), cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
        
        lines = []
        for b in new_boxes:
            cx = (b['x1'] + b['x2']) / 2 / PATCH_SIZE
            cy = (b['y1'] + b['y2']) / 2 / PATCH_SIZE
            w = (b['x2'] - b['x1']) / PATCH_SIZE
            h = (b['y2'] - b['y1']) / PATCH_SIZE
            lines.append(f"{b['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
        with open(os.path.join(self.save_root, 'labels', base+'.txt'), 'w') as f:
            f.write("\n".join(lines))
        self.viewer.text_overlay.text = f"Saved {base}!"