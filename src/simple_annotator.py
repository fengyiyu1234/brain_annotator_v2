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

from .config import (PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE, TILE_SHAPE_PX, PATCH_SIZE)
from .utils import normalize_percentile
from .widgets import OntologyTreeWidget
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QFormLayout # 确保引入了这些

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
def process_single_csv(args):
    """
    独立 Worker 函数
    Args:
        args: tuple (file_path, tile_offsets_px)
    """
    file_path, tile_offsets_px, tile_idx = args
    print(f"Processing CSV: {file_path}") # 调试时可以打开，生产环境建议注释以减少IO
    try:
        df = pd.read_csv(file_path, header=None, 
                         names=['filename', 'x1', 'y1', 'x2', 'y2', 'cls', 'conf', 'intensity', 'z'])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

    results = {}
    
    for idx, row in df.iterrows():
        fname = row['filename']
        
        parts = fname.split('_')
        off_x_px, off_y_px = 0.0, 0.0
        
        if len(parts) >= 2:
            key = f"{parts[0]}_{parts[1]}"
            # 从字典获取，如果没有则默认为 (0,0)
            if tile_offsets_px and key in tile_offsets_px:
                off_x_px, off_y_px = tile_offsets_px[key]
        
        # 计算全局像素坐标
        gx = off_x_px + row['x1']
        gy = off_y_px + row['y1']
        
        # 构建结果
        aid = int(idx)
        if aid not in results: results[aid] = []
        
        # 简单的类别映射
        cls_name = str(row['cls']).strip().lower()
        if 'green' in cls_name: cls_val = 1
        elif 'yellow' in cls_name: cls_val = 2
        else: cls_val = 0 # red
        
        results[aid].append({
            'filename': fname,
            'z': int(row['z']),
            'x1': row['x1'], 'y1': row['y1'],
            'x2': row['x2'], 'y2': row['y2'],
            'gx': gx,
            'gy': gy,
            'cls': cls_val,
            'name': f"Cell {aid}",
            'tile_idx': tile_idx
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
            
            # 创建结果字典，首先复制原始 cell (c) 的所有信息 (包含 gx, gy, x1, y1 等)
            patch_item = c.copy()
            # 然后覆盖/添加图像和框信息
            patch_item.update({
                'image': crop,
                'boxes': [box_loc], # 这里暂时只存了中心这个框，show_patch 会重写它
                'tile_idx': tile_idx,
                'z': z
            })
            patches.append(patch_item)

    except Exception as e:
        print(f"Error cutting patches: {e}")
        
    return patches

class IndexerThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict, int)
    
    def __init__(self, tiles, csv_root, atlas_array, tile_offsets_px):
        super().__init__()
        self.tiles = tiles
        self.csv_root = csv_root
        self.atlas = atlas_array
        self.tile_offsets_px = tile_offsets_px
        
    def run(self):
        tasks = []
        if not os.path.exists(self.csv_root):
            print(f"[FATAL] CSV Root does not exist: {self.csv_root}")
            self.finished_signal.emit({}, 0)
            return

        files = [f for f in os.listdir(self.csv_root) if f.endswith('.csv')]
        print(f"[DEBUG] Found {len(files)} CSV files in {self.csv_root}")

        for tile in self.tiles:
            t_idx = tile['list_index']
            base_name = tile['dir'].split('/')[-1]
            candidates = [f for f in files if base_name in f]
            if not candidates: 
                print(f"[DEBUG] No CSV found for tile {base_name}")
                continue
            
            target_csv = candidates[0]
            for c in candidates:
                if "_result" in c: target_csv = c; break
            
            tasks.append((
                os.path.join(self.csv_root, target_csv), 
                self.tile_offsets_px,
                t_idx
            ))
            
        print(f"[DEBUG] Created {len(tasks)} indexing tasks.")
        
        region_index = {}
        total_cells = 0
        processed = 0
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(self.atlas,)) as ex:
                futures = {ex.submit(process_single_csv, t): t for t in tasks}
                for future in futures:
                    try:
                        res = future.result() 
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
        
        self.info_panel = QWidget()
        self.info_layout = QFormLayout()
        self.info_panel.setLayout(self.info_layout)
        
        # 显示当前选中的类别
        self.lbl_current_class = QLabel("None")
        self.lbl_current_class.setStyleSheet("font-weight: bold; color: blue; font-size: 14px;")
        self.info_layout.addRow("Selected Class:", self.lbl_current_class)
        
        # 显示当前坐标
        self.lbl_coords = QLabel("-")
        self.info_layout.addRow("Position (XYZ):", self.lbl_coords)
        
        # 将这个面板加到你的右侧布局 layout 中 (在按钮上方或下方)
        r_lay.addWidget(self.info_panel)

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

    def bind_layer_keys(self):
        """
        每次 show_patch 生成新的 shapes_layer 后，
        必须重新绑定针对该图层的快捷键 (删除、修改类别)。
        """
        if self.shapes_layer is None:
            return

        # --- 1. 绑定 Delete 键 (删除选中框) ---
        def on_delete(layer):
            layer.remove_selected()
        
        on_delete.__name__ = "delete_selected"
        self.shapes_layer.bind_key('Delete', on_delete)

        # --- 2. 绑定数字键 (修改类别 1-6) ---
        # 我们使用循环来绑定，这样代码更整洁
        # 注意：label ID 是 0-5，键盘按键是 '1'-'6'
        
        for i in range(6):
            key_char = str(i + 1) # 键盘上的 '1', '2'...
            class_id = i          # 对应的 ID 0, 1...
            
            # 定义回调函数 (利用默认参数捕获当前的 class_id)
            def change_cls(layer, cid=class_id):
                self.update_class_of_selection(cid)
            
            # 必须给函数起个独立的名字，否则 Napari 会报错
            change_cls.__name__ = f"set_class_{class_id}"
            
            # 绑定到当前图层
            self.shapes_layer.bind_key(key_char, change_cls)
            
        print("Layer keys (Delete, 1-6) rebound to new layer.")

    def on_box_select(self, event):
        """
        当在 Napari 画布上选中/取消选中框时触发。
        功能：更新右侧面板的 Label，显示当前选中的细胞类别。
        """
        # 获取当前被选中的框的索引集合
        selected_indices = list(self.shapes_layer.selected_data)        
        # 1. 如果没选中任何东西
        if len(selected_indices) == 0:
            self.lbl_current_class.setText("None")
            return            
        # 2. 如果选中了多个
        if len(selected_indices) > 1:
            self.lbl_current_class.setText(f"Multiple ({len(selected_indices)})")
            return            
        # 3. 如果只选中了一个 (显示详情)
        idx = selected_indices[0]
        current_cls = self.shapes_layer.properties['class_id'][idx]
        ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
        cls_name = ID_TO_NAME.get(int(current_cls), str(current_cls))

        self.lbl_current_class.setText(f"{cls_name} (ID: {current_cls})")

    def update_class_of_selection(self, new_class_id):
        """
        当你按下数字键 (1, 2, 3...) 时调用此函数。
        功能：修改当前选中框的类别 ID，并立即更新颜色和 UI。
        """
        selected_indices = list(self.shapes_layer.selected_data)
        if not selected_indices: 
            print("No box selected.")
            return
        color_map = {0: 'red', 1: 'green', 2: 'yellow', 3: 'magenta', 4: 'cyan', 5: 'blue'}
        new_color = color_map.get(new_class_id, 'white')
        # --- 核心修改步骤 ---
        # 1. 更新数据属性 (Properties)
        current_props = self.shapes_layer.properties
        for idx in selected_indices:
            current_props['class_id'][idx] = new_class_id        
        # 必须重新赋值 properties 才能触发 Napari 内部更新
        self.shapes_layer.properties = current_props        
        # 2. 立即更新边框颜色 (Visuals)
        # Napari 的 edge_color 是一个 numpy 数组，直接修改对应索引的颜色
        current_colors = self.shapes_layer.edge_color
        for idx in selected_indices:
            current_colors[idx] = str(new_color) # 或者转换成 RGBA            
        # 强制刷新颜色
        self.shapes_layer.edge_color = current_colors
        self.shapes_layer.refresh()        
        # 3. 更新 UI 反馈
        ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
        cls_name = ID_TO_NAME.get(new_class_id, str(new_class_id))
        self.lbl_current_class.setText(f"{cls_name} (ID: {new_class_id}) [SAVED]")
        
        print(f"Updated {len(selected_indices)} cells to class {new_class_id}")

    # --- Standard Methods (Same as before) ---
    def start_indexing(self):
        self.pd = QProgressDialog("Indexing...", "Cancel", 0, len(self.coord_sys.tiles), self)
        self.pd.setWindowModality(Qt.WindowModal)
        self.pd.show()
        self.idx_worker = IndexerThread(self.coord_sys.tiles, self.csv_root, self.atlas, self.coord_sys.tile_offsets_px)
        self.idx_worker.progress_signal.connect(self.pd.setValue)
        self.idx_worker.finished_signal.connect(self.indexing_done)
        self.idx_worker.start()

    def build_spatial_index(self, all_cells_list):
        print("Building KDTree (Global Microns)...")
        # 使用 gx, gy (全局微米) 构建树
        self.coords_array = np.array([[c['gx'], c['gy']] for c in all_cells_list]) 
        self.cell_data_db = all_cells_list 
        self.tree = cKDTree(self.coords_array)
        print("KDTree built.")

    def indexing_done(self, idx_data, count):
        self.pd.close()
        self.region_index = idx_data
        self.lbl_info.setText(f"Ready. Indexed {count} cells.")
        # --- 构建 KDTree ---
        # 1. 把字典里的细胞全部提取到一个大列表里
        all_cells = []
        for region_id, cells in idx_data.items():
            all_cells.extend(cells)

        if all_cells:
            self.build_spatial_index(all_cells)
        else:
            print("No cells indexed, skipping KDTree build.")

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
        """
        [调试版] 假设 gx, gy 就是全局像素坐标，不除以分辨率。
        """
        if not self.current_patches: return
        
        target = self.current_patches[self.current_idx]
        
        center_gx = target['gx']
        center_gy = target['gy']
        
        search_radius = PATCH_SIZE / 2 + 50 
        
        # 注意：如果 build_spatial_index 用的是 gx/gy，这里单位必须一致
        indices = self.tree.query_ball_point([center_gx, center_gy], r=search_radius)
        
        layer_boxes = []
        layer_classes = []
        layer_colors = []
        all_box_dicts = [] 
        
        color_map = {0: 'red', 1: 'green', 2: 'yellow', 3: 'magenta', 4: 'cyan', 5: 'blue'}
        
        # Patch 中心 (256 / 2 = 128)
        patch_center_px = PATCH_SIZE / 2
        
        print(f"--- Debug Patch {self.current_idx} ---")
        print(f"Center Global: {center_gx:.1f}, {center_gy:.1f}")

        for i, idx in enumerate(indices):
            cell = self.cell_data_db[idx]
            if cell['z'] != target['z']: continue
            delta_x = cell['gx'] - center_gx
            delta_y = cell['gy'] - center_gy
            cx_local = patch_center_px + delta_x
            cy_local = patch_center_px + delta_y
            
            #计算 Box 宽高 (假设 x1, x2 是像素)
            w_px = cell['x2'] - cell['x1']
            h_px = cell['y2'] - cell['y1']
            
            x1_loc = cx_local - w_px / 2
            x2_loc = cx_local + w_px / 2
            y1_loc = cy_local - h_px / 2
            y2_loc = cy_local + h_px / 2
            
            # 5. 边界过滤 (保留部分在视野内的)
            if x2_loc < 0 or x1_loc > PATCH_SIZE or y2_loc < 0 or y1_loc > PATCH_SIZE:
                continue

            # Napari 格式: [[y1, x1], [y2, x2]]
            layer_boxes.append([[y1_loc, x1_loc], [y2_loc, x2_loc]])
            layer_classes.append(cell['cls'])
            layer_colors.append(color_map.get(cell['cls'], 'white'))
            
            all_box_dicts.append({
                'cls': cell['cls'],
                'x1': x1_loc, 'y1': y1_loc,
                'x2': x2_loc, 'y2': y2_loc
            })

        self.current_patches[self.current_idx]['boxes'] = all_box_dicts

        # Napari 更新
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        
        self.image_layer = self.viewer.add_image(target['image'], name='Patch Image')
        
        if layer_boxes:
            self.shapes_layer = self.viewer.add_shapes(
                layer_boxes,
                shape_type='rectangle',
                edge_width=2,
                edge_color=layer_colors,
                face_color=[0,0,0,0],
                properties={'class_id': layer_classes}, 
                name='boxes'
            )
            self.shapes_layer.events.highlight.connect(self.on_box_select)
            self.shapes_layer.events.data.connect(self.push_undo) 
            self.bind_layer_keys()
        else:
            self.shapes_layer = None

        self.lbl_counter.setText(f"{self.current_idx + 1} / {len(self.current_patches)}")
        self.lbl_coords.setText(f"GX:{int(center_gx)} GY:{int(center_gy)}")

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
        if not shapes: 
            self.shapes_layer = None
            return

        self.shapes_layer = self.viewer.add_shapes(
            shapes, shape_type='rectangle', edge_width=2, edge_color=edges,
            face_color='transparent', name='boxes',
            text={'string': '{label}', 'color': 'white', 'anchor': 'upper_left', 'translation': [-5, 0], 'size': 8},
            properties={'label': labels, 'class_id': classes}
        )
        self.shapes_layer.events.highlight.connect(self.on_box_select)
        self.shapes_layer.events.data.connect(self.push_undo)
        self.bind_layer_keys()

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
        
        # 1. 保存图片
        # 确保目录存在
        img_save_dir = os.path.join(self.save_root, 'images')
        lbl_save_dir = os.path.join(self.save_root, 'labels')
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(lbl_save_dir, exist_ok=True)

        cv2.imwrite(os.path.join(img_save_dir, base+'.jpg'), cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
        
        # 2. 从 Napari 图层获取最新的框 (用户可能修改过)
        # 如果没有图层，说明没有框，或者被用户删光了
        current_boxes = []
        current_classes = []
        
        if hasattr(self, 'shapes_layer') and self.shapes_layer is not None:
            # shapes_layer.data 返回的是 [[y1, x1], [y2, x2]] 格式
            current_boxes = self.shapes_layer.data
            current_classes = self.shapes_layer.properties['class_id']
        
        lines = []
        for i, box in enumerate(current_boxes):
            cls_id = current_classes[i]
            
            # Napari data is [[y1, x1], [y2, x2], [y2, x1], [y1, x2]] ... (4 points)
            # 或者 [[y1, x1], [y2, x2]] (2 points)取决于形状类型
            # 最稳健的方法是取 min/max
            ys = box[:, 0]
            xs = box[:, 1]
            
            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)
            
            # 1. 计算原始中心点 (未切断前)
            raw_cx = (x1 + x2) / 2
            raw_cy = (y1 + y2) / 2
            
            # 2. 规则：如果中心点完全在图片外，丢弃 (防止邻居 Patch 重复训练)
            if raw_cx < 0 or raw_cx > PATCH_SIZE or raw_cy < 0 or raw_cy > PATCH_SIZE:
                continue

            # 3. 规则：坐标截断 (Clip) 到 [0, PATCH_SIZE]
            # YOLO 格式不允许负数或大于1的数
            x1 = max(0, min(PATCH_SIZE, x1))
            y1 = max(0, min(PATCH_SIZE, y1))
            x2 = max(0, min(PATCH_SIZE, x2))
            y2 = max(0, min(PATCH_SIZE, y2))
            
            # 4. 规则：防止退化 (比如 x1=x2，宽度为0)
            if x2 <= x1 or y2 <= y1:
                continue
                
            # --- 转换为 YOLO 格式 (Normalized Center X, Y, W, H) ---
            final_w = x2 - x1
            final_h = y2 - y1
            final_cx = x1 + final_w / 2
            final_cy = y1 + final_h / 2
            
            # 归一化 (除以 256)
            norm_cx = final_cx / PATCH_SIZE
            norm_cy = final_cy / PATCH_SIZE
            norm_w = final_w / PATCH_SIZE
            norm_h = final_h / PATCH_SIZE
            
            # 必须限制在 0-1 之间 (浮点数精度误差可能导致 1.000001)
            norm_cx = min(max(norm_cx, 0), 1)
            norm_cy = min(max(norm_cy, 0), 1)
            norm_w = min(max(norm_w, 0), 1)
            norm_h = min(max(norm_h, 0), 1)
            
            lines.append(f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        with open(os.path.join(lbl_save_dir, base+'.txt'), 'w') as f:
            f.write("\n".join(lines))
            
        self.viewer.text_overlay.text = f"Saved {base}! ({len(lines)} boxes)"