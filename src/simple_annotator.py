import os
import numpy as np
import tifffile
import pandas as pd
import cv2
import napari
import SimpleITK as sitk
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSplitter, QMessageBox, QProgressDialog, QFrame,QApplication)
from PyQt5.QtCore import Qt, QTimer

from .config import (PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE, 
                     CLASS_ID_MAP, TILE_SHAPE_PX)
from .utils import normalize_percentile
from .widgets import OntologyTreeWidget # <--- 导入新控件

# YOLO Patch 设定
PATCH_SIZE = 256

class SimpleAnnotator(QWidget):
    def __init__(self, coord_sys, ontology, root_red, root_green, csv_root, nav_atlas_path, save_root):
        super().__init__()
        self.coord_sys = coord_sys
        self.ontology = ontology
        self.root_red = root_red
        self.root_green = root_green
        self.csv_root = csv_root
        self.save_root = save_root
        
        # 1. 加载 Atlas 用于定位
        print("Loading Atlas...")
        if nav_atlas_path.endswith('.mhd'):
            self.atlas = sitk.GetArrayFromImage(sitk.ReadImage(nav_atlas_path))
        else:
            self.atlas = tifffile.imread(nav_atlas_path)
            
        # 2. 索引数据 (这是启动时最耗时的一步，但为了后续速度必须做)
        self.region_index = {} # { region_id: [ {tile_idx, z, local_x, local_y, class, ...} ] }
        self.index_all_detections()

        # 3. 内存中的 Patches
        self.current_patches = [] # List of {'image': rgb_array, 'boxes': [], 'meta': str}
        self.current_idx = 0

        self.init_ui()

    def index_all_detections(self):
        """
        启动时一次性读取所有CSV，并计算它们所属的 Atlas ID
        """
        print("--- Indexing Detections (This may take a moment) ---")
        # 遍历 csv_root 下的所有文件
        files = [f for f in os.listdir(self.csv_root) if f.endswith('.csv')]
        
        total_cells = 0
        
        # 这里的逻辑需要根据你的 CSV 命名规则调整
        # 假设文件名对应 Tile 名字，或者我们需要遍历 CoordinateSystem 里的 tiles
        for tile in self.coord_sys.tiles:
            t_idx = tile['list_index']
            # 尝试找到对应的 CSV
            # 这里假设 CSV 命名规则比较复杂，我们简略为遍历所有可能的 CSV
            # 如果你有确定的 CSV 对应 Tile 的逻辑，请在这里优化
            # 暂时沿用之前 loader 的逻辑：假设 CSV 就在 csv_root 下，需要匹配文件名
            
            # 为了简单，我们只演示核心逻辑：假设我们能找到 Tile 对应的 CSV
            # 你需要根据实际情况修改这里的 CSV 查找逻辑
            csv_name = f"{tile['dir'].split('/')[-1]}.csv" # 举例
            csv_path = os.path.join(self.csv_root, csv_name)
            
            # 如果找不到，尝试去 csv_root 搜索包含 dir 名字的文件
            if not os.path.exists(csv_path):
                candidates = [f for f in files if tile['dir'].split('/')[-1] in f]
                if candidates: csv_path = os.path.join(self.csv_root, candidates[0])
                else: continue

            try:
                # 读取 CSV (假设格式: filename, x1, y1, x2, y2, class, score, intensity, z_idx)
                df = pd.read_csv(csv_path, header=None, 
                                 names=['fname', 'x1', 'y1', 'x2', 'y2', 'cls', 'sc', 'int', 'z'])
                
                # 批量计算 Atlas 坐标
                # 1. Local ZYX -> Global UM -> Global Atlas Pixel
                # 注意：CSV 里的 z 通常是 local stack index (0-300)
                
                origin = tile['origin_um']
                
                # 向量化计算 (为了速度)
                z_local = df['z'].values - 1 # 修正索引
                cx_local = (df['x1'].values + df['x2'].values) / 2
                cy_local = (df['y1'].values + df['y2'].values) / 2
                
                gz_um = origin[0] + z_local * PIXEL_SIZE_RAW[0]
                gy_um = origin[1] + cy_local * PIXEL_SIZE_RAW[1]
                gx_um = origin[2] + cx_local * PIXEL_SIZE_RAW[2]
                
                az = (gz_um / PIXEL_SIZE_NAV_SAMPLE[0]).astype(int)
                ay = (gy_um / PIXEL_SIZE_NAV_SAMPLE[1]).astype(int)
                ax = (gx_um / PIXEL_SIZE_NAV_SAMPLE[2]).astype(int)
                
                # 边界检查
                np.clip(az, 0, self.atlas.shape[0]-1, out=az)
                np.clip(ay, 0, self.atlas.shape[1]-1, out=ay)
                np.clip(ax, 0, self.atlas.shape[2]-1, out=ax)
                
                # 获取 Atlas ID
                atlas_ids = self.atlas[az, ay, ax]
                
                # 存入索引
                for i, aid in enumerate(atlas_ids):
                    if aid == 0: continue
                    if aid not in self.region_index: self.region_index[aid] = []
                    
                    self.region_index[aid].append({
                        'tile_idx': t_idx,
                        'z': int(z_local[i]),
                        'x1': df.iloc[i]['x1'], 'y1': df.iloc[i]['y1'],
                        'x2': df.iloc[i]['x2'], 'y2': df.iloc[i]['y2'],
                        'cls': df.iloc[i]['cls']
                    })
                    total_cells += 1
                    
            except Exception as e:
                print(f"Error reading CSV {csv_path}: {e}")

        print(f"--- Indexing Complete. Found {total_cells} cells in {len(self.region_index)} regions. ---")

    def init_ui(self):
        self.setWindowTitle("Brain Annotator - Hierarchy Mode")
        self.resize(1400, 900)
        
        main_layout = QHBoxLayout(self) # 改为水平布局
        
        # --- Left: Ontology Tree ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        
        lbl_tree = QLabel("Brain Hierarchy")
        lbl_tree.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        left_layout.addWidget(lbl_tree)
        
        self.tree_widget = OntologyTreeWidget(self.ontology)
        self.tree_widget.region_selected.connect(self.on_region_tree_selected)
        left_layout.addWidget(self.tree_widget)
        
        # --- Right: Viewer & Controls ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Info Bar
        self.lbl_info = QLabel("Select a region from the tree to start.")
        self.lbl_info.setStyleSheet("font-size: 12px; color: #333; padding: 5px;")
        right_layout.addWidget(self.lbl_info)
        
        # Napari Viewer
        self.viewer = napari.Viewer(show=False)
        self.viewer.dims.ndisplay = 2
        self.viewer.text_overlay.visible = True
        right_layout.addWidget(self.viewer.window.qt_viewer)
        
        # Bottom Controls
        bot_bar = QHBoxLayout()
        self.btn_prev = QPushButton("<< Prev (A)"); self.btn_prev.clicked.connect(lambda: self.change_patch(-1))
        self.btn_next = QPushButton("Next (D) >>"); self.btn_next.clicked.connect(lambda: self.change_patch(1))
        self.btn_save = QPushButton("Save Patch (S)"); self.btn_save.clicked.connect(self.save_current_patch)
        self.lbl_counter = QLabel("0 / 0")
        
        bot_bar.addWidget(self.btn_prev); bot_bar.addWidget(self.lbl_counter)
        bot_bar.addWidget(self.btn_next); bot_bar.addWidget(self.btn_save)
        right_layout.addLayout(bot_bar)
        
        # Splitter to resize Left/Right
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050]) # 左侧给350px宽度
        
        main_layout.addWidget(splitter)
        
        self.bind_keys()

    def on_region_tree_selected(self, region_id, region_name):
        """
        树节点被点击时的回调
        """
        self.lbl_info.setText(f"Querying: {region_name} (ID: {region_id})...")
        # 强制界面刷新一下，否则会卡顿
        QApplication.processEvents()
        
        # 1. 获取该 ID 及其所有子 ID
        target_ids = self.ontology.get_all_descendant_ids(region_id)
        
        # 2. 收集所有相关细胞
        all_cells = []
        for tid in target_ids:
            if tid in self.region_index:
                all_cells.extend(self.region_index[tid])
                
        if not all_cells:
            QMessageBox.information(self, "Empty", f"No cells found in {region_name} (including sub-regions).")
            self.lbl_info.setText(f"Region: {region_name} - Empty.")
            return
            
        self.lbl_info.setText(f"Found {len(all_cells)} cells in {region_name} tree. Generating patches...")
        
        # 3. 调用加载逻辑 (复用之前的逻辑，只是输入变成了 cell list)
        self.load_patches_from_cells(all_cells)

    def load_patches_from_cells(self, cells):
        """
        根据传入的 cell list 生成 Patches (提取自原 load_region_patches 逻辑)
        """
        # Group by Tile -> Z
        tasks = {} 
        for c in cells:
            key = (c['tile_idx'], c['z'])
            if key not in tasks: tasks[key] = []
            tasks[key].append(c)

        self.current_patches = []
        
        pd = QProgressDialog("Generating Patches...", "Cancel", 0, len(tasks), self)
        pd.setWindowModality(Qt.WindowModal)
        pd.show()
        
        count = 0
        processed_tasks = 0
        
        for (tile_idx, z), cell_list in tasks.items():
            if pd.wasCanceled(): break
            
            try:
                # --- IO Loading Logic (Same as before) ---
                tile_meta = self.coord_sys.tiles[tile_idx]
                r_dir = os.path.join(self.root_red, tile_meta['dir'])
                g_dir = os.path.join(self.root_green, tile_meta['dir'])
                
                # 假设文件名按顺序
                files_r = sorted(os.listdir(r_dir)) 
                files_g = sorted(os.listdir(g_dir)) # 这里实际应该用更稳健的 matching
                
                if 0 <= z < len(files_r):
                    path_r = os.path.join(r_dir, files_r[z])
                    path_g = os.path.join(g_dir, files_g[z])
                    
                    img_r = tifffile.imread(path_r)
                    img_g = tifffile.imread(path_g)
                    
                    r8 = normalize_percentile(img_r)
                    g8 = normalize_percentile(img_g)
                    rgb = np.zeros((*r8.shape, 3), dtype=np.uint8)
                    rgb[..., 2] = r8 
                    rgb[..., 1] = g8 
                    
                    # --- Cutting Logic ---
                    for c in cell_list:
                        cx, cy = (c['x1']+c['x2'])//2, (c['y1']+c['y2'])//2
                        half = PATCH_SIZE // 2
                        y1, y2 = int(cy - half), int(cy + half)
                        x1, x2 = int(cx - half), int(cx + half)
                        
                        # Pad
                        pad_t, pad_l = max(0, -y1), max(0, -x1)
                        pad_b, pad_r = max(0, y2 - rgb.shape[0]), max(0, x2 - rgb.shape[1])
                        
                        crop = rgb[max(0,y1):min(y2, rgb.shape[0]), max(0,x1):min(x2, rgb.shape[1])]
                        if any([pad_t, pad_l, pad_b, pad_r]):
                            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                            
                        # Box Logic
                        box_loc = {
                            'cls': c['cls'],
                            'x1': c['x1'] - x1, 'x2': c['x2'] - x1,
                            'y1': c['y1'] - y1, 'y2': c['y2'] - y1
                        }
                        
                        self.current_patches.append({
                            'image': crop,
                            'boxes': [box_loc],
                            'name': f"T{tile_idx}_Z{z}_ID{count}"
                        })
                        count += 1
            except Exception as e:
                print(f"Error loading patch: {e}")
                
            processed_tasks += 1
            pd.setValue(processed_tasks)
            
        pd.close()
        self.lbl_info.setText(f"Loaded {len(self.current_patches)} patches into RAM.")
        self.current_idx = 0
        self.show_patch()

    def show_patch(self):
        if not self.current_patches: return
        
        data = self.current_patches[self.current_idx]
        img = data['image']
        
        self.viewer.layers.clear()
        self.viewer.add_image(img, name=data['name'])
        
        # Add Shapes
        shapes = []
        classes = []
        for b in data['boxes']:
            shapes.append([[b['y1'], b['x1']], [b['y2'], b['x2']]])
            classes.append(b['cls'])
            
        self.shapes_layer = self.viewer.add_shapes(
            shapes, 
            shape_type='rectangle',
            edge_width=2,
            edge_color='cyan',
            face_color='transparent',
            name='Detections',
            properties={'class': classes},
            text={'string': '{class}', 'color': 'cyan', 'size': 8}
        )
        self.shapes_layer.mode = 'SELECT'
        self.shapes_layer.bind_key('Delete', self.delete_box)
        
        self.lbl_counter.setText(f"{self.current_idx + 1} / {len(self.current_patches)}")
        self.viewer.text_overlay.text = data['name']

    def change_patch(self, delta):
        if not self.current_patches: return
        self.current_idx = (self.current_idx + delta) % len(self.current_patches)
        self.show_patch()

    def delete_box(self, layer):
        layer.remove_selected()

    def bind_keys(self):
        self.viewer.bind_key('d', lambda e: self.change_patch(1))
        self.viewer.bind_key('a', lambda e: self.change_patch(-1))
        self.viewer.bind_key('s', lambda e: self.save_current_patch())
        # 可以添加数字键修改类别
        for k, v in CLASS_ID_MAP.items():
            # 绑定逻辑略
            pass

    def save_current_patch(self):
        if not self.current_patches: return
        data = self.current_patches[self.current_idx]
        base_name = data['name']
        
        # 1. Save Image
        img_path = os.path.join(self.save_root, 'images', base_name + ".jpg")
        # Napari uses RGB, cv2 uses BGR
        cv2.imwrite(img_path, cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
        
        # 2. Save Labels (YOLO)
        txt_path = os.path.join(self.save_root, 'labels', base_name + ".txt")
        lines = []
        
        for i, box in enumerate(self.shapes_layer.data):
            # Napari shape data: [[y1, x1], [y2, x2]] or polygon
            ys = [p[0] for p in box]
            xs = [p[1] for p in box]
            y1, x1, y2, x2 = min(ys), min(xs), max(ys), max(xs)
            
            # Normalize to 0-1
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w/2
            cy = y1 + h/2
            
            nx = cx / PATCH_SIZE
            ny = cy / PATCH_SIZE
            nw = w / PATCH_SIZE
            nh = h / PATCH_SIZE
            
            cls_name = self.shapes_layer.properties['class'][i]
            cid = 0
            for k, v in CLASS_ID_MAP.items():
                if k in cls_name: 
                    cid = v
                    break
            
            lines.append(f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            
        with open(txt_path, 'w') as f:
            f.write("\n".join(lines))
            
        self.viewer.text_overlay.text = f"Saved {base_name}!"
        QTimer.singleShot(1000, lambda: setattr(self.viewer.text_overlay, 'text', base_name))