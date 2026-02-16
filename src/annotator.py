# src/annotator.py
#has navigation viewer, atlas overlay, region-based filtering, crop navigation, and search functionality.
import os
import numpy as np
import tifffile
import pandas as pd
import cv2
import napari
import SimpleITK as sitk  # 确保包含此行
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, 
                             QPushButton, QSplitter, QMessageBox, QProgressDialog, 
                             QLineEdit, QListWidget, QListWidgetItem, QGridLayout, 
                             QCheckBox, QSlider, QFrame)
from PyQt5.QtCore import Qt, QTimer

from .config import (PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE, TILE_SHAPE_PX, 
                     CROP_SHAPE_PX, CLASS_ID_MAP, CACHE_LIMIT_GB)
from .data_loader import TileCache, DataLoaderThread
from .coordinate_system import CoordinateSystem
from .ontology import Ontology

class BrainAnnotator(QWidget):
    def __init__(self, ontology_path, xml_path, root_red, root_green, csv_root, nav_sample_path, nav_atlas_path, save_root):
        super().__init__()
        
        self.root_red = root_red
        self.root_green = root_green
        self.csv_root = csv_root
        self.save_root = save_root
        
        # Init Ontology & Coords
        self.ontology = Ontology(ontology_path)
        if os.path.exists(xml_path):
            self.coord_sys = CoordinateSystem(xml_path)
        else:
            QMessageBox.critical(self, "Error", f"XML not found: {xml_path}")
            self.coord_sys = None

        # State
        self.tile_cache = TileCache(CACHE_LIMIT_GB)
        self.current_tile_idx = -1
        self.current_stack_r = None
        self.current_stack_g = None
        
        # Detections
        self.raw_detections = {}      
        self.filtered_detections = {} 
        
        # Region State
        self.current_atlas_id = 0
        self.crop_list = [] 
        self.current_crop_idx = 0
        
        # Contrast State
        self.contrast_r = [0, 65535]
        self.contrast_g = [0, 65535]

        # Layers Refs
        self.nav_atlas_layer = None

        os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'labels'), exist_ok=True)

        self.init_ui()
        # 延迟一小会儿加载数据，确保窗口先显示
        QTimer.singleShot(100, lambda: self.load_nav_data(nav_sample_path, nav_atlas_path))

    def init_ui(self):
        self.setWindowTitle("Brain Annotator - Region Guided Mode")
        self.resize(1600, 1000)
        self.layout_main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: Main Viewer ---
        self.viewer_main = napari.Viewer(show=False)
        self.viewer_main.dims.ndisplay = 2 
        self.viewer_main.text_overlay.visible = True
        self.viewer_main.text_overlay.text = "Select a region to start"
        splitter.addWidget(self.viewer_main.window.qt_viewer)

        # --- Right: Controls ---
        right_widget = QWidget()
        r_layout = QVBoxLayout(right_widget)

        # 1. Nav Viewer
        self.viewer_nav = napari.Viewer(show=False)
        self.viewer_nav.window.qt_viewer.setMinimumHeight(350)
        self.viewer_nav.dims.ndisplay = 2 
        self.viewer_nav.mouse_drag_callbacks.append(self.on_nav_click)
        r_layout.addWidget(self.viewer_nav.window.qt_viewer)

        # 2. Region Info
        g_info = QGroupBox("Region Info")
        l_info = QVBoxLayout()
        self.lbl_region_name = QLabel("None")
        self.lbl_region_name.setStyleSheet("font-weight:bold; color:blue; font-size:13px;")
        self.lbl_region_name.setWordWrap(True)
        l_info.addWidget(self.lbl_region_name)
        self.lbl_stats = QLabel("Region Count: -\nTile Total: -")
        l_info.addWidget(self.lbl_stats)
        lbl_help = QLabel("Shortcuts: [S] Save | [Del] Delete Box | [A/D] Prev/Next Crop")
        lbl_help.setStyleSheet("color: #666; font-size: 11px;")
        l_info.addWidget(lbl_help)
        g_info.setLayout(l_info)
        r_layout.addWidget(g_info)

        # 3. Crop Controls
        g_crop = QGroupBox("Crop Navigation")
        h_crop = QHBoxLayout()
        self.btn_prev_crop = QPushButton("<< (A)"); self.btn_prev_crop.clicked.connect(lambda: self.change_crop(-1))
        self.btn_next_crop = QPushButton(">> (D)"); self.btn_next_crop.clicked.connect(lambda: self.change_crop(1))
        self.lbl_crop_idx = QLabel("0 / 0")
        h_crop.addWidget(self.btn_prev_crop); h_crop.addWidget(self.lbl_crop_idx); h_crop.addWidget(self.btn_next_crop)
        g_crop.setLayout(h_crop)
        r_layout.addWidget(g_crop)
        
        # 4. Image Adjustment
        g_adj = QGroupBox("Image Adjustment")
        l_adj = QGridLayout()
        l_adj.addWidget(QLabel("Red Contrast:"), 0, 0)
        self.sl_red_min = self.create_slider(l_adj, 0, 1, self.on_contrast_change)
        self.sl_red_max = self.create_slider(l_adj, 0, 2, self.on_contrast_change, init_val=2000)
        l_adj.addWidget(QLabel("Green Contrast:"), 1, 0)
        self.sl_grn_min = self.create_slider(l_adj, 1, 1, self.on_contrast_change)
        self.sl_grn_max = self.create_slider(l_adj, 1, 2, self.on_contrast_change, init_val=2000)
        g_adj.setLayout(l_adj)
        r_layout.addWidget(g_adj)

        # 5. Search
        g_search = QGroupBox("Search Region")
        l_search = QVBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Type region name...")
        self.txt_search.textChanged.connect(self.perform_search)
        self.list_search = QListWidget()
        self.list_search.setMaximumHeight(100)
        self.list_search.itemClicked.connect(self.on_search_result_clicked)
        l_search.addWidget(self.txt_search); l_search.addWidget(self.list_search)
        g_search.setLayout(l_search)
        r_layout.addWidget(g_search)

        # 6. Tools (Added Atlas Toggle)
        g_tools = QGroupBox("Tools")
        l_tools = QHBoxLayout()
        self.chk_red = QCheckBox("Red"); self.chk_red.setChecked(True); self.chk_red.toggled.connect(self.update_viewer_visibility)
        self.chk_grn = QCheckBox("Green"); self.chk_grn.setChecked(True); self.chk_grn.toggled.connect(self.update_viewer_visibility)
        
        # --- 新增 Atlas 开关 ---
        self.chk_atlas = QCheckBox("Show Atlas"); self.chk_atlas.setChecked(True); self.chk_atlas.toggled.connect(self.toggle_atlas_visibility)
        
        self.btn_save = QPushButton("Save (S)"); self.btn_save.clicked.connect(self.save_current_crop)
        l_tools.addWidget(self.chk_red); l_tools.addWidget(self.chk_grn)
        l_tools.addWidget(self.chk_atlas) # Add to layout
        l_tools.addWidget(self.btn_save)
        g_tools.setLayout(l_tools)
        r_layout.addWidget(g_tools)

        r_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setSizes([1100, 500])
        self.layout_main.addWidget(splitter)
        
        self.bind_shortcuts()

    def create_slider(self, layout, row, col, callback, init_val=0):
        sl = QSlider(Qt.Horizontal)
        sl.setRange(0, 10000)
        sl.setValue(init_val)
        sl.valueChanged.connect(callback)
        layout.addWidget(sl, row, col)
        return sl

    def load_nav_data(self, sample_path, atlas_path):
        try:
            print(f"--- Loading Navigation Data ---")
            self.d_sample = tifffile.imread(sample_path)
            self.viewer_nav.add_image(self.d_sample, name='Nav Sample', colormap='gray', blending='additive')
            
            if os.path.exists(atlas_path):
                print(f"Atlas Path: {atlas_path}")
                if atlas_path.endswith('.mhd'):
                    self.d_atlas = sitk.GetArrayFromImage(sitk.ReadImage(atlas_path))
                else:
                    self.d_atlas = tifffile.imread(atlas_path)
                
                print(f"Atlas Loaded. Shape: {self.d_atlas.shape}")
                
                # Add Labels Layer (Visible by default now)
                self.nav_atlas_layer = self.viewer_nav.add_labels(self.d_atlas, name='Atlas', opacity=0.3)
                self.nav_atlas_layer.visible = True 
            else:
                print(f"[WARNING] Atlas file not found: {atlas_path}")

        except Exception as e:
            print(f"!!! FATAL ERROR in load_nav_data: {e}")
            import traceback
            traceback.print_exc()

    def toggle_atlas_visibility(self):
        if self.nav_atlas_layer:
            self.nav_atlas_layer.visible = self.chk_atlas.isChecked()

    # --- Interaction Logic (Same as before) ---
    def perform_search(self, text):
        self.list_search.clear()
        if len(text) < 2: return
        results = self.ontology.search(text)
        for nid, display_name in results:
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, nid)
            self.list_search.addItem(item)

    def on_search_result_clicked(self, item):
        target_id = item.data(Qt.UserRole)
        self.jump_to_atlas_id(target_id)

    def jump_to_atlas_id(self, target_id):
        if not hasattr(self, 'd_atlas'): return
        locs = np.where(self.d_atlas == target_id)
        if len(locs[0]) == 0:
            QMessageBox.warning(self, "Not Found", "Region empty in Atlas.")
            return
        cz, cy, cx = int(np.mean(locs[0])), int(np.mean(locs[1])), int(np.mean(locs[2]))
        self.viewer_nav.dims.set_current_step(0, cz)
        self.viewer_nav.camera.center = (0, cy, cx)
        self.viewer_nav.camera.zoom = 2
        self.trigger_region_load(cz, cy, cx, force_id=target_id)

    def on_nav_click(self, layer, event):
        if event.button != 1 or 'Shift' in event.modifiers: return
        c_z, c_y, c_x = self.viewer_nav.cursor.position
        atlas_id = 0
        if hasattr(self, 'd_atlas'):
            try:
                az, ay, ax = int(c_z), int(c_y), int(c_x)
                if 0 <= az < self.d_atlas.shape[0] and \
                   0 <= ay < self.d_atlas.shape[1] and \
                   0 <= ax < self.d_atlas.shape[2]:
                    atlas_id = self.d_atlas[az, ay, ax]
            except: pass
        self.trigger_region_load(int(c_z), int(c_y), int(c_x), force_id=atlas_id)

    def trigger_region_load(self, g_z, g_y, g_x, force_id=None):
        self.current_atlas_id = force_id if force_id is not None else 0
        region_name = self.ontology.get_name(self.current_atlas_id)
        self.lbl_region_name.setText(f"{region_name} (ID: {self.current_atlas_id})")
        if not self.coord_sys: return
        tid, _, _, _ = self.coord_sys.global_to_local(g_z, g_y, g_x)
        if tid is None:
            self.viewer_main.text_overlay.text = "Outside Raw Data."
            return
        if tid != self.current_tile_idx:
            self.load_tile_data(tid)
        else:
            self.process_loaded_tile()

    def load_tile_data(self, tile_idx):
        cached = self.tile_cache.get(tile_idx)
        if cached:
            self.current_stack_r, self.current_stack_g, self.raw_detections = cached
            self.current_tile_idx = tile_idx
            self.process_loaded_tile()
            return
        self.progress = QProgressDialog(f"Loading Tile {tile_idx}...", None, 0, 100, self)
        self.progress.show()
        meta = self.coord_sys.tiles[tile_idx]
        self.loader = DataLoaderThread(meta, self.root_red, self.root_green, self.csv_root)
        self.loader.data_loaded.connect(self.on_tile_loaded)
        self.loader.progress_update.connect(lambda v, m: self.progress.setValue(v))
        self.loader.start()

    def on_tile_loaded(self, r, g, dets, idx, fnames):
        self.progress.close()
        self.tile_cache.add(idx, r, g, dets)
        self.current_stack_r, self.current_stack_g, self.raw_detections = r, g, dets
        self.current_tile_idx = idx
        self.process_loaded_tile()

    def process_loaded_tile(self):
        if self.current_atlas_id == 0:
            self.viewer_main.text_overlay.text = "Background selected."
            self.lbl_stats.setText("Region Count: 0")
            return
        self.viewer_main.text_overlay.text = "Processing Region..."
        QTimer.singleShot(10, self._process_logic)

    def _process_logic(self):
        self.filtered_detections = {}
        total_in_region = 0
        tile_meta = self.coord_sys.tiles[self.current_tile_idx]
        origin_um = tile_meta['origin_um']
        res_raw = PIXEL_SIZE_RAW
        res_nav = PIXEL_SIZE_NAV_SAMPLE
        
        for z_local, df in self.raw_detections.items():
            if df.empty: continue
            cx_local = (df['x1'] + df['x2']) / 2
            cy_local = (df['y1'] + df['y2']) / 2
            gz_um = origin_um[0] + z_local * res_raw[0]
            gy_um = origin_um[1] + cy_local * res_raw[1]
            gx_um = origin_um[2] + cx_local * res_raw[2]
            az = (gz_um / res_nav[0]).astype(int)
            ay = (gy_um / res_nav[1]).astype(int)
            ax = (gx_um / res_nav[2]).astype(int)
            az = np.clip(az, 0, self.d_atlas.shape[0]-1)
            ay = np.clip(ay, 0, self.d_atlas.shape[1]-1)
            ax = np.clip(ax, 0, self.d_atlas.shape[2]-1)
            
            # Important: Check if d_atlas is loaded before accessing
            if hasattr(self, 'd_atlas'):
                atlas_ids = self.d_atlas[az, ay, ax]
                mask = (atlas_ids == self.current_atlas_id)
                if np.any(mask):
                    valid_df = df[mask].copy()
                    self.filtered_detections[z_local] = valid_df
                    total_in_region += len(valid_df)

        total_raw = sum([len(df) for df in self.raw_detections.values()])
        self.lbl_stats.setText(f"Region Count: {total_in_region}\nTile Total: {total_raw}")

        active_spots = []
        for z, df in self.filtered_detections.items():
            for _, r in df.iterrows():
                active_spots.append((z, (r['y1']+r['y2'])//2, (r['x1']+r['x2'])//2))
        
        if not active_spots:
            self.viewer_main.text_overlay.text = "No cells found in this region."
            self.crop_list = []
            self.show_blank_crop()
            return

        crop_set = set()
        step_y, step_x = CROP_SHAPE_PX[0]//2, CROP_SHAPE_PX[1]//2
        for z, cy, cx in active_spots:
            gy = int(cy // step_y) * step_y
            gx = int(cx // step_x) * step_x
            gy = min(max(0, gy), TILE_SHAPE_PX[0] - CROP_SHAPE_PX[0])
            gx = min(max(0, gx), TILE_SHAPE_PX[1] - CROP_SHAPE_PX[1])
            crop_set.add((z, gy, gx))
            
        self.crop_list = sorted(list(crop_set))
        self.current_crop_idx = 0
        self.show_crop()

    def show_crop(self):
        if not self.crop_list: return
        z, y, x = self.crop_list[self.current_crop_idx]
        self.current_z = z
        h, w = CROP_SHAPE_PX
        z_safe = min(max(0, z), self.current_stack_r.shape[0]-1)
        img_r = self.current_stack_r[z_safe, y:y+h, x:x+w]
        img_g = self.current_stack_g[z_safe, y:y+h, x:x+w]
        self.viewer_main.layers.clear()
        self.viewer_main.add_image(img_r, name='Red', colormap='red', blending='additive')
        self.viewer_main.add_image(img_g, name='Green', colormap='green', blending='additive')
        self.apply_contrast()
        self.lbl_crop_idx.setText(f"Crop {self.current_crop_idx+1}/{len(self.crop_list)}")
        self.load_crop_boxes(z, y, x)
        self.viewer_main.text_overlay.text = ""

    def load_crop_boxes(self, z, crop_y, crop_x):
        self.shapes_layer = self.viewer_main.add_shapes(
            name='Detections', edge_width=2, face_color='transparent',
            text={'string': '{class_name}', 'color': 'cyan', 'size': 8}
        )
        self.shapes_layer.mode = 'SELECT'
        self.shapes_layer.bind_key('Delete', self.on_delete_key)
        
        if z in self.filtered_detections:
            df = self.filtered_detections[z]
            mask = (df['x2'] > crop_x) & (df['x1'] < crop_x + CROP_SHAPE_PX[1]) & \
                   (df['y2'] > crop_y) & (df['y1'] < crop_y + CROP_SHAPE_PX[0])
            subset = df[mask]
            boxes, props = [], {'class_name': []}
            for _, r in subset.iterrows():
                ly1, lx1 = max(0, r['y1'] - crop_y), max(0, r['x1'] - crop_x)
                ly2, lx2 = min(CROP_SHAPE_PX[0], r['y2'] - crop_y), min(CROP_SHAPE_PX[1], r['x2'] - crop_x)
                if (lx2-lx1) > 2 and (ly2-ly1) > 2:
                    boxes.append([[ly1, lx1], [ly2, lx2]])
                    props['class_name'].append(r['class_name'])
            if boxes:
                self.shapes_layer.add(boxes, shape_type='rectangle', edge_color='cyan')
                self.shapes_layer.properties = props

    def change_crop(self, delta):
        if not self.crop_list: return
        self.current_crop_idx = (self.current_crop_idx + delta) % len(self.crop_list)
        self.show_crop()

    def show_blank_crop(self):
        self.viewer_main.layers.clear()
        z = np.zeros(CROP_SHAPE_PX, dtype=np.uint8)
        self.viewer_main.add_image(z, name='Empty', colormap='gray')

    def on_contrast_change(self):
        max_val = 65535.0
        rmin = (self.sl_red_min.value() / 10000.0) * max_val
        rmax = (self.sl_red_max.value() / 10000.0) * max_val
        gmin = (self.sl_grn_min.value() / 10000.0) * max_val
        gmax = (self.sl_grn_max.value() / 10000.0) * max_val
        if rmin >= rmax: rmax = rmin + 100
        if gmin >= gmax: gmax = gmin + 100
        self.contrast_r = [rmin, rmax]
        self.contrast_g = [gmin, gmax]
        self.apply_contrast()

    def apply_contrast(self):
        try:
            if 'Red' in self.viewer_main.layers:
                self.viewer_main.layers['Red'].contrast_limits = self.contrast_r
            if 'Green' in self.viewer_main.layers:
                self.viewer_main.layers['Green'].contrast_limits = self.contrast_g
        except: pass

    def update_viewer_visibility(self):
        for l in self.viewer_main.layers:
            if l.name == 'Red': l.visible = self.chk_red.isChecked()
            if l.name == 'Green': l.visible = self.chk_grn.isChecked()

    def on_delete_key(self, layer):
        layer.remove_selected()

    def save_current_crop(self):
        if not self.crop_list: return
        z, cy, cx = self.crop_list[self.current_crop_idx]
        base = f"crop_T{self.current_tile_idx}_Z{z}_Y{cy}_X{cx}"
        img_out = np.zeros((*CROP_SHAPE_PX, 3), dtype=np.uint8)
        z_safe = min(max(0, z), self.current_stack_r.shape[0]-1)
        r_slice = self.current_stack_r[z_safe, cy:cy+CROP_SHAPE_PX[0], cx:cx+CROP_SHAPE_PX[1]]
        g_slice = self.current_stack_g[z_safe, cy:cy+CROP_SHAPE_PX[0], cx:cx+CROP_SHAPE_PX[1]]
        r_8 = (r_slice // 256).astype(np.uint8)
        g_8 = (g_slice // 256).astype(np.uint8)
        if self.chk_red.isChecked(): img_out[..., 2] = r_8
        if self.chk_grn.isChecked(): img_out[..., 1] = g_8
        cv2.imwrite(os.path.join(self.save_root, 'images', base+'.tif'), img_out)
        lines = []
        if self.shapes_layer:
            for i, box in enumerate(self.shapes_layer.data):
                ys, xs = [p[0] for p in box], [p[1] for p in box]
                y1, x1, y2, x2 = min(ys), min(xs), max(ys), max(xs)
                cls_name = self.shapes_layer.properties['class_name'][i]
                nx, ny = (x1 + (x2-x1)/2)/CROP_SHAPE_PX[1], (y1 + (y2-y1)/2)/CROP_SHAPE_PX[0]
                nw, nh = (x2-x1)/CROP_SHAPE_PX[1], (y2-y1)/CROP_SHAPE_PX[0]
                cid = 99
                for k,v in CLASS_ID_MAP.items(): 
                    if k in cls_name: cid = v
                lines.append(f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
        with open(os.path.join(self.save_root, 'labels', base+'.txt'), 'w') as f:
            f.write("\n".join(lines))
        self.viewer_main.text_overlay.text = f"Saved {base}"
        QTimer.singleShot(1500, lambda: setattr(self.viewer_main.text_overlay, 'text', ''))

    def bind_shortcuts(self):
        self.viewer_main.bind_key('s', lambda e: self.save_current_crop())
        self.viewer_main.bind_key('a', lambda e: self.change_crop(-1))
        self.viewer_main.bind_key('d', lambda e: self.change_crop(1))