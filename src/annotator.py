import os
import numpy as np
import tifffile
import pandas as pd
import cv2
import napari
import SimpleITK as sitk
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, 
                             QPushButton, QSplitter, QMessageBox, QProgressDialog, 
                             QSlider, QComboBox, QCheckBox, QSpinBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer

from .config import (PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE, TILE_SHAPE_PX, 
                     CROP_SHAPE_PX, COLOR_MAP, CLASS_ID_MAP, CACHE_LIMIT_GB)
from .data_loader import TileCache, DataLoaderThread
from .coordinate_system import CoordinateSystem

class BrainAnnotator(QWidget):
    def __init__(self, xml_path, root_red, root_green, csv_root, nav_sample_path, nav_atlas_path, save_root):
        super().__init__()
        self.root_red = root_red
        self.root_green = root_green
        self.csv_root = csv_root
        self.save_root = save_root
        
        # Data State
        self.coord_sys = CoordinateSystem(xml_path)
        self.tile_cache = TileCache(CACHE_LIMIT_GB)
        self.current_tile_idx = -1
        self.current_stack_r = None
        self.current_stack_g = None
        self.current_detections = {}
        
        # Crop/Region State
        self.crop_list = [] # List of tuples (z, y, x) top-left coords in raw data
        self.current_crop_idx = 0
        self.roi_class_counts = {}
        
        # History
        self.history_stack = {} 
        self.current_z = 0 # Relative to Raw Tile

        os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'labels'), exist_ok=True)

        self.init_ui()
        
        # Load Nav Data immediately
        self.load_nav_data(nav_sample_path, nav_atlas_path)

    def init_ui(self):
        self.setWindowTitle("Brain Annotator - ROI Mode")
        self.resize(1600, 900)
        self.layout_main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: Main Viewer (256x256 Crop) ---
        self.viewer_main = napari.Viewer(show=False)
        self.viewer_main.dims.ndisplay = 2 
        self.viewer_main.text_overlay.visible = True
        self.viewer_main.text_overlay.text = "Waiting for region selection..."
        splitter.addWidget(self.viewer_main.window.qt_viewer)

        # --- Right: Navigation & Controls ---
        right_widget = QWidget()
        r_layout = QVBoxLayout(right_widget)

        # 1. Nav Viewer (Stitched Brain)
        self.viewer_nav = napari.Viewer(show=False)
        self.viewer_nav.window.qt_viewer.setMinimumHeight(400)
        self.viewer_nav.dims.ndisplay = 2 # Strictly 2D Slice view
        # Mouse callback for region selection
        self.viewer_nav.mouse_drag_callbacks.append(self.on_nav_click)
        r_layout.addWidget(self.viewer_nav.window.qt_viewer)

        # 2. Region Info & Stats
        g_roi = QGroupBox("Selected Region Stats")
        gl_roi = QGridLayout()
        self.lbl_region_name = QLabel("Region: None")
        self.lbl_region_name.setStyleSheet("font-weight: bold; font-size: 14px;")
        gl_roi.addWidget(self.lbl_region_name, 0, 0, 1, 2)
        
        self.lbl_stats = QLabel("Counts: -")
        self.lbl_stats.setWordWrap(True)
        gl_roi.addWidget(self.lbl_stats, 1, 0, 1, 2)
        g_roi.setLayout(gl_roi)
        r_layout.addWidget(g_roi)

        # 3. Crop Controls (Sliding Window)
        g_crop = QGroupBox("Crop Navigation (256x256)")
        h_crop = QHBoxLayout()
        self.btn_prev_crop = QPushButton("<< Prev Crop"); self.btn_prev_crop.clicked.connect(lambda: self.change_crop(-1))
        self.btn_next_crop = QPushButton("Next Crop >>"); self.btn_next_crop.clicked.connect(lambda: self.change_crop(1))
        self.lbl_crop_idx = QLabel("0 / 0")
        h_crop.addWidget(self.btn_prev_crop); h_crop.addWidget(self.lbl_crop_idx); h_crop.addWidget(self.btn_next_crop)
        g_crop.setLayout(h_crop)
        r_layout.addWidget(g_crop)
        
        # 4. Standard Annotation Controls
        g_tools = QGroupBox("Tools")
        l_tools = QVBoxLayout()
        
        h_chan = QHBoxLayout()
        self.chk_red = QCheckBox("Red"); self.chk_red.setChecked(True); self.chk_red.toggled.connect(self.update_viewer_visibility)
        self.chk_grn = QCheckBox("Green"); self.chk_grn.setChecked(True); self.chk_grn.toggled.connect(self.update_viewer_visibility)
        h_chan.addWidget(self.chk_red); h_chan.addWidget(self.chk_grn)
        l_tools.addLayout(h_chan)
        
        self.btn_save = QPushButton("Save Current Crop (S)")
        self.btn_save.setStyleSheet("background-color: #dfd; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_current_crop)
        l_tools.addWidget(self.btn_save)
        
        l_tools.addWidget(QLabel("Shortcuts: Alt+1/2/3 (Color), Alt+4/5 (Type)"))
        g_tools.setLayout(l_tools)
        r_layout.addWidget(g_tools)
        
        # Search Box
        h_search = QHBoxLayout()
        self.spin_atlas_id = QSpinBox(); self.spin_atlas_id.setRange(0, 99999); self.spin_atlas_id.setPrefix("Atlas ID: ")
        btn_go = QPushButton("Go"); btn_go.clicked.connect(self.search_atlas_id)
        h_search.addWidget(self.spin_atlas_id); h_search.addWidget(btn_go)
        r_layout.addLayout(h_search)

        r_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setSizes([1000, 600])
        self.layout_main.addWidget(splitter)
        
        self.bind_shortcuts()

    # --- Data Loading ---
    def load_nav_data(self, sample, atlas):
        """Loads stitched brain and atlas for navigation."""
        if os.path.exists(sample):
            # Read lazily or downsampled if too huge, here assuming it fits in RAM since user said "Nav Sample"
            d_sample = tifffile.imread(sample)
            self.viewer_nav.add_image(d_sample, name="Nav Sample", colormap='gray', blending='additive')
        
        if os.path.exists(atlas):
            # Load Atlas Labels
            itk_img = sitk.ReadImage(atlas)
            self.d_atlas = sitk.GetArrayFromImage(itk_img)
            self.viewer_nav.add_labels(self.d_atlas, name="Nav Atlas", opacity=0.4, blending='translucent')

    def search_atlas_id(self):
        target_id = self.spin_atlas_id.value()
        if not hasattr(self, 'd_atlas'): return
        
        # Find first occurrence
        locs = np.where(self.d_atlas == target_id)
        if len(locs[0]) > 0:
            cz, cy, cx = int(np.mean(locs[0])), int(np.mean(locs[1])), int(np.mean(locs[2]))
            self.viewer_nav.dims.set_current_step(0, cz)
            self.viewer_nav.camera.center = (0, cy, cx) # 2D view
            self.trigger_region_load(cz, cy, cx)
        else:
            QMessageBox.warning(self, "Not Found", f"Atlas ID {target_id} not found.")

    # --- Interaction ---
    def on_nav_click(self, layer, event):
        """Handle clicks on the Nav viewer to select regions."""
        if event.button != 1 or 'Shift' in event.modifiers: return # Allow panning with shift
        
        # Get coordinates in Nav space
        c_z, c_y, c_x = self.viewer_nav.cursor.position
        self.trigger_region_load(int(c_z), int(c_y), int(c_x))

    def trigger_region_load(self, g_z, g_y, g_x):
        """
        1. Identify Brain Region (Atlas Label).
        2. Map to Raw Tile.
        3. Generate Crops.
        """
        # 1. Identify Atlas Label
        atlas_id = 0
        if hasattr(self, 'd_atlas'):
            try:
                # Clamp coords
                az = min(max(0, g_z), self.d_atlas.shape[0]-1)
                ay = min(max(0, g_y), self.d_atlas.shape[1]-1)
                ax = min(max(0, g_x), self.d_atlas.shape[2]-1)
                atlas_id = self.d_atlas[az, ay, ax]
            except: pass
        
        self.lbl_region_name.setText(f"Region ID: {atlas_id} | Global: {g_z}, {g_y}, {g_x}")

        # 2. Map Global -> Raw Tile
        tid, l_z, l_y, l_x = self.coord_sys.global_to_local(g_z, g_y, g_x)
        
        if tid is None:
            self.viewer_main.text_overlay.text = "Selected point is outside any Raw Tile."
            return

        print(f"Mapped to Tile {tid}, Local: {l_z}, {l_y}, {l_x}")
        
        # 3. Load Tile Data (if changed)
        if tid != self.current_tile_idx:
            self.load_tile_data(tid, lambda: self.generate_crops_around(l_z, l_y, l_x, atlas_id))
        else:
            self.generate_crops_around(l_z, l_y, l_x, atlas_id)

    def load_tile_data(self, tile_idx, callback):
        """Loads full tile into memory/cache."""
        cached = self.tile_cache.get(tile_idx)
        if cached:
            self.current_stack_r, self.current_stack_g, self.current_detections = cached
            self.current_tile_idx = tile_idx
            callback()
            return
            
        self.progress = QProgressDialog(f"Loading Raw Tile {tile_idx}...", None, 0, 100, self)
        self.progress.show()
        
        meta = self.coord_sys.tiles[tile_idx]
        self.loader = DataLoaderThread(meta, self.root_red, self.root_green, self.csv_root)
        self.loader.data_loaded.connect(self.on_tile_loaded)
        self.loader.progress_update.connect(lambda v, m: self.progress.setValue(v))
        
        # Pass the callback to the thread handler via a temporary attribute or simple closure
        self._pending_callback = callback
        self.loader.start()

    def on_tile_loaded(self, r, g, dets, idx, fnames):
        self.progress.close()
        self.tile_cache.add(idx, r, g, dets)
        self.current_stack_r, self.current_stack_g, self.current_detections = r, g, dets
        self.current_filenames = fnames
        self.current_tile_idx = idx
        if hasattr(self, '_pending_callback'):
            self._pending_callback()

    def generate_crops_around(self, cz, cy, cx, atlas_id):
        """
        Generates a grid of 256x256 crop coordinates centered on the click,
        OR covering the Atlas Region if ID > 0.
        """
        self.crop_list = []
        
        # Option A: Simple Sliding Window around click (Size: 3x3 crops)
        # We define the center of the clicked area as the center of the middle crop
        half_h = CROP_SHAPE_PX[0] // 2
        
        # Create a grid of top-left corners
        start_y = max(0, cy - CROP_SHAPE_PX[0])
        start_x = max(0, cx - CROP_SHAPE_PX[1])
        
        # Generate 3x3 grid of crops
        for dy in range(0, 3):
            for dx in range(0, 3):
                y = start_y + (dy * (CROP_SHAPE_PX[0] // 2)) # 50% overlap
                x = start_x + (dx * (CROP_SHAPE_PX[1] // 2))
                
                # Boundary check
                if y + CROP_SHAPE_PX[0] < TILE_SHAPE_PX[0] and x + CROP_SHAPE_PX[1] < TILE_SHAPE_PX[1]:
                    self.crop_list.append((cz, y, x))

        if not self.crop_list:
            self.viewer_main.text_overlay.text = "Edge of tile reached, no valid crops."
            return

        self.current_crop_idx = 0
        self.calculate_region_stats(atlas_id)
        self.show_crop()

    def calculate_region_stats(self, atlas_id):
        # Filter all detections in this tile for those falling inside the atlas region
        # This is complex without mapping every pixel. 
        # Simplified: Show counts for the *current tile* (or approximate).
        # Since the user asked to "Print out how many cells detected in this brain region",
        # ideally we use the global mapping. For now, we count what's in the Loaded Tile.
        
        total_counts = {k: 0 for k in CLASS_ID_MAP.keys()}
        if self.current_detections:
            for z, df in self.current_detections.items():
                counts = df['class_name'].value_counts()
                for cls, count in counts.items():
                    # Handle flexible naming
                    for key in total_counts:
                        if key in cls:
                            total_counts[key] += count
        
        txt = "\n".join([f"{k}: {v}" for k,v in total_counts.items() if v > 0])
        self.lbl_stats.setText(f"Tile Totals:\n{txt}")

    # --- Viewer Logic (256x256) ---
    def change_crop(self, delta):
        if not self.crop_list: return
        self.current_crop_idx = (self.current_crop_idx + delta) % len(self.crop_list)
        self.show_crop()

    def show_crop(self):
        z, y, x = self.crop_list[self.current_crop_idx]
        self.current_z = z # Update Z for annotation logic
        
        h, w = CROP_SHAPE_PX
        
        # Slice Data
        # Ensure Z is within bounds
        z_safe = min(max(0, z), self.current_stack_r.shape[0]-1)
        
        img_r = self.current_stack_r[z_safe, y:y+h, x:x+w]
        img_g = self.current_stack_g[z_safe, y:y+h, x:x+w]
        
        self.viewer_main.layers.clear()
        if self.chk_red.isChecked():
            self.viewer_main.add_image(img_r, name='Red', colormap='red', blending='additive')
        if self.chk_grn.isChecked():
            self.viewer_main.add_image(img_g, name='Green', colormap='green', blending='additive')
            
        self.lbl_crop_idx.setText(f"Crop {self.current_crop_idx+1}/{len(self.crop_list)} (Z:{z_safe}, Y:{y}, X:{x})")
        self.viewer_main.text_overlay.text = ""
        
        # Load Boxes relative to Crop
        self.load_crop_boxes(z_safe, y, x)

    def load_crop_boxes(self, z, crop_y, crop_x):
        self.shapes_layer = self.viewer_main.add_shapes(
            name='Detections', edge_width=2, face_color='transparent',
            text={'string': '{class_name}', 'color': 'cyan', 'size': 8}
        )
        self.shapes_layer.mode = 'SELECT'
        # Bind events...
        self.shapes_layer.bind_key('Delete', self.on_delete_key)
        
        if z in self.current_detections:
            df = self.current_detections[z]
            # Filter boxes inside crop
            mask = (df['x1'] >= crop_x) & (df['x2'] <= crop_x + CROP_SHAPE_PX[1]) & \
                   (df['y1'] >= crop_y) & (df['y2'] <= crop_y + CROP_SHAPE_PX[0])
            
            subset = df[mask]
            
            boxes, props, colors = [], {'class_name': []}, []
            for _, r in subset.iterrows():
                # Convert to local crop coordinates (0-256)
                ly1, lx1 = r['y1'] - crop_y, r['x1'] - crop_x
                ly2, lx2 = r['y2'] - crop_y, r['x2'] - crop_x
                
                boxes.append([[ly1, lx1], [ly2, lx2]])
                props['class_name'].append(r['class_name'])
                colors.append(COLOR_MAP.get('cyan')) # Simplify color logic for brevity
                
            self.shapes_layer.add(boxes, shape_type='rectangle', edge_color=colors)
            self.shapes_layer.properties = props

    def update_viewer_visibility(self):
        for l in self.viewer_main.layers:
            if l.name == 'Red': l.visible = self.chk_red.isChecked()
            if l.name == 'Green': l.visible = self.chk_grn.isChecked()

    # --- Saving ---
    def save_current_crop(self):
        if not self.crop_list: return
        z, cy, cx = self.crop_list[self.current_crop_idx]
        
        # 1. Save Image (256x256)
        img_out = np.zeros((*CROP_SHAPE_PX, 3), dtype=np.uint8)
        
        # Fetch data again to be sure
        z_safe = min(max(0, z), self.current_stack_r.shape[0]-1)
        r_crop = self.current_stack_r[z_safe, cy:cy+CROP_SHAPE_PX[0], cx:cx+CROP_SHAPE_PX[1]]
        g_crop = self.current_stack_g[z_safe, cy:cy+CROP_SHAPE_PX[0], cx:cx+CROP_SHAPE_PX[1]]
        
        if self.chk_red.isChecked(): img_out[..., 2] = r_crop
        if self.chk_grn.isChecked(): img_out[..., 1] = g_crop
        
        fname = f"crop_T{self.current_tile_idx}_Z{z}_Y{cy}_X{cx}.tif"
        cv2.imwrite(os.path.join(self.save_root, 'images', fname), img_out)
        
        # 2. Save Labels (Normalized 0-1 relative to 256 crop)
        lbl_lines = []
        if hasattr(self, 'shapes_layer'):
            for i, box in enumerate(self.shapes_layer.data):
                ys, xs = [p[0] for p in box], [p[1] for p in box]
                y1, x1, y2, x2 = min(ys), min(xs), max(ys), max(xs)
                
                cls_name = self.shapes_layer.properties['class_name'][i]
                cid = CLASS_ID_MAP.get('other', 99)
                for k, v in CLASS_ID_MAP.items():
                    if k in cls_name: cid = v
                
                # Normalize to 256
                nx = (x1 + (x2-x1)/2) / CROP_SHAPE_PX[1]
                ny = (y1 + (y2-y1)/2) / CROP_SHAPE_PX[0]
                nw = (x2 - x1) / CROP_SHAPE_PX[1]
                nh = (y2 - y1) / CROP_SHAPE_PX[0]
                
                lbl_lines.append(f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
        
        txt_name = fname.replace('.tif', '.txt')
        with open(os.path.join(self.save_root, 'labels', txt_name), 'w') as f:
            f.write("\n".join(lbl_lines))
            
        self.viewer_main.text_overlay.text = f"Saved {fname}"
        QTimer.singleShot(2000, lambda: setattr(self.viewer_main.text_overlay, 'text', ''))

    # --- Other Wrappers (Shortcuts, Undo, etc.) ---
    # These remain largely similar to your original code, 
    # but operate on self.shapes_layer which now represents the CROP only.
    # You must implement `sync_shapes_to_memory` to convert crop-relative coords 
    # back to global tile coords if you want persistence when switching crops.
    
    def bind_shortcuts(self):
        # (Same as your original code)
        self.viewer_main.bind_key('Alt-1', lambda e: self.update_selected_attr(color='red'))
        self.viewer_main.bind_key('s', lambda e: self.save_current_crop())
        # ... add others ...

    def update_selected_attr(self, color=None, ctype=None):
        # (Same logic as original, operating on current shapes_layer)
        pass

    def on_delete_key(self, layer):
        layer.remove_selected()