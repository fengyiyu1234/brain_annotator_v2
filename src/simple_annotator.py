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
import time
import random
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSplitter, QProgressDialog, 
                             QApplication, QGroupBox, QShortcut, QSlider)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from .config import PATCH_SIZE, COLOR_FACTOR, TYPE_FACTOR, CLASS_ID_MAP, NAPARI_COLOR_MAP, TILE_SHAPE_PX, TYPE_SHORTCUTS, COLOR_SHORTCUTS, PRELOAD_GLOBAL_DATA
# 引入工具函数
from .utils import export_yolo_patch 
# 引入我们刚刚剥离的线程类
from .data_loader import RegistrationLoaderThread, PatchLoaderThread, GlobalPreloadThread
from .widgets import OntologyTreeWidget

# Main GUI
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
        self.global_image_cache = {}
        
        self.region_cache = {} 
        self.current_region_name = ""
        self.init_ui()
        QTimer.singleShot(100, self.start_indexing)
        
        # 实时刷新选中框信息的定时器 (100ms)
        self.sel_timer = QTimer(self)
        self.sel_timer.timeout.connect(self.update_info_panel)
        self.sel_timer.start(100)
        
        def _setup_probe_event(self):
            @self.viewer.mouse_drag_callbacks.append
            def probe_pixel(viewer, event):
                if 'Shift' in event.modifiers:
                    # 必须确保当前有图像加载
                    if not hasattr(self, 'current_img_stack') or self.current_img_stack is None:
                        return
                    pos = event.position
                    z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
                    stack = self.current_img_stack
                    if 0 <= z < stack.shape[0] and 0 <= y < stack.shape[1] and 0 <= x < stack.shape[2]:
                        val_r = stack[z, y, x, 0]
                        val_g = stack[z, y, x, 1]
                        self.lbl_pixel_val.setText(
                            f"<b>Pixel Probe:<br>"
                            f"Z:{z} Y:{y} X:{x}<br>"
                            f"<span style='color:red;'>R: {val_r}</span><br>"
                            f"<span style='color:green;'>G: {val_g}</span>"
                        )
                yield
            # 执行一次绑定
            self._setup_probe_event()

    def init_ui(self):
        self.setWindowTitle("Brain Annotator - 3D Anchor Cropping Mode")
        self.resize(1600, 950)
        main_layout = QHBoxLayout(self)
        
        # --- Left Panel ---
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
        
        # --- Middle Panel (Viewer & Sliders) ---
        mid_panel = QWidget()
        m_lay = QVBoxLayout(mid_panel)
        self.viewer = ViewerModel()
        self.qt_viewer = QtViewer(self.viewer)
        # 强制显示 2D 切面，Napari 遇到 3D 数据会自动把维度 0（Z轴）变成滑块
        self.viewer.dims.ndisplay = 2
        m_lay.addWidget(self.qt_viewer)
        
        # Z 轴层级滑块 (Z-1, Z, Z+1)
        z_lay = QHBoxLayout()
        z_lay.addWidget(QLabel("<b>Z-Slice (Z-1 / Z / Z+1):</b>"))
        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(0, 2)
        self.slider_z.setTickPosition(QSlider.TicksBelow)
        self.slider_z.setTickInterval(1)
        self.slider_z.valueChanged.connect(self.on_z_slider_changed)
        z_lay.addWidget(self.slider_z)
        m_lay.addLayout(z_lay)
        
        # 同步 Napari 自带的维度变化
        self.viewer.dims.events.current_step.connect(self.sync_z_slider_from_napari)

        # Patch 跳转滑块与按钮
        bot_bar = QHBoxLayout()
        bot_style = "font-size: 15px; font-weight: bold; padding: 5px;"

        self.btn_prev = QPushButton("<< Prev (A)")
        self.btn_prev.setStyleSheet(bot_style)
        self.btn_prev.clicked.connect(lambda: self.change_patch(-1))
        
        self.slider_patch = QSlider(Qt.Horizontal)
        self.slider_patch.valueChanged.connect(self.jump_to_patch)
        
        self.lbl_counter = QLabel("0 / 0")
        self.lbl_counter.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.lbl_counter.setAlignment(Qt.AlignCenter)

        self.btn_next = QPushButton("Next (D) >>")
        self.btn_next.setStyleSheet(bot_style)
        self.btn_next.clicked.connect(lambda: self.change_patch(1))

        self.btn_save = QPushButton("Save Patch (S)")
        self.btn_save.setStyleSheet(bot_style)
        self.btn_save.clicked.connect(self.save_current_patch)
        
        bot_bar.addWidget(self.btn_prev)
        bot_bar.addWidget(self.slider_patch)
        bot_bar.addWidget(self.lbl_counter)
        bot_bar.addWidget(self.btn_next)
        bot_bar.addWidget(self.btn_save)
        m_lay.addLayout(bot_bar)
        
        # --- Right Panel ---
        right_panel = QWidget()
        right_panel.setFixedWidth(380)

        right_panel.setStyleSheet("""
            QLabel { font-size: 15px; }
            QGroupBox { font-size: 15px; font-weight: bold; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
        """)

        r_lay = QVBoxLayout(right_panel)
        
        # =========================================================
        # 1. Selected Cell Info & Pixel Probe (选中细胞信息 & 像素探针)
        # =========================================================
        info_grp = QGroupBox("Selected Cell Info")
        i_lay = QVBoxLayout()
        self.lbl_info_file = QLabel("Raw File: --")
        self.lbl_info_class = QLabel("Class: --")
        self.lbl_info_color = QLabel("Color: --")
        self.lbl_info_conf = QLabel("Conf: --")
        
        # 【新增】：用于显示探针点击的像素强度
        self.lbl_pixel_val = QLabel("<b>Pixel Probe:</b> (Hold <b>Shift + Click</b> on image)")
        
        # 加粗显示更清晰
        self.lbl_info_file.setWordWrap(True)
        self.lbl_info_class.setStyleSheet("font-weight: bold;")
        self.lbl_info_color.setStyleSheet("font-weight: bold;")
        
        i_lay.addWidget(self.lbl_info_file)
        i_lay.addWidget(self.lbl_info_class)
        i_lay.addWidget(self.lbl_info_color)
        i_lay.addWidget(self.lbl_info_conf)
        i_lay.addWidget(self.lbl_pixel_val) # 加入探针面板
        
        info_grp.setLayout(i_lay)
        r_lay.addWidget(info_grp)
        
        # =========================================================
        # 2. Shortcuts (快捷键说明)
        # =========================================================
        short_grp = QGroupBox("Shortcuts")
        s_lay = QVBoxLayout()

        self.lbl_current_mode = QLabel("<b>Current Mode:</b> <span style='color:green;'>Pan / Zoom (Drag)</span>")
        s_lay.addWidget(self.lbl_current_mode)
        color_txt = ", ".join([f"{i+1}={v['name']}" for i, v in enumerate(COLOR_FACTOR.values())])
        type_txt = "<br>".join([f"{TYPE_SHORTCUTS[i]}: {k}" for i, k in enumerate(TYPE_FACTOR.keys())])
        mode_txt = "V: Toggle Select / Pan Mode"
        s_text = f"<b>Tools:</b><br>{mode_txt}<hr><b>Color:</b><br>{color_txt}<hr><b>Type:</b><br>{type_txt}"
        
        lbl_shortcuts = QLabel(s_text)
        lbl_shortcuts.setWordWrap(True)
        s_lay.addWidget(lbl_shortcuts)
        
        short_grp.setLayout(s_lay)
        r_lay.addWidget(short_grp)

        # =========================================================
        # 3. Image Adjustments (图像滑块调节)
        # =========================================================
        adj_grp = QGroupBox("Image Adjustments")
        a_lay = QVBoxLayout()
        
        def make_slider(name, min_val, max_val, default_val, callback):
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(130) # 调宽一点以容纳 Min/Max 数值
            sl = QSlider(Qt.Horizontal)
            sl.setRange(min_val, max_val)
            sl.setValue(default_val)
            sl.valueChanged.connect(callback)
            row.addWidget(lbl)
            row.addWidget(sl)
            return sl, lbl, row

        # 获取滑块、标签和布局
        self.sl_red_c, self.lbl_red_c, r1 = make_slider("R Cont", 100, 65535, 15000, self.update_image_display)
        self.sl_red_l, self.lbl_red_l, r2 = make_slider("R Light", 10, 300, 100, self.update_image_display)
        self.sl_grn_c, self.lbl_grn_c, r3 = make_slider("G Cont", 100, 65535, 15000, self.update_image_display)
        self.sl_grn_l, self.lbl_grn_l, r4 = make_slider("G Light", 10, 300, 100, self.update_image_display)

        a_lay.addLayout(r1); a_lay.addLayout(r2)
        a_lay.addLayout(r3); a_lay.addLayout(r4)
        
        # 界面上提示通道开关快捷键
        hint_lbl = QLabel("<span style='color:gray;'><small>Shortcut: Press <b>Z</b> (Toggle Red), <b>X</b> (Toggle Green)</small></span>")
        a_lay.addWidget(hint_lbl)
        
        adj_grp.setLayout(a_lay)
        r_lay.addWidget(adj_grp)

        # =========================================================
        # 4. 结尾收尾工作
        # =========================================================
        r_lay.addStretch() # 把所有框顶到上面

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(mid_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 1050, 380])
        main_layout.addWidget(splitter)
        
        self.bind_keys()
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_action)
        
        # 绑定 Z 和 X 键用于快速开关通道 (在这里一并加上)
        QShortcut(QKeySequence('Z'), self).activated.connect(lambda: self.toggle_channel('Red'))
        QShortcut(QKeySequence('X'), self).activated.connect(lambda: self.toggle_channel('Green'))


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
        QShortcut(QKeySequence("V"), self).activated.connect(self.toggle_tool_mode)

    def toggle_tool_mode(self):
        """单键切换：在 选中 (select) 和 拖拽 (pan_zoom) 之间来回切换，并更新 UI"""
        if hasattr(self, 'shapes_layer'):
            current_mode = self.shapes_layer.mode
            
            if current_mode == 'select':
                # 切回拖拽模式
                self.shapes_layer.mode = 'pan_zoom'
                self.lbl_current_mode.setText("<b>Current Mode:</b> <span style='color:green;'>Pan / Zoom (Drag)</span>")
            else:
                # 切换到选中模式
                self.shapes_layer.mode = 'select'
                self.lbl_current_mode.setText("<b>Current Mode:</b> <span style='color:blue;'>Select Box (Click)</span>")

    def start_indexing(self):
        self.progress.setLabelText("Loading Registration Anchors...")
        self.progress.show()
        self.loader = RegistrationLoaderThread(self.reg_dir)
        self.loader.progress_signal.connect(self.progress.setValue)
        self.loader.finished_signal.connect(self.indexing_done)
        self.loader.start()
        self.start_global_preload()

    def indexing_done(self, anchor_map):
        self.progress.close()
        self.anchor_map = anchor_map
        total = sum(len(v) for v in anchor_map.values())
        self.lbl_status.setText(f"Loaded {total} Anchors.")
        self.start_global_preload()

    def start_global_preload(self):
        """如果开启了预加载开关，就在开局执行一次大加载"""
        if not PRELOAD_GLOBAL_DATA:
            return
            
        self.progress.setLabelText("Global Preloading (Loading all images to RAM)...")
        self.progress.setMinimumDuration(0)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.resize(550, 200) 
        self.progress.setValue(0)
        self.progress.show()
        
        self.preloader = GlobalPreloadThread(
            self.coord_sys.tiles, 
            self.root_red, 
            self.root_green, 
            self.det_dir
        )
        
        self.preloader.progress_signal.connect(self._update_preload_ui)
        self.preloader.finished_signal.connect(self._on_preload_finished)
        self.preloader.start()

    def _update_preload_ui(self, val, text):
        self.progress.setValue(val)
        self.progress.setLabelText(text) # 这里会接收到我们拼接好的详尽信息

    def _on_preload_finished(self, cache_dict):
        """预加载线程结束后的回调"""
        self.global_image_cache = cache_dict
        self.progress.close()
        self.lbl_status.setText("All data preloaded to Memory! You can now click regions instantly.")
        print(f"[SUCCESS] Global preload finished! Cached {len(cache_dict)} tiles.")

    def on_tree_click(self, item, col):
        nid = item.data(0, Qt.UserRole)
        region_name = item.text(0)
        self.current_region_name = region_name
        
        print(f"\n{'='*50}")
        print(f"[DEBUG] Clicked Region: '{region_name}'")
        print(f"[DEBUG] Current Cache Keys: {list(self.region_cache.keys())}")
        
        if region_name in self.region_cache:
            print(f"[DEBUG] 🟢 CACHE HIT! Loading '{region_name}' instantly from memory.")
            self.lbl_status.setText(f"Loaded '{region_name}' from Memory Cache.")
            self.patches_loaded(self.region_cache[region_name])
            return  

        print(f"[DEBUG] 🔴 CACHE MISS! Preparing to crop patches for '{region_name}'...")
        desc_ids = self.ontology.get_all_descendant_ids(nid)
        target_gos = [self.ontology.id_to_go[i] for i in desc_ids if i in self.ontology.id_to_go]
        
        anchors = []
        for go in target_gos:
            if go in self.anchor_map: anchors.extend(self.anchor_map[go])
                
        if not anchors:
            self.lbl_status.setText(f"No cells found for {region_name}.")
            return

        self.lbl_status.setText(f"Cropping {len(anchors)} patches...")
        self.progress.setLabelText("Cropping Tiles...")
        self.progress.setValue(0)
        self.progress.show()
        
        self.patch_loader = PatchLoaderThread(anchors, self.coord_sys, self.root_red, self.root_green,self.det_dir, global_cache=self.global_image_cache)
        self.patch_loader.progress_signal.connect(self.progress.setValue)
        self.patch_loader.finished_signal.connect(self.patches_loaded)
        self.patch_loader.start()

    def patches_loaded(self, patches):
        self.progress.close()
        if not patches: return
        
        # ------------------ 写入缓存逻辑 ------------------
        if self.current_region_name and self.current_region_name not in self.region_cache:
            print(f"[DEBUG] 💾 SAVING '{self.current_region_name}' TO CACHE. Total patches: {len(patches)}")
            self.region_cache[self.current_region_name] = patches
            
        self.current_patches = patches
        self.current_idx = 0
        
        self.slider_patch.blockSignals(True)
        self.slider_patch.setRange(0, len(patches)-1)
        self.slider_patch.setValue(0)
        self.slider_patch.blockSignals(False)
        
        self.undo_stack = []
        self.lbl_status.setText(f"Loaded {len(patches)} Patches.")
        self.show_patch()

    def show_patch(self):
        t_start = time.time()
        
        if not self.current_patches: 
            return # 连 clear 都不用了，直接 return
            
        # ==========================================
        # 1. UI 重置
        # ==========================================
        for sl in [self.sl_red_c, self.sl_red_l, self.sl_grn_c, self.sl_grn_l]:
            sl.blockSignals(True)
        self.sl_red_c.setValue(65535)  
        self.sl_red_l.setValue(100)    
        self.sl_grn_c.setValue(65535)
        self.sl_grn_l.setValue(100)
        for sl in [self.sl_red_c, self.sl_red_l, self.sl_grn_c, self.sl_grn_l]:
            sl.blockSignals(False)
            
        # ==========================================
        # 2. 数据获取与极值计算
        # ==========================================
        data = self.current_patches[self.current_idx]
        self.current_img_stack = data['image_stack'] 
        
        r_data = self.current_img_stack[::4, ::4, ::4, 0] 
        g_data = self.current_img_stack[::4, ::4, ::4, 1]
        self.lbl_red_c.setText(
            f"R Cont<br><span style='color: gray; font-size: 12px;'>[{r_data.min()}~{r_data.max()}]</span>"
        )
        self.lbl_grn_c.setText(
            f"G Cont<br><span style='color: gray; font-size: 12px;'>[{g_data.min()}~{g_data.max()}]</span>"
        )
        
        # ==========================================
        # 3. 👑 核心黑魔法：原地更新图像 (In-place Update)
        # 完美保留你的通道独立控制，同时抹杀所有延迟！
        # ==========================================
        if 'Red' in self.viewer.layers and 'Green' in self.viewer.layers and 'Blue' in self.viewer.layers:
            # 极速通道：图层已存在，直接替换底层 numpy 数据！
            self.viewer.layers['Red'].data = self.current_img_stack[..., 0]
            self.viewer.layers['Green'].data = self.current_img_stack[..., 1]
            self.viewer.layers['Blue'].data = self.current_img_stack[..., 2]
        else:
            # 只有第一次打开软件时，才执行耗时的初始化渲染
            self.viewer.add_image(
                self.current_img_stack, 
                channel_axis=3, 
                name=['Red', 'Green', 'Blue'],
                colormap=['red', 'green', 'blue'],
                blending='additive' 
            )

        # ==========================================
        # 4. 解析与原地更新 Bounding Boxes
        # ==========================================
        shapes, classes, edges = [], [], []
        confs, raw_files = [], []
        
        for b in data['boxes']:
            z = b['z_idx']
            shapes.append([
                [z, b['y1'], b['x1']], [z, b['y1'], b['x2']], 
                [z, b['y2'], b['x2']], [z, b['y2'], b['x1']]  
            ])
            classes.append(b['cls'])
            edges.append(NAPARI_COLOR_MAP.get(b['cls'], 'white'))
            confs.append(b['conf'])
            raw_files.append(b['raw_file'])
            
        # 同样对 Shapes 图层进行原地更新
        if 'boxes' in self.viewer.layers:
            self.shapes_layer = self.viewer.layers['boxes']
            self.shapes_layer.events.data.disconnect(self.push_undo) # 先断开监听防止报错
            
            # 直接替换数据和属性
            self.shapes_layer.data = shapes
            self.shapes_layer.properties = {'class_id': classes, 'conf': confs, 'raw_file': raw_files}
            if edges:
                self.shapes_layer.edge_color = edges
                
            self.shapes_layer.events.data.connect(self.push_undo)
            self.shapes_layer.mode = 'select'
        else:
            # 第一次初始化 Shapes 图层（就算没有框也要塞个空列表进去占位）
            self.shapes_layer = self.viewer.add_shapes(
                shapes if shapes else [], 
                shape_type='rectangle', edge_width=2, 
                edge_color=edges if edges else 'white',
                face_color='transparent', name='boxes', ndim=3,
                properties={'class_id': classes, 'conf': confs, 'raw_file': raw_files}
            )
            self.shapes_layer.events.data.connect(self.push_undo)
            self.shapes_layer.mode = 'select'

        if hasattr(self, 'lbl_current_mode'):
            self.lbl_current_mode.setText("<b>Current Mode:</b> <span style='color:blue;'>Select Box (Click)</span>")
            
        # ==========================================
        # 5. 尾部 UI 更新
        # ==========================================
        self.slider_z.setValue(1) 
        self.lbl_counter.setText(f"{self.current_idx + 1} / {len(self.current_patches)}")
        self.viewer.text_overlay.text = data.get('name', 'Unknown')
        if hasattr(self, 'update_info_panel'):
            self.update_info_panel()

    def sync_canvas_to_memory(self):
        """将 Napari 画布上最新的框状态（删除、拖拽修改后），反向覆盖回内存中"""
        # 如果根本没有加载数据，直接跳过
        if not hasattr(self, 'current_patches') or not self.current_patches:
            return
            
        # 如果当前图层被清空了（比如你把框全删了）
        if not hasattr(self, 'shapes_layer') or self.shapes_layer not in self.viewer.layers:
            self.current_patches[self.current_idx]['boxes'] = []
            return

        new_boxes = []
        props = self.shapes_layer.properties
        
        # 遍历画布上现存的所有框
        for i, box in enumerate(self.shapes_layer.data):
            # 提取 3D 坐标 (z, y, x)
            z = int(box[0][0])
            ys, xs = box[:, 1], box[:, 2]
            
            # 重新计算左上角和右下角（即使你用鼠标拖动了框，这里也会抓到最新坐标）
            y1, y2 = float(min(ys)), float(max(ys))
            x1, x2 = float(min(xs)), float(max(xs))
            
            # 从 properties 属性表中把分类、置信度等对应信息拿回来
            cls_id = int(props['class_id'][i]) if 'class_id' in props else 0
            conf = float(props['conf'][i]) if 'conf' in props else 1.0
            raw_file = str(props['raw_file'][i]) if 'raw_file' in props else ""
            
            # 组装成你原始的代码需要的字典格式
            new_boxes.append({
                'z_idx': z,
                'y1': y1, 'x1': x1,
                'y2': y2, 'x2': x2,
                'cls': cls_id,
                'conf': conf,
                'raw_file': raw_file
            })
            
        # 霸道覆盖：用画面上的最新状态替换掉内存里的旧状态
        self.current_patches[self.current_idx]['boxes'] = new_boxes

    def on_z_slider_changed(self, value):
        if self.viewer.dims.ndim >= 3:
            self.viewer.dims.set_current_step(0, value)

    def sync_z_slider_from_napari(self, event):
        if self.viewer.dims.ndim >= 3:
            step = self.viewer.dims.current_step[0]
            self.slider_z.blockSignals(True)
            self.slider_z.setValue(step)
            self.slider_z.blockSignals(False)

    def update_info_panel(self):
        """定时器调用：实时读取当前选中的框，并更新右侧面板文字"""
        if not hasattr(self, 'shapes_layer'): return
        
        selected = list(self.shapes_layer.selected_data)
        if len(selected) == 1:
            idx = selected[0]
            props = self.shapes_layer.properties
            c_id = int(props['class_id'][idx])
            c_name = CLASS_ID_MAP.get(c_id, {}).get('name', 'Unknown')
            c_color = CLASS_ID_MAP.get(c_id, {}).get('color', 'white')
            conf = props['conf'][idx]
            raw_f = props['raw_file'][idx]
            
            self.lbl_info_file.setText(f"Raw File: <span style='color:blue;'>{raw_f}</span>")
            self.lbl_info_class.setText(f"Class: {c_name}")
            self.lbl_info_color.setText(f"Color: {c_color.capitalize()}")
            self.lbl_info_conf.setText(f"Conf: {conf:.3f}")
        else:
            self.lbl_info_file.setText(f"Raw File: --")
            self.lbl_info_class.setText(f"Class: --")
            self.lbl_info_color.setText(f"Color: --")
            self.lbl_info_conf.setText(f"Conf: --")

    def update_image_display(self):
        """实时读取右侧滑块数值，并应用到 Napari 的对应通道层"""
        if not hasattr(self, 'viewer'): return
        
        r_max = self.sl_red_c.value()
        r_gamma = self.sl_red_l.value() / 100.0
        g_max = self.sl_grn_c.value()
        g_gamma = self.sl_grn_l.value() / 100.0

        for layer in self.viewer.layers:
            if type(layer).__name__ == 'Image':
                if layer.name == 'Red':
                    layer.contrast_limits = (0, r_max)
                    layer.gamma = r_gamma
                elif layer.name == 'Green':
                    layer.contrast_limits = (0, g_max)
                    layer.gamma = g_gamma

    def toggle_channel(self, ch_name):
        """按键切换对应通道的显示/隐藏"""
        if not hasattr(self, 'viewer'): return
        for layer in self.viewer.layers:
            if layer.name == ch_name:
                layer.visible = not layer.visible

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
            
            from napari.utils.colormaps.standardize_color import transform_color
            edge_c[idx] = transform_color(cfg['color'])[0]
            
            self.current_patches[self.current_idx]['boxes'][idx]['cls'] = new_id
            
        self.shapes_layer.properties = props
        self.shapes_layer.edge_color = edge_c
        self.shapes_layer.refresh()
        self.update_info_panel()

    def push_undo(self, event):
        if not self.current_patches: return
        data = self.current_patches[self.current_idx]
        self.undo_stack.append(copy.deepcopy(data['boxes']))
        if len(self.undo_stack) > 10: self.undo_stack.pop(0)

    def undo_action(self):
        if not self.undo_stack: return
        
        restored_boxes = self.undo_stack.pop()
        self.current_patches[self.current_idx]['boxes'] = restored_boxes
        
        if hasattr(self, 'shapes_layer') and self.shapes_layer in self.viewer.layers:
            # 移除前先解绑信号，防止触发多余的 undo 记录
            self.shapes_layer.events.data.disconnect(self.push_undo)
            self.viewer.layers.remove(self.shapes_layer)
            
        shapes, classes, edges = [], [], []
        confs, raw_files = [], []
        
        for b in restored_boxes:
            z = b['z_idx']
            shapes.append([
                [z, b['y1'], b['x1']], # 左上
                [z, b['y1'], b['x2']], # 右上
                [z, b['y2'], b['x2']], # 右下
                [z, b['y2'], b['x1']]  # 左下
            ])
            classes.append(b['cls'])
            edges.append(NAPARI_COLOR_MAP.get(b['cls'], 'white'))
            confs.append(b['conf'])
            raw_files.append(b['raw_file'])
            
        if shapes:
            self.shapes_layer = self.viewer.add_shapes(
                shapes, shape_type='rectangle', edge_width=2, edge_color=edges,
                face_color='transparent', name='boxes', ndim=3,
                properties={'class_id': classes, 'conf': confs, 'raw_file': raw_files}
            )
            # 重新绑定监听，并保持在选中模式
            self.shapes_layer.events.data.connect(self.push_undo)
            self.shapes_layer.mode = 'select'
            if hasattr(self, 'lbl_current_mode'):
                self.lbl_current_mode.setText("<b>Current Mode:</b> <span style='color:blue;'>Select Box (Click)</span>")

    def change_patch(self, delta):
        if not self.current_patches: return
        self.sync_canvas_to_memory()
        new_idx = (self.current_idx + delta) % len(self.current_patches)
        # 通过触发滑块改变来跳转，保证 UI 统一
        self.slider_patch.setValue(new_idx)

    def jump_to_patch(self, idx):
        if not self.current_patches: return
        self.sync_canvas_to_memory()
        self.current_idx = idx
        self.undo_stack = []
        self.show_patch()

    def save_current_patch(self):
        if not self.current_patches: return
        self.sync_canvas_to_memory()
        
        data = self.current_patches[self.current_idx]
        base_name = data.get('name', f"patch_{self.current_idx}")

        s_data = self.shapes_layer.data if hasattr(self, 'shapes_layer') else None
        s_props = self.shapes_layer.properties if hasattr(self, 'shapes_layer') else None
        
        success = export_yolo_patch(
            image_stack=data['image_stack'],
            shapes_layer_data=s_data,
            properties=s_props,
            patch_size=PATCH_SIZE,
            save_root=self.save_root,
            base_name=base_name
        )
        
        if success:
            self.viewer.text_overlay.text = f"Saved Central Slice for {base_name}!"