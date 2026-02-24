# src/data_loader.py
import os
import glob
import pandas as pd
import tifffile
import cv2
import numpy as np
import concurrent.futures
from PyQt5.QtCore import QThread, pyqtSignal
from collections import OrderedDict
import random

# 引入配置和工具
from .config import PATCH_SIZE, TILE_SHAPE_PX, CELL_SAMPLE_RATIO
from .utils import natural_sort_key, parse_class_string

class TileCache:
    def __init__(self, max_gb):
        self.cache = OrderedDict()
        self.max_bytes = max_gb * 1024**3
        self.current_bytes = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def add(self, key, r_8, g_8, dets):
        size = r_8.nbytes + g_8.nbytes
        if size > self.max_bytes: return
        while self.current_bytes + size > self.max_bytes and self.cache:
            old_k, old_v = self.cache.popitem(last=False)
            self.current_bytes -= (old_v[0].nbytes + old_v[1].nbytes)
        self.cache[key] = (r_8, g_8, dets)
        self.current_bytes += size
    
    def info(self):
        return f"{len(self.cache)} Tiles | {self.current_bytes/1024**3:.1f} GB"
    
    def get_cached_keys(self):
        return set(self.cache.keys())

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


class PatchLoaderThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(list)
    
    def __init__(self, anchors, coord_sys, root_red, root_green, det_dir):
        super().__init__()
        self.anchors = anchors 
        self.tiles = coord_sys.tiles
        self.z_max = coord_sys.z_max
        self.root_red = root_red
        self.root_green = root_green
        self.det_dir = det_dir

    def _process_single_crop(self, pt, rgb_full_stack, dfs, raw_files, t_prefix, center_lz):
        """
        [多线程工作函数]：只负责内存切片和 DataFrame 过滤，脱离 GIL 锁！
        """
        lx, ly = pt['lx'], pt['ly']
        cx_pt, cy_pt = int(lx), int(ly)
        half = PATCH_SIZE // 2
        x1, y1 = cx_pt - half, cy_pt - half
        x2, y2 = cx_pt + half, cy_pt + half
        
        pad_l = max(0, -x1); pad_t = max(0, -y1)
        pad_r = max(0, x2 - TILE_SHAPE_PX[1]); pad_b = max(0, y2 - TILE_SHAPE_PX[0])
        sx1 = max(0, x1); sy1 = max(0, y1)
        sx2 = min(x2, TILE_SHAPE_PX[1]); sy2 = min(y2, TILE_SHAPE_PX[0])
        
        # 1. 图像切片与补齐
        crop_stack = []
        for rgb_f in rgb_full_stack:
            if rgb_f is None:
                crop_stack.append(np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint16))
            else:
                crop = rgb_f[sy1:sy2, sx1:sx2]
                if any([pad_l, pad_t, pad_r, pad_b]):
                    crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                crop_stack.append(crop)
        
        img_3d = np.stack(crop_stack)
        
        # 2. 细胞框筛选
        boxes = []
        for z_i, df_z in enumerate(dfs):
            if not df_z.empty:
                mask = (df_z['cx'] >= x1) & (df_z['cx'] <= x2) & (df_z['cy'] >= y1) & (df_z['cy'] <= y2)
                
                for _, row in df_z[mask].iterrows():
                    boxes.append({
                        'z_idx': z_i,  # 0, 1, or 2 (Z-1, Z, Z+1)
                        'cls': parse_class_string(row['cls']),
                        'conf': float(row.get('conf', 1.0)),
                        'raw_file': raw_files[z_i],
                        'x1': np.clip(row['x1'] - x1, 0, PATCH_SIZE), 
                        'y1': np.clip(row['y1'] - y1, 0, PATCH_SIZE),
                        'x2': np.clip(row['x2'] - x1, 0, PATCH_SIZE), 
                        'y2': np.clip(row['y2'] - y1, 0, PATCH_SIZE)
                    })
        
        return {
            'name': f"{t_prefix}_GZ{int(pt['gz'])}_LZ{center_lz}_GX{int(pt['gx'])}_GY{int(pt['gy'])}",
            'image_stack': img_3d,
            'boxes': boxes,
            'center_z': center_lz
        }

    def run(self):
        # ==========================================================
        # 1. 10% 随机抽样
        # ==========================================================
        total_anchors = len(self.anchors)
        if 0.0 < CELL_SAMPLE_RATIO < 1.0: 
            keep_count = max(1, int(total_anchors * CELL_SAMPLE_RATIO))
            self.anchors = random.sample(self.anchors, keep_count)
            print(f"random sampling: origin {total_anchors} anchors -> keeping {keep_count} ({CELL_SAMPLE_RATIO*100}%)")

        print(f"\n[CHECKPOINT] allocating {len(self.anchors)} anchors to Tiles...")
        
        # ==========================================================
        # 2. 分配锚点到特定的 Tile 和 Z 层
        # ==========================================================
        tasks_by_tile_z = {}
        for (gx, gy, gz) in self.anchors:
            target_tile = None
            for t in self.tiles:
                if t['abs_x'] <= gx < t['abs_x'] + t['w'] and \
                   t['abs_y'] <= gy < t['abs_y'] + t['h']:
                    target_tile = t
                    break
            
            if not target_tile: continue
            
            gz_0based = gz - 1
            z0 = self.z_max - target_tile['abs_z']
            local_z = gz_0based + z0
            
            if local_z < 0: continue
            
            lx = gx - target_tile['abs_x']
            ly = gy - target_tile['abs_y']
            key = (target_tile['dir'], local_z)
            
            if key not in tasks_by_tile_z:
                tasks_by_tile_z[key] = {'tile': target_tile, 'pts': []}
            tasks_by_tile_z[key]['pts'].append({'lx': lx, 'ly': ly, 'gx': gx, 'gy': gy, 'gz': gz})
            
        total_tasks = len(tasks_by_tile_z)
        patches = []
        det_cache = {} 
        processed = 0
        
        # 开启多线程池 (最多 32 核)
        workers = min(32, (os.cpu_count() or 4) + 4)
        print(f"[CHECKPOINT] 启动多线程并发，分配核心数: {workers}")

        # ==========================================================
        # 3. 遍历分组，执行硬盘 I/O 读取并派发多线程切片任务
        # ==========================================================
        for (t_dir, center_lz), data in tasks_by_tile_z.items():
            tile = data['tile']
            t_prefix = t_dir.replace('\\', '/').split('/')[-1] 
            
            # --- 读 CSV 缓存 (只读一次，且提前算好 cx, cy) ---
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
                        if not df.empty:
                            df['cx'] = (df['x1'] + df['x2']) / 2
                            df['cy'] = (df['y1'] + df['y2']) / 2
                        det_cache[t_prefix] = df
                    except: det_cache[t_prefix] = pd.DataFrame()
                else: det_cache[t_prefix] = pd.DataFrame()

            df_tile = det_cache[t_prefix]
            
            r_dir = os.path.join(self.root_red, t_dir)
            g_dir = os.path.join(self.root_green, t_dir)
            fr = sorted(os.listdir(r_dir), key=natural_sort_key) if os.path.exists(r_dir) else []
            fg = sorted(os.listdir(g_dir), key=natural_sort_key) if os.path.exists(g_dir) else []
            
            rgb_full_stack = []
            raw_files = []
            dfs = []
            
            z_indices = [center_lz - 1, center_lz, center_lz + 1]
            for z_idx, lz in enumerate(z_indices):
                if 0 <= lz < len(fr):
                    ir = tifffile.imread(os.path.join(r_dir, fr[lz]))
                    ig = tifffile.imread(os.path.join(g_dir, fg[lz])) if lz < len(fg) else np.zeros_like(ir)
                    rgb_full_stack.append(np.dstack((ir, ig, np.zeros_like(ir))))
                    raw_files.append(fr[lz])
                else:
                    rgb_full_stack.append(None)
                    raw_files.append("OUT_OF_BOUNDS")
                
                csv_z = lz + 1
                dfs.append(df_tile[df_tile['z'] == csv_z].copy() if not df_tile.empty else pd.DataFrame())

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        self._process_single_crop, pt, rgb_full_stack, dfs, raw_files, t_prefix, center_lz
                    ) for pt in data['pts']
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        patches.append(future.result())
                    except Exception as e:
                        print(f"[ERROR] Crop Failed: {e}")

            processed += 1
            self.progress_signal.emit(int(processed / total_tasks * 100))
            
        print(f"[CHECKPOINT] Patch Generation Complete. Yielded {len(patches)} image patches.")
        self.finished_signal.emit(patches)

# --- 数据加载线程 ---
class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(object, object, object, int, list) 
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)

    def __init__(self, meta, root_r, root_g, csv_root):
        super().__init__()
        self.meta, self.root_r, self.root_g, self.csv_root = meta, root_r, root_g, csv_root

    def read_stack_parallel(self, file_list, channel_name):
        count = len(file_list)
        if count == 0: return None
        try: first = tifffile.imread(file_list[0])
        except Exception as e: raise Exception(f"Read error {file_list[0]}: {e}")
        h, w = first.shape
        stack = np.zeros((count, h, w), dtype=first.dtype)
        
        def load_idx(i):
            try: stack[i] = tifffile.imread(file_list[i])
            except: pass
            
        workers = min(16, os.cpu_count() or 4) 
        chunk = max(1, count // 10)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(load_idx, i) for i in range(count)]
            for i, _ in enumerate(concurrent.futures.as_completed(futures)):
                if i % chunk == 0:
                    prog = int((i / count) * 45)
                    base = 0 if channel_name == "Red" else 50
                    self.progress_update.emit(base + prog, f"Loading {channel_name}: {i}/{count}")
        return stack

    def run(self):
        try:
            dir_rel = self.meta['dir']
            path_r = os.path.join(self.root_r, dir_rel)
            path_g = os.path.join(self.root_g, dir_rel)
            
            print(f"--- [Checkpoint 1] Start Loading Tile {self.meta['list_index']} ---")
            
            exts = ['*.tif', '*.tiff']
            files_r = sorted([f for e in exts for f in glob.glob(os.path.join(path_r, e))], key=natural_sort_key)
            files_g = sorted([f for e in exts for f in glob.glob(os.path.join(path_g, e))], key=natural_sort_key)
            
            if not files_r:
                self.error_occurred.emit(f"No images found in {path_r}")
                return
            
            min_len = min(len(files_r), len(files_g)) if files_g else len(files_r)
            files_r = files_r[:min_len]
            files_g = files_g[:min_len] if files_g else []
            
            stack_r = self.read_stack_parallel(files_r, "Red")
            stack_g = self.read_stack_parallel(files_g, "Green") if files_g else np.zeros_like(stack_r)
            
            self.progress_update.emit(95, "Parsing Detections...")
            detections = {}
            
            possible_names = [os.path.basename(dir_rel) + "_result.csv"]
            print(f"   > Targets: {possible_names}")

            csv_path = None
            for name in possible_names:
                p = os.path.join(self.csv_root, name)
                if os.path.exists(p):
                    csv_path = p
                    break
            
            if csv_path:
                try:
                    df = pd.read_csv(csv_path, header=None, 
                                     names=['filename', 'x1', 'y1', 'x2', 'y2', 'class_name', 'score', 'intensity', 'z_idx'])
                    df['z_idx'] = df['z_idx'] - 1 
                    
                    loaded_count = 0
                    for z, group in df.groupby('z_idx'):
                        if 0 <= z < min_len:
                            detections[z] = group.copy()
                            loaded_count += len(group)
                    print(f"--- [Checkpoint 5] Success. Loaded {loaded_count} boxes. ---")

                except Exception as e:
                    print(f"[ERROR] CSV Parse Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[WARNING] CSV Not Found.")

            self.data_loaded.emit(stack_r, stack_g, detections, self.meta['list_index'], files_r)
            print("--- [Checkpoint 6] Thread Finished ---")
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            import traceback
            traceback.print_exc()