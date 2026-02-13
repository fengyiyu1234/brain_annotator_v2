# src/data_loader.py
import os
import glob
import pandas as pd
import tifffile
import numpy as np
import concurrent.futures
from PyQt5.QtCore import QThread, pyqtSignal
from collections import OrderedDict

# 引入配置和工具
from .config import CACHE_LIMIT_GB
from .utils import natural_sort_key, normalize_percentile

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
            
            self.progress_update.emit(90, "Normalizing Images...")
            stack_r = normalize_percentile(stack_r)
            stack_g = normalize_percentile(stack_g)
            
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