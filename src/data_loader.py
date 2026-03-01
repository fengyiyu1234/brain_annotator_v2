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
import time
import traceback
from .config import PATCH_SIZE, TILE_SHAPE_PX, CELL_SAMPLE_RATIO, PRELOAD_GLOBAL_DATA
from .utils import natural_sort_key, parse_class_string
import psutil

class GlobalPreloadThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, tiles, root_red, root_green, det_dir):
        super().__init__()
        self.tiles = tiles
        self.root_red = root_red
        self.root_green = root_green
        self.det_dir = det_dir
        self._is_running = True

    def run(self):
        print("\n" + "🚀" * 15)
        print(" [SYSTEM] GLOBAL PRELOAD INITIATED")
        print(f" [SYSTEM] Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
        print("🚀" * 15 + "\n")

        global_cache = {}
        total_tiles = len(self.tiles)
        
        all_tasks = []
        total_files_count = 0
        for t in self.tiles:
            r_dir = os.path.join(self.root_red, t['dir'])
            g_dir = os.path.join(self.root_green, t['dir'])
            
            if os.path.exists(r_dir):
                r_files = [f for f in os.listdir(r_dir) if f.endswith(('.tif', '.tiff'))]
                g_files = []
                if os.path.exists(g_dir):
                    g_files = [f for f in os.listdir(g_dir) if f.endswith(('.tif', '.tiff'))]
                
                if not r_files: continue
                
                if len(r_files) != len(g_files):
                    print(f"❌ [FATAL] 通道数量不匹配！Tile {t['dir']}: 红通道 {len(r_files)}, 绿通道 {len(g_files)}。")
                    self.finished_signal.emit({})
                    return 
                
                total_files_count += len(r_files)
                all_tasks.append({
                    'tile': t, 
                    'r_files': sorted(r_files, key=natural_sort_key),
                    'g_files': sorted(g_files, key=natural_sort_key),
                    'r_folder': r_dir,
                    'g_folder': g_dir
                })

        if total_files_count == 0:
            print("❌ [ERROR] No images found. Check your root paths!")
            self.finished_signal.emit({})
            return

        print(f"统计完成: 共计 {len(all_tasks)} 个有效 Tiles, {total_files_count} 张图像任务。")
        
        start_time = time.time()
        loaded_files_total = 0
        max_workers = 24 # 本地 SSD 可以放心拉高线程数！

        for t_idx, task in enumerate(all_tasks):
            t = task['tile']
            t_files = task['r_files']
            t_prefix = t['dir'].replace('\\', '/').split('/')[-1]
            
            print(f"--- [Loading Tile {t_idx+1}/{total_tiles}] {t_prefix} ---")
            
            stack_r = [None] * len(t_files)
            stack_g = [None] * len(t_files)

            def load_pair(file_idx):
                r_name = task['r_files'][file_idx]
                g_name = task['g_files'][file_idx] 
                
                r_path = os.path.join(task['r_folder'], r_name)
                g_path = os.path.join(task['g_folder'], g_name)
                
                if not os.path.exists(r_path): raise FileNotFoundError(f"红通道文件丢失: {r_path}")
                if not os.path.exists(g_path): raise FileNotFoundError(f"绿通道文件丢失: {g_path}")
                img_r = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)
                img_g = cv2.imread(g_path, cv2.IMREAD_UNCHANGED)

                return file_idx, img_r, img_g

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(load_pair, i) for i in range(len(t_files))]
                for future in concurrent.futures.as_completed(futures):
                    if not self._is_running: break
                    try:
                        f_idx, img_r, img_g = future.result()
                        stack_r[f_idx] = img_r
                        stack_g[f_idx] = img_g
                        
                        loaded_files_total += 1
                        if loaded_files_total % 5 == 0:
                            self._emit_progress(loaded_files_total, total_files_count, t_prefix, t_idx, total_tiles, start_time)
                            
                    except Exception as e:
                        print(f"\n❌ [FATAL ERROR] 图像加载失败: {e}")
                        self._is_running = False
                        break 
            
            if not self._is_running:
                print("🚨 任务已因错误中断！")
                self.finished_signal.emit({})
                return

            # 🌟 核心：传入当前 Tile 真实加载的文件名列表
            df_tile = self._load_csv_for_tile(t_prefix, t_files)

            try:
                global_cache[t['dir']] = {
                    'stack_r': np.array(stack_r),
                    'stack_g': np.array(stack_g),
                    'df': df_tile,
                    'files': t_files,
                    'name': t_prefix
                }
            except MemoryError:
                print("❌ [FATAL] Out of Memory during Numpy conversion!")
                break

        print(f"\n✅ [DONE] All data cached. Time elapsed: {(time.time()-start_time)/60:.2f} min")
        self.finished_signal.emit(global_cache)

    def _load_csv_for_tile(self, t_prefix, loaded_files):
        """🌟 核心逻辑：利用 fname 桥接原始 Z 和当前数组 Index"""
        df_tile = pd.DataFrame()
        if os.path.exists(self.det_dir):
            try:
                target_csv = None
                for f in os.listdir(self.det_dir):
                    if t_prefix in f and f.endswith('.csv') and "_result" in f:
                        target_csv = os.path.join(self.det_dir, f)
                        break
                
                if target_csv:
                    df = pd.read_csv(target_csv, header=None, 
                                     names=['fname', 'x1', 'y1', 'x2', 'y2', 'cls', 'conf', 'intensity', 'z'])
                    if not df.empty:
                        # 1. 扔掉那些没有对应图像的细胞（抽帧抽没的）
                        df = df[df['fname'].isin(loaded_files)].copy()
                        
                        if not df.empty:
                            # 2. 绝对不碰原来的 Z！新增 matrix_idx 记录它在缓存数组里的层数
                            name2idx = {name: idx for idx, name in enumerate(loaded_files)}
                            df['matrix_idx'] = df['fname'].map(name2idx)
                            
                            df['cx'] = (df['x1'] + df['x2']) / 2
                            df['cy'] = (df['y1'] + df['y2']) / 2
                            df_tile = df
            except: pass
        return df_tile

    def _emit_progress(self, current, total, t_name, t_idx, t_total, start_time):
        elapsed = time.time() - start_time
        fps = current / elapsed if elapsed > 0 else 0
        remaining = total - current
        eta_sec = remaining / fps if fps > 0 else 0
        eta_str = f"{int(eta_sec//60)}m {int(eta_sec%60)}s" if eta_sec > 60 else f"{int(eta_sec)}s"
        ram_usage = psutil.virtual_memory().percent

        msg = (
            f"Tile: {t_name} ({t_idx+1}/{t_total})\n"
            f"Files: {current}/{total} | Speed: {fps:.1f} fps\n"
            f"ETA: {eta_str} | RAM: {ram_usage}%"
        )
        self.progress_signal.emit(int((current / total) * 100), msg)

    def stop(self):
        self._is_running = False

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
    
    def __init__(self, anchors, coord_sys, root_red, root_green, det_dir, global_cache=None):
        super().__init__()
        self.anchors = anchors 
        self.tiles = coord_sys.tiles
        self.z_max = coord_sys.z_max
        self.root_red = root_red
        self.root_green = root_green
        self.det_dir = det_dir
        self.global_cache = global_cache or {}

    def _process_single_crop(self, pt, rgb_full_stack, dfs_records, raw_files, t_prefix, center_lz):
        lx, ly = pt['lx'], pt['ly']
        cx_pt, cy_pt = int(lx), int(ly)
        half = PATCH_SIZE // 2
        x1, y1 = cx_pt - half, cy_pt - half
        x2, y2 = cx_pt + half, cy_pt + half
        
        pad_l = max(0, -x1); pad_t = max(0, -y1)
        pad_r = max(0, x2 - TILE_SHAPE_PX[1]); pad_b = max(0, y2 - TILE_SHAPE_PX[0])
        sx1 = max(0, x1); sy1 = max(0, y1)
        sx2 = min(x2, TILE_SHAPE_PX[1]); sy2 = min(y2, TILE_SHAPE_PX[0])
        
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
        
        boxes = []
        for z_i, records in enumerate(dfs_records):
            for row in records:
                if x1 <= row['cx'] <= x2 and y1 <= row['cy'] <= y2:
                    boxes.append({
                        'z_idx': z_i,  
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
        total_anchors = len(self.anchors)
        if 0.0 < CELL_SAMPLE_RATIO < 1.0: 
            keep_count = max(1, int(total_anchors * CELL_SAMPLE_RATIO))
            self.anchors = random.sample(self.anchors, keep_count)

        tasks_by_tile_z = {}
        # 🌟 此处严格遵守你原本的 coordinate 转换逻辑，计算出 target_tile 和 local_z (原始 Z)
        for (gx, gy, gz) in self.anchors:
            target_tile = next((t for t in self.tiles if t['abs_x'] <= gx < t['abs_x'] + t['w'] and t['abs_y'] <= gy < t['abs_y'] + t['h']), None)
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
        patches, det_cache, processed = [], {}, 0
        workers = min(32, (os.cpu_count() or 4) + 4)

        for (t_dir, center_lz), data in tasks_by_tile_z.items():
            t_prefix = t_dir.replace('\\', '/').split('/')[-1] 
            rgb_full_stack, raw_files, dfs_records = [], [], []
            
            for z_idx, lz in enumerate([center_lz - 1, center_lz, center_lz + 1]):
                csv_z = lz + 1  # 回退到原始 CSV 里的 1-based Z 坐标
                
                # --- 分支 A：秒开模式（走全局缓存） ---
                if t_dir in self.global_cache:
                    cache = self.global_cache[t_dir]
                    # 🌟 核心：通过原始 Z 去找对应的 fname，一切以 fname 为准
                    df_z = cache['df'][cache['df']['z'] == csv_z]
                    
                    if not df_z.empty:
                        fname = df_z.iloc[0]['fname']
                        if fname in cache['files']:
                            matrix_idx = int(df_z.iloc[0]['matrix_idx'])
                            ir, ig = cache['stack_r'][matrix_idx], cache['stack_g'][matrix_idx]
                            rgb_full_stack.append(np.dstack((ir, ig, np.zeros_like(ir))))
                            raw_files.append(fname)
                            dfs_records.append(df_z.to_dict('records'))
                        else:
                            rgb_full_stack.append(None); raw_files.append("OUT_OF_BOUNDS"); dfs_records.append([])
                    else:
                        rgb_full_stack.append(None); raw_files.append("OUT_OF_BOUNDS"); dfs_records.append([])
                        
                # --- 分支 B：现场读盘模式 ---
                else:
                    if t_prefix not in det_cache:
                        det_path = next((os.path.join(self.det_dir, f) for f in os.listdir(self.det_dir) if t_prefix in f and f.endswith('.csv') and "_result" in f), None)
                        if det_path:
                            try:
                                df = pd.read_csv(det_path, header=None, names=['fname', 'x1', 'y1', 'x2', 'y2', 'cls', 'conf', 'intensity', 'z'])
                                df['cx'], df['cy'] = (df['x1'] + df['x2']) / 2, (df['y1'] + df['y2']) / 2
                                det_cache[t_prefix] = df
                            except: det_cache[t_prefix] = pd.DataFrame()
                        else: det_cache[t_prefix] = pd.DataFrame()
        
                    df_tile = det_cache[t_prefix]
                    df_z = df_tile[df_tile['z'] == csv_z]

                    r_dir = os.path.join(self.root_red, t_dir)
                    g_dir = os.path.join(self.root_green, t_dir)
                    fr = sorted(os.listdir(r_dir), key=natural_sort_key) if os.path.exists(r_dir) else []
                    fg = sorted(os.listdir(g_dir), key=natural_sort_key) if os.path.exists(g_dir) else []
                    
                    if not df_z.empty:
                        fname = df_z.iloc[0]['fname']
                        if fname in fr:
                            idx = fr.index(fname)
                            r_path = os.path.join(r_dir, fname)
                            g_path = os.path.join(g_dir, fg[idx]) if idx < len(fg) else os.path.join(g_dir, fname)
                            
                            ir = tifffile.imread(r_path)
                            ig = tifffile.imread(g_path) if os.path.exists(g_path) else np.zeros_like(ir)
                            rgb_full_stack.append(np.dstack((ir, ig, np.zeros_like(ir))))
                            raw_files.append(fname)
                            dfs_records.append(df_z.to_dict('records'))
                        else:
                            rgb_full_stack.append(None); raw_files.append("OUT_OF_BOUNDS"); dfs_records.append([])
                    else:
                        rgb_full_stack.append(None); raw_files.append("OUT_OF_BOUNDS"); dfs_records.append([])

            if rgb_full_stack[1] is None:
                processed += 1
                self.progress_signal.emit(int(processed / total_tasks * 100))
                continue

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self._process_single_crop, pt, rgb_full_stack, dfs_records, raw_files, t_prefix, center_lz) for pt in data['pts']]
                for future in concurrent.futures.as_completed(futures):
                    try: patches.append(future.result())
                    except: pass

            processed += 1
            self.progress_signal.emit(int(processed / total_tasks * 100))
            
        self.finished_signal.emit(patches)

# --- 数据加载线程 (也同步加入了按 fname 过滤与映射逻辑) ---
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
        try: first = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
        except Exception as e: raise Exception(f"Read error {file_list[0]}: {e}")
        h, w = first.shape
        stack = np.zeros((count, h, w), dtype=first.dtype)
        
        def load_idx(i):
            try: stack[i] = cv2.imread(file_list[i], cv2.IMREAD_UNCHANGED)
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
            
            exts = ['*.tif', '*.tiff']
            files_r = sorted([f for e in exts for f in glob.glob(os.path.join(path_r, e))], key=natural_sort_key)
            files_g = sorted([f for e in exts for f in glob.glob(os.path.join(path_g, e))], key=natural_sort_key)
            
            if not files_r:
                self.error_occurred.emit(f"No images found in {path_r}")
                return
            
            min_len = min(len(files_r), len(files_g)) if files_g else len(files_r)
            files_r = files_r[:min_len]
            files_g = files_g[:min_len] if files_g else []
            
            # 由于可能只提取文件名，去掉路径
            file_names_only = [os.path.basename(f) for f in files_r]
            
            stack_r = self.read_stack_parallel(files_r, "Red")
            stack_g = self.read_stack_parallel(files_g, "Green") if files_g else np.zeros_like(stack_r)
            
            self.progress_update.emit(95, "Parsing Detections...")
            detections = {}
            
            possible_names = [os.path.basename(dir_rel) + "_result.csv"]
            csv_path = next((os.path.join(self.csv_root, name) for name in possible_names if os.path.exists(os.path.join(self.csv_root, name))), None)
            
            if csv_path:
                try:
                    df = pd.read_csv(csv_path, header=None, 
                                     names=['fname', 'x1', 'y1', 'x2', 'y2', 'class_name', 'score', 'intensity', 'z_idx'])
                    
                    # 🌟 核心：过滤并把 z_idx 重写为本地数组层的 Index
                    df = df[df['fname'].isin(file_names_only)].copy()
                    name2idx = {name: idx for idx, name in enumerate(file_names_only)}
                    df['z_idx'] = df['fname'].map(name2idx)
                    
                    loaded_count = 0
                    for z, group in df.groupby('z_idx'):
                        if pd.notna(z) and 0 <= int(z) < min_len:
                            detections[int(z)] = group.copy()
                            loaded_count += len(group)

                except Exception as e:
                    print(f"[ERROR] CSV Parse Error: {e}")

            self.data_loaded.emit(stack_r, stack_g, detections, self.meta['list_index'], file_names_only)
            
        except Exception as e:
            self.error_occurred.emit(str(e))