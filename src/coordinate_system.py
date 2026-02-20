import xml.etree.ElementTree as ET
import os
from .config import TILE_SHAPE_PX

class CoordinateSystem:
    def __init__(self, xml_path):
        self.tiles = [] 
        self.XY_RESIZING = 1.0 
        self.z_max = 0  # 新增：保存全局最大的 Z 起跳点 (用于削平山头)
        self._parse_xml(xml_path)
        
    def _parse_xml(self, path):
        if not os.path.exists(path):
            print(f"[Error] XML not found: {path}")
            return

        try:
            tree = ET.parse(path)
            root = tree.getroot()
            stacks = root.findall(".//Stack")
            
            if not stacks: return

            # 寻找 XY 极小值用于平移原点
            min_x = min(float(s.get('ABS_H')) for s in stacks)
            min_y = min(float(s.get('ABS_V')) for s in stacks)
            
            # 寻找最大的 ABS_D 作为 Z 轴的齐平基准！
            self.z_max = max(int(float(s.get('ABS_D', 0.0))) for s in stacks)
            
            count = 0
            for i, stack in enumerate(stacks):
                raw_abs_h = float(stack.get('ABS_H'))
                raw_abs_v = float(stack.get('ABS_V'))
                raw_abs_z = int(float(stack.get('ABS_D', 0.0)))
                
                # 减去极小值，绝对不除以 10
                abs_x = int((raw_abs_h - min_x) * self.XY_RESIZING)
                abs_y = int((raw_abs_v - min_y) * self.XY_RESIZING)
                
                dir_name = stack.get('DIR_NAME', '')
                norm_dir = dir_name.replace("\\", "/") 
                
                self.tiles.append({
                    'id': i,
                    'dir': norm_dir,
                    'abs_x': abs_x,
                    'abs_y': abs_y,
                    'abs_z': raw_abs_z,  # 记录原始 Z 偏移量
                    'w': TILE_SHAPE_PX[1], 
                    'h': TILE_SHAPE_PX[0]  
                })
                count += 1
                
            print(f"[CoordinateSystem] Loaded {count} tiles. Z_MAX baseline = {self.z_max}")
            
        except Exception as e:
            print(f"[Error] Failed to parse XML: {e}")