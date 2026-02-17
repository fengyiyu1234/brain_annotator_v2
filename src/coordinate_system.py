import xml.etree.ElementTree as ET
import os
from .config import TILE_SHAPE_PX

class CoordinateSystem:
    def __init__(self, xml_path):
        self.tile_offsets = {} # filename_prefix -> (x, y)
        self.tiles = self._parse_xml(xml_path)
        
    def _parse_xml(self, path):
        if not os.path.exists(path):
            return []

        tree = ET.parse(path)
        root = tree.getroot()
        tiles = []
        
        for i, stack in enumerate(root.findall(".//Stack")):
            # 你确认 XML 中的 ABS_H/V 是像素单位
            # 如果 XML 里的值已经是像 10404 这样的大整数，直接取 int
            abs_x = int(float(stack.get('ABS_H')))
            abs_y = int(float(stack.get('ABS_V')))
            
            dir_name = stack.get('DIR_NAME') # e.g. "346900/346900_306100"
            
            # Key 用于匹配文件名: 346900_306100
            key = dir_name.replace("\\", "/").split("/")[-1]
            
            self.tile_offsets[key] = (abs_x, abs_y)

            tiles.append({
                'id': i,
                'dir': dir_name,
                'abs_x': abs_x,
                'abs_y': abs_y,
                'w': TILE_SHAPE_PX[1],
                'h': TILE_SHAPE_PX[0]
            })
            
        print(f"[CoordSys] Loaded {len(tiles)} tiles.")
        return tiles

    def get_tile_info(self, filename):
        # 从文件名提取 prefix: "346900_306100_277600.tiff" -> "346900_306100"
        parts = filename.split('_')
        if len(parts) >= 2:
            key = f"{parts[0]}_{parts[1]}"
            if key in self.tile_offsets:
                # 返回该 Tile 的全局起始坐标
                ox, oy = self.tile_offsets[key]
                # 还需要返回 dir 路径部分用于拼接路径
                # 简单遍历一下找到 dir (或者优化数据结构)
                for t in self.tiles:
                    if t['dir'].endswith(key):
                        return t
        return None