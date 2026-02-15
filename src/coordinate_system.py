# src/coordinate_system.py
import numpy as np
import xml.etree.ElementTree as ET
from .config import PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE, TILE_SHAPE_PX

class CoordinateSystem:
    def __init__(self, xml_path):
        self.tiles = self._parse_xml(xml_path)
        
    def _parse_xml(self, path):
        """
        Parses the XML to get absolute offsets for each tile.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        
        # 尝试获取 Voxel Dims
        v_dims = root.find("voxel_dims")
        if v_dims is not None:
            res_z = float(v_dims.get('D'))
            res_y = float(v_dims.get('V'))
            res_x = float(v_dims.get('H'))
        else:
            res_z, res_y, res_x = PIXEL_SIZE_RAW

        tiles = []
        for i, stack in enumerate(tree.findall(".//Stack")):
            # Absolute offsets in pixels (TeraStitcher XML: V=Y, H=X)
            abs_v_px = float(stack.get('ABS_V'))
            abs_h_px = float(stack.get('ABS_H'))
            abs_d_px = float(stack.get('ABS_D'))
            
            # Convert to Microns
            abs_z_um = abs_d_px * res_z
            abs_y_um = abs_v_px * res_y
            abs_x_um = abs_h_px * res_x
            
            # Calculate Bounding Box (Microns)
            min_um = np.array([abs_z_um, abs_y_um, abs_x_um])
            max_um = np.array([
                abs_z_um + (100 * res_z), # Z buffer
                abs_y_um + (TILE_SHAPE_PX[0] * res_y),
                abs_x_um + (TILE_SHAPE_PX[1] * res_x)
            ])
            
            tiles.append({
                'id': i,
                'list_index': i,      # 【必须保留】修复 data_loader
                'row': int(stack.get('ROW')),
                'col': int(stack.get('COL')),
                'dir': stack.get('DIR_NAME'),
                'min_um': min_um,
                'max_um': max_um,
                'origin_um': min_um   # 【核心修复】这里必须有 origin_um
            })
            
        print(f"[CoordinateSystem] Loaded {len(tiles)} tiles from XML.")
        return tiles

    def global_to_local(self, g_z, g_y, g_x):
        """
        Convert Global (Nav/Atlas) pixel coordinates to Tile ID and Local Raw coordinates.
        """
        # 1. Global Pixels -> Global Microns
        um_z = g_z * PIXEL_SIZE_NAV_SAMPLE[0]
        um_y = g_y * PIXEL_SIZE_NAV_SAMPLE[1]
        um_x = g_x * PIXEL_SIZE_NAV_SAMPLE[2]
        
        # 2. Find target tile
        target_tile = None
        for t in self.tiles:
            # Check Y and X bounds with small buffer
            if (t['min_um'][1] <= um_y < t['max_um'][1]) and \
               (t['min_um'][2] <= um_x < t['max_um'][2]):
                target_tile = t
                break
        
        if not target_tile:
            return None, None, None, None

        # 3. Global Microns -> Local Pixels
        # Local microns = Global microns - Tile Origin
        local_um_z = um_z - target_tile['origin_um'][0]
        local_um_y = um_y - target_tile['origin_um'][1]
        local_um_x = um_x - target_tile['origin_um'][2]
        
        # Convert back to raw pixels
        raw_z = int(local_um_z / PIXEL_SIZE_RAW[0])
        raw_y = int(local_um_y / PIXEL_SIZE_RAW[1])
        raw_x = int(local_um_x / PIXEL_SIZE_RAW[2])
        
        return target_tile['list_index'], raw_z, raw_y, raw_x