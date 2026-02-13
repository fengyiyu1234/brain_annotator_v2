# src/coordinate_system.py
import numpy as np
import xml.etree.ElementTree as ET
from .config import PIXEL_SIZE_RAW, PIXEL_SIZE_NAV_SAMPLE, TILE_SHAPE_PX

class CoordinateSystem:
    def __init__(self, xml_path):
        self.tiles = self._parse_xml(xml_path)
        
    def _parse_xml(self, path):
        """Parses the XML to get absolute offsets for each tile."""
        tree = ET.parse(path)
        tiles = []
        for i, stack in enumerate(tree.findall(".//Stack")):
            # Absolute offsets in microns
            abs_z = float(stack.get('ABS_D')) * PIXEL_SIZE_RAW[0]
            abs_y = float(stack.get('ABS_V')) * PIXEL_SIZE_RAW[1]
            abs_x = float(stack.get('ABS_H')) * PIXEL_SIZE_RAW[2]
            
            tiles.append({
                'id': i,
                'row': int(stack.get('ROW')),
                'col': int(stack.get('COL')),
                'dir': stack.get('DIR_NAME'),
                # Limits in microns (z, y, x)
                'min_um': np.array([abs_z, abs_y, abs_x]),
                'max_um': np.array([
                    abs_z + (100 * PIXEL_SIZE_RAW[0]), # Arbitrary depth buffer if Z not in XML
                    abs_y + (TILE_SHAPE_PX[0] * PIXEL_SIZE_RAW[1]),
                    abs_x + (TILE_SHAPE_PX[1] * PIXEL_SIZE_RAW[2])
                ])
            })
        return tiles

    def global_to_local(self, g_z, g_y, g_x):
        """
        Convert Global (Nav/Atlas) pixel coordinates to Tile ID and Local Raw coordinates.
        """
        # 1. Convert Global Pixel -> Global Micron
        um_z = g_z * PIXEL_SIZE_NAV_SAMPLE[0]
        um_y = g_y * PIXEL_SIZE_NAV_SAMPLE[1]
        um_x = g_x * PIXEL_SIZE_NAV_SAMPLE[2]
        
        # 2. Find which tile contains this point (simplistic "first hit" logic)
        # Note: Z matching depends on if XML defines Z stacks. 
        # Assuming 2D grid stitching for XY, and Z is aligned.
        
        target_tile = None
        for t in self.tiles:
            # Check Y and X bounds
            if (t['min_um'][1] <= um_y < t['max_um'][1]) and \
               (t['min_um'][2] <= um_x < t['max_um'][2]):
                target_tile = t
                break
        
        if not target_tile:
            return None, None, None, None

        # 3. Calculate Local Micron -> Local Pixel
        local_um_z = um_z - target_tile['min_um'][0] # Adjust if Z offset exists
        local_um_y = um_y - target_tile['min_um'][1]
        local_um_x = um_x - target_tile['min_um'][2]
        
        raw_z = int(local_um_z / PIXEL_SIZE_RAW[0])
        raw_y = int(local_um_y / PIXEL_SIZE_RAW[1])
        raw_x = int(local_um_x / PIXEL_SIZE_RAW[2])
        
        return target_tile['id'], raw_z, raw_y, raw_x