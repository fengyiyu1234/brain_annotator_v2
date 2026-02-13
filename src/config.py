# config.py
import numpy as np

# --- Parameters ---
# Resolution: Z, Y, X (um)
PIXEL_SIZE_RAW = (20.0, 0.65, 0.65)   
PIXEL_SIZE_NAV_SAMPLE = (25.0, 25.0, 25.0) 

# Calculated Scaling Factors (Global to Raw)
# Example: 1 px in Nav (25um) = ~38.4 px in Raw (0.65um)
SCALE_FACTOR_Z = PIXEL_SIZE_NAV_SAMPLE[0] / PIXEL_SIZE_RAW[0]
SCALE_FACTOR_Y = PIXEL_SIZE_NAV_SAMPLE[1] / PIXEL_SIZE_RAW[1]
SCALE_FACTOR_X = PIXEL_SIZE_NAV_SAMPLE[2] / PIXEL_SIZE_RAW[2]

TILE_SHAPE_PX = (2048, 2048)          
CROP_SHAPE_PX = (256, 256) # New crop size
CACHE_LIMIT_GB = 200 

# --- Colors & Classes ---
COLOR_MAP = {
    'red':     (1, 0, 0, 1),
    'green':   (0, 1, 0, 1),
    'yellow':  (1, 1, 0, 1),
    'cyan':    (0, 1, 1, 1),
    'magenta': (1, 0, 1, 1), 
    'blue':    (0, 0.6, 1, 1),
    'white':   (1, 1, 1, 1)   
}

CLASS_ID_MAP = {
    'red glia': 0, 'green glia': 1, 'yellow glia': 2,
    'red neuron': 3, 'green neuron': 4, 'yellow neuron': 5,
    'type_q': 6, 'type_w': 7, 'type_e': 8,
    'other': 99 
}