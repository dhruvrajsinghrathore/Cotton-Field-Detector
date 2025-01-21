#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 02:08:22 2024

@author: nitaishah
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from scipy.ndimage import label

# Parameters
tile_size = 128  # Define the size of each smaller tile in pixels
target_value = 2  # The value representing the color #ff2626 (cotton)
output_dir = "/Users/nitaishah/Desktop/UNET2"  # New output directory for U-Net data

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to save a tile with annotation for U-Net (image and corresponding mask)
def save_tile_with_annotation_for_unet(src, window, row, col):
    # Read the data within the window
    tile = src.read(1, window=window)  # Read the first band (assumed single-band image)
    
    # Create a binary mask where target_value represents cotton
    mask = (tile == target_value).astype(np.uint8)
    
    # Check if any target pixels are present (i.e., cotton regions)
    if np.any(mask):
        # Save the tile as an image (input for U-Net)
        tile_img = Image.fromarray(tile.astype(np.uint8))  # Original image tile
        tile_img_path = os.path.join(output_dir, f"tile_{row}_{col}.png")
        tile_img.save(tile_img_path)
        
        # Save the mask as an image (output mask for U-Net)
        mask_img = Image.fromarray(mask * 255)  # Mask image (cotton = 255, background = 0)
        mask_img_path = os.path.join(output_dir, f"tile_{row}_{col}_mask.png")
        mask_img.save(mask_img_path)

# Open the TIFF and divide it into tiles
with rasterio.open('/Users/nitaishah/Downloads/2023_30m_cdls/2023_30m_cdls.tif') as src:
    # Calculate number of tiles in each dimension, handling remainder tiles
    num_tiles_x = (src.width + tile_size - 1) // tile_size
    num_tiles_y = (src.height + tile_size - 1) // tile_size
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Define the window for the tile
            window = Window(j * tile_size, i * tile_size, tile_size, tile_size)
            
            # Save tile and annotation if it contains target regions (cotton)
            save_tile_with_annotation_for_unet(src, window, i, j)