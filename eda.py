#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 23:47:07 2024

@author: nitaishah
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt

tile_size = 128
target_value = 2

arr = []
mask_list = []

with rasterio.open('/Users/nitaishah/Downloads/2023_30m_cdls/2023_30m_cdls.tif') as src:
    # Calculate number of tiles in each dimension, handling remainder tiles
    num_tiles_x = 1000
    num_tiles_y = 1000
    print(src.count)
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Define the window for the tile
            window = Window(j * tile_size, i * tile_size, tile_size, tile_size)
            tile = src.read(1, window=window)
            #arr.append(tile)
            mask = (tile == target_value).astype(np.uint8)
            print(mask)
            if np.any(mask):
                # Save the tile as an image (input for U-Net)
                tile_img = Image.fromarray(tile.astype(np.uint8)) 
                print(tile_img)
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                arr.append(tile_img)
                mask_list.append(mask_img)

arr[0]
arr[987]
mask_list[987]


plt.imshow(arr[0])
#plt.imshow(mask_img)

len(arr)
len(mask_list)

for i in range(0,10):
    plt.imshow(arr[i])

arr[10]
mask_list[100]
