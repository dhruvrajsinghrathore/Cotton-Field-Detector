# Cotton-Field-Detector

Certainly! Here's the content formatted as a GitHub markdown file:

# Cotton Crop Classification from Satellite Imagery

## Project Overview

In this challenge, we developed an end-to-end pipeline for automated cotton crop classification from satellite imagery using models based on semantic segmentation. We utilized the Cropland Data Layer Dataset available on Google Earth Engine to develop our models.

## Dataset

We started with a large TIF file covering the entire country. We then:

1. Supplied windows of 128x128 with no overlap
2. Created grayscale regions with white color annotated to cotton (as per Google Earth Engine annotations)

## Methodology

### Model Architecture

We implemented a U-Net model for semantic segmentation.

### Training Process

1. Sent our training images to the U-Net model
2. Trained the model
3. Obtained test images with annotations based on the model training

### Acreage Calculation

We used mathematical calculations to obtain acreage from the model outputs.

## Challenges and Solutions

### Dataset Development

**Challenge:** Developing the dataset from the raster image was time-consuming.

**Solution:** 
- Divided the image into multiple tiles using a sliding window
- Iteratively saved 128x128 TIF images from the raster file

### Image Processing

**Challenge:** Extracting cotton data from single-channel images.

**Solution:**
- Identified cotton using corresponding pixel values of (2,2,2) in the grayscale image
- Created masks by setting cotton pixels to (255,255,255) and all others to (0,0,0)

### Model Selection

**Challenge:** Initial CNN approach led to overfitting due to dataset imbalance.

**Solution:**
- Switched to semantic segmentation using U-Net
- Allowed identification of cotton regions within each tile, regardless of other crops present

### Area Calculation

**Challenge:** Calculating the amount of cotton present in every image.

**Solution:**
- Determined that each original tile pixel represented 30m²
- Applied semantic model to images
- Calculated the ratio of white pixels to black pixels to obtain the area cotton covers in a single tile

## Conclusion

By overcoming these challenges, we successfully developed a pipeline for cotton crop classification using semantic segmentation, providing a more detailed analysis than traditional classification methods.
