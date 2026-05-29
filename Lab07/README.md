# Lab 7: Image Segmentation Part 1 - Color-Based Object Separation

## Overview

Lab 7 introduces the fundamental concepts of image segmentation, the process of partitioning images into distinct regions or objects with homogeneous properties. This first segmentation laboratory focuses on color-based segmentation techniques, leveraging the information encoded in color channels to separate objects from their backgrounds. Color-based segmentation is particularly effective when objects possess distinct color characteristics different from their surroundings.

This laboratory provides practical experience in analyzing color characteristics, developing segmentation strategies based on color information, and implementing methods to extract specific objects or regions from complex images. Students will work with RGB color space decomposition, channel analysis, and threshold-based separation techniques.

## Detailed Description

### Color Space Fundamentals

#### RGB Color Space

The RGB (Red, Green, Blue) color model represents colors as combinations of three primary color channels:

- **Red Channel**: Contains intensity information for the red component (0-255)
- **Green Channel**: Contains intensity information for the green component (0-255)
- **Blue Channel**: Contains intensity information for the blue component (0-255)

**Properties:**
- Additive color model: colors formed by adding red, green, and blue light
- Widely used in digital imaging and display devices
- Straightforward implementation and interpretation
- Effective for many color-based segmentation tasks

#### Color Decomposition and Channel Analysis

In this laboratory, images are decomposed into separate color channels for individual analysis:

```python
red_channel = image[:,:,0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]
```

Each channel can be analyzed independently to determine:
- Which channel contains the strongest distinction between object and background
- Intensity distributions within specific regions
- Optimal threshold values for separation

### Color-Based Segmentation Strategy

The laboratory uses a systematic approach to color-based segmentation:

#### Phase 1: Visual Analysis and Channel Selection

Before implementing any algorithmic processing:

1. **Visual Inspection**: Examine the image to identify objects and background
2. **Dominant Color Identification**: Determine which color dominates the object of interest
3. **Channel Comparison**: Display each color channel separately to identify the channel with best object-background separation
4. **Intensity Range Analysis**: Determine the typical intensity values for object and background in the selected channel

#### Phase 2: Threshold Development

Once a dominant channel is identified:

1. **Threshold Determination**: Select a threshold value that separates object and background
2. **Binary Mask Creation**: Create a binary image where 1 represents object pixels and 0 represents background
3. **Threshold Validation**: Visually inspect results to ensure correct object extraction

#### Phase 3: Multi-Channel Discrimination

For improved segmentation accuracy:

1. **Comparative Analysis**: Compare relative intensities across channels
2. **Differential Channel Masking**: Create masks based on differences between channels
3. **Composite Mask Creation**: Combine multiple channel-based masks for robust segmentation

### Segmentation Approach for Golf Course Image

The laboratory uses a golf course image where the objective is to segment the golf ball (white object) from the green background:

**Challenge Analysis:**
- Golf ball: Bright white or light color on multiple channels
- Green grass: Strong green channel intensity
- Objective: Extract the white ball while excluding the green background

**Solution Strategy:**

The dominant channel approach:
- The green channel has maximum intensity in the green grass
- The red and blue channels have lower intensity in green grass
- The white ball has high intensity in all channels

**Discrimination Method:**
A mask identifying the golf ball can be created using:
```
Ball_Mask = (Red > Threshold) AND (Blue > Threshold) AND (Red > Green) AND (Blue > Green)
```

This captures pixels that are:
- Sufficiently bright in red and blue channels (white characteristics)
- Brighter in red/blue than in green (non-green characteristics)
- Isolated from pure green background pixels

### Binary Image Conversion

The final step converts the multi-channel analysis into a binary segmentation mask:

**Binary Image Properties:**
- Value of 1 (white/true) indicates object pixels (golf ball)
- Value of 0 (black/false) indicates background pixels (grass)
- Can be visualized directly or used for further processing
- Serves as input for subsequent morphological operations

## Key Learning Objectives

Upon successful completion of this laboratory, students will:

1. Understand color space decomposition and channel-based analysis
2. Perform visual analysis to identify dominant color characteristics
3. Implement threshold-based segmentation strategies
4. Develop multi-channel discrimination criteria
5. Create binary segmentation masks from color information
6. Visualize segmentation results and assess quality
7. Apply post-processing techniques to improve masks
8. Recognize limitations of color-based approaches
9. Extend techniques to different objects and backgrounds
10. Integrate segmentation with morphological operations

## Technical Implementation Details

### Channel Decomposition

Extract individual color channels from RGB image:

```python
def decompose_image(image):
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]
    return red, green, blue
```

### Threshold-Based Segmentation

Create binary mask using threshold:

```python
def threshold_segmentation(channel, threshold):
    mask = np.zeros_like(channel)
    mask[channel > threshold] = 1
    return mask
```

### Multi-Channel Mask Creation

Combine conditions from multiple channels:

```python
def color_based_segmentation(image, r_thresh, g_thresh, b_thresh):
    r, g, b = decompose_image(image)
    
    # White pixels have high values in all channels
    white_mask = (r > r_thresh) & (g > g_thresh) & (b > b_thresh)
    
    # Non-green pixels (for golf ball on green background)
    non_green_mask = (r > g) & (b > g)
    
    # Combined mask
    final_mask = white_mask & non_green_mask
    
    return final_mask.astype(int)
```

### Binary Image Visualization

Display both input image and resulting segmentation mask for comparison and validation.

## How to Run the Laboratory

### Prerequisites

Install required Python packages:

```bash
pip install numpy scikit-image matplotlib scipy
```

### Execution Steps

1. Navigate to the Lab 7 directory:
```bash
cd Lab07
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Lab_Seg_1_STD.ipynb
```

3. Execute cells in sequence:
   - Import necessary libraries and helper functions
   - Load the golf course image from imgs/exp1/
   - Convert image to float for proper computation
   - Extract and visualize individual color channels
   - Visually identify dominant channel for the target object
   - Display thresholded channel to validate discrimination ability
   - Implement channel-based discrimination logic
   - Create and visualize binary segmentation mask
   - Analyze and refine results

4. For extended exploration:
   - Experiment with different threshold values
   - Test on different images in the experiments folders
   - Combine multiple segmentation criteria
   - Apply morphological operations to refine masks
   - Compare results with different approaches

5. Validation steps:
   - Verify that the object is correctly segmented
   - Check for false positives (background regions incorrectly labeled as object)
   - Check for false negatives (object regions incorrectly labeled as background)
   - Assess the continuity and completeness of the segmented region

### Expected Outputs

- Individual display of red, green, and blue channels
- Thresholded channel images showing object-background separation
- Binary segmentation masks clearly showing the extracted object
- Comparison between different threshold values
- Analysis of segmentation quality and accuracy

## Advanced Topics and Extensions

### Multi-Object Segmentation

Extending to segment multiple objects:
- Using different threshold ranges for different colors
- Creating separate masks for each object type
- Combining masks appropriately

### Color Space Alternatives

Using different color spaces for improved discrimination:
- **HSV (Hue, Saturation, Value)**: Better for color-based separation
- **Lab Color Space**: Perceptually uniform, good for natural images
- **YCbCr**: Common in video and image compression
- **Normalized RGB**: Reduces illumination sensitivity

### Illumination Invariance

Handling varying lighting conditions:
- Normalization techniques
- Adaptive thresholding based on local statistics
- Color constancy algorithms
- Preprocessing normalization

### Morphological Post-Processing

Improving segmentation masks:
- Removing noise using opening
- Filling holes using closing
- Connectivity analysis to remove spurious regions
- Edge refinement through dilation/erosion

### Machine Learning Integration

Advanced segmentation approaches:
- Pixel classification using machine learning
- Training classifiers on color features
- Probability maps and soft segmentation
- Neural network-based segmentation

## Real-World Applications

### Agricultural Analysis
- Crop and weed segmentation in field images
- Plant health assessment through color analysis
- Automated harvesting systems

### Medical Imaging
- Tissue type segmentation based on staining
- Skin lesion analysis and classification
- Organ and abnormality detection

### Quality Control and Inspection
- Defect detection on colored products
- Color-based object sorting
- Surface analysis and anomaly detection

### Autonomous Systems
- Road and lane segmentation
- Traffic sign detection
- Obstacle identification

### Retail and E-Commerce
- Product recognition and classification
- Color-based object detection
- Inventory management systems

## Laboratory Files

- `Lab_Seg_1_STD.ipynb`: Main notebook with segmentation demonstrations
- `commonfunctions.py`: Helper functions for image processing
- `imgs/exp1/golf.jpeg`: Golf course image for experiment 1
- `imgs/exp2/`: Additional images for experiment 2

## References and Resources

- scikit-image color processing documentation
- RGB color space theory and applications
- Image segmentation fundamentals
- Thresholding techniques and selection methods

## Important Notes

- Color-based segmentation effectiveness depends on object-background color contrast
- Lighting conditions significantly affect color values; consider preprocessing for robustness
- Single-threshold approaches may produce suboptimal results; multi-channel analysis often improves results
- Visual inspection is essential for validating segmentation quality
- Different images and objects require parameter adjustments
- Document color characteristics and optimal thresholds for reproducibility
- Consider combining color information with texture or edge information for robust segmentation
- Always preserve original images; segmentation results depend on accurate parameter tuning
