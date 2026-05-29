# Lab 5: Edge Detection and Boundary Identification

## Overview

Lab 5 provides a comprehensive exploration of edge detection techniques, which form the foundation of numerous computer vision applications. Edge detection is the process of identifying and extracting boundaries and transitions within digital images where significant changes in pixel intensity occur. These boundaries are crucial for object recognition, shape analysis, feature extraction, and image understanding tasks.

This laboratory investigates multiple edge detection algorithms, each with distinct mathematical properties, computational characteristics, and suitability for different applications. Students will implement custom algorithms and compare them with optimized library implementations to understand both the theoretical foundations and practical performance characteristics.

## Detailed Description

### Fundamental Concepts

#### Edge Definition and Significance

Edges in images represent:
- Boundaries between objects and background
- Transitions between regions of different intensity or color
- Locations where image content changes significantly
- Critical features for object recognition and scene understanding

Edges are detected by identifying locations where image intensity derivatives (gradients) are large. The gradient measures how rapidly pixel intensities change in different directions.

#### 1. Sobel Edge Detection

The Sobel operator is a foundational edge detection method that uses discrete differentiation approximated through convolution. The algorithm:

**Mathematical Foundation:**
- Approximates the gradient (partial derivatives) of image intensity
- Uses two 3x3 kernels to compute horizontal (Gx) and vertical (Gy) derivatives
- The Sobel kernels emphasize the center pixel in the computation

**Horizontal Sobel Kernel (Gx):**
```
[-1  0  +1]
[-2  0  +2]
[-1  0  +1]
```

**Vertical Sobel Kernel (Gy):**
```
[-1  -2  -1]
[ 0   0   0]
[+1  +2  +1]
```

**Characteristics:**
- Produces separate maps for horizontal and vertical edges
- Gradient magnitude combines both components: G = √(Gx² + Gy²)
- Gradient direction indicates edge orientation: θ = arctan(Gy/Gx)
- Computationally efficient using convolution
- Sensitive to noise; smoothing is often applied beforehand

**Applications:**
- General-purpose edge detection
- Object boundary extraction
- Preprocessing for Hough transform
- Feature point detection

#### 2. Prewitt Edge Detection

The Prewitt operator is similar to Sobel but uses different kernel weights:

**Prewitt Kernels:**
- Similar 3x3 structure to Sobel
- Different weighting scheme: uniform weighting (1, 1, 1) instead of (1, 2, 1)
- Slightly different mathematical properties affecting directional sensitivity

**Characteristics:**
- Can detect edges in 8 possible directions (unlike Sobel's 4)
- Enhanced directional sensitivity
- Comparable computational cost to Sobel
- Marginally different noise handling characteristics

#### 3. Roberts Edge Detection

The Roberts operator is a simpler alternative with a smaller computational footprint:

**Mathematical Foundation:**
- Uses 2x2 kernels for approximating derivatives
- Operates on diagonal directions primarily
- Lower computational cost than Sobel or Prewitt
- Highly sensitive to noise due to smaller kernel size

**Characteristics:**
- Excellent edge localization due to smaller kernel
- High sensitivity to noise and artifacts
- Often requires preprocessing with smoothing
- Suitable for clean, high-quality images

**Advantages and Disadvantages:**
- Advantage: Fast computation and precise edge localization
- Disadvantage: Noise sensitivity requires careful preprocessing

#### 4. Canny Edge Detection

The Canny edge detector is an advanced multi-stage algorithm providing superior edge detection performance:

**Algorithm Stages:**

1. **Gaussian Smoothing**: Reduces noise using Gaussian filtering with adjustable sigma parameter
2. **Gradient Computation**: Calculates image gradients (magnitude and direction)
3. **Non-Maximum Suppression**: Thins edges to single-pixel width by suppressing non-maximum gradient values
4. **Double Thresholding**: Classifies edge pixels using two threshold values
5. **Edge Tracking by Hysteresis**: Connects edges and removes isolated edge fragments

**Parameter Meanings:**

- **Sigma**: Controls the Gaussian smoothing strength
  - Large sigma (e.g., 2.0): Detects large-scale features, smoother edges
  - Small sigma (e.g., 0.5): Detects fine details, noisier results
  
- **Low Threshold**: Minimum gradient magnitude for edge classification
  - Pixels below this are definitely not edges
  
- **High Threshold**: Maximum gradient magnitude for strong edges
  - Pixels above this are definitely edges
  - Pixels between thresholds are edges only if connected to strong edges

**Edge Tracking Logic:**
- Strong edges (above high threshold) are always included
- Weak edges (between thresholds) are included only if connected to strong edges
- This connectivity-based approach reduces spurious edge fragments from noise

**Advantages:**
- Superior edge detection quality with minimal spurious detection
- Produces thin, well-localized edges
- Robust parameter selection due to hysteresis mechanism
- Industry-standard for many applications

**Disadvantages:**
- More computationally complex than simpler operators
- Requires careful threshold selection for optimal results
- Parameter tuning needed for different image types

### Edge Detection Principles

All derivative-based edge detection methods operate on the principle that edges occur where:
- The first derivative (gradient) of image intensity is large
- Sharp transitions between intensity levels exist
- The second derivative changes sign (zero-crossings)

The choice of detection method depends on:
- Image quality and noise characteristics
- Required computational speed
- Desired edge properties (thickness, connectivity, artifact rejection)
- Application-specific requirements

## Key Learning Objectives

Upon completion of this laboratory, students will:

1. Understand the mathematical foundations of derivative-based edge detection
2. Implement custom Sobel operators with horizontal and vertical components
3. Comprehend the differences between various edge detection algorithms
4. Apply appropriate edge detection techniques based on image characteristics
5. Utilize gradient magnitude and direction for edge analysis
6. Implement non-maximum suppression for edge thinning
7. Apply double thresholding and hysteresis in Canny edge detection
8. Analyze and compare detection results across different algorithms
9. Make informed decisions about parameter selection
10. Recognize real-world applications of edge detection

## Technical Implementation Details

### Sobel Implementation

Custom Sobel edge detection follows these steps:

1. Apply Sobel-X kernel via convolution to compute horizontal gradients (Gx)
2. Apply Sobel-Y kernel via convolution to compute vertical gradients (Gy)
3. Compute gradient magnitude: G = √(Gx² + Gy²)
4. Optionally compute gradient direction: θ = arctan(Gy/Gx)
5. Normalize magnitude values to displayable range

The convolution operation:
- Pads the image to handle boundary pixels
- Slides the kernel across the image
- Computes dot product of kernel and image region
- Replaces the center pixel with the result

### Canny Algorithm Implementation Details

The Canny implementation in scikit-image:

1. Applies Gaussian filter with specified sigma for noise reduction
2. Computes image gradients using Sobel operators
3. Performs non-maximum suppression:
   - For each pixel, suppresses gradients not aligned with edge direction
   - Keeps only local maxima perpendicular to edge direction
4. Applies double thresholding with low and high thresholds
5. Performs edge tracking using connectivity analysis (8-connectivity)

## How to Run the Laboratory

### Prerequisites

Install required packages:

```bash
pip install numpy scikit-image matplotlib scipy
```

### Execution Steps

1. Navigate to the Lab 5 directory:
```bash
cd "lab 5"
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Lab_Edge_Detection_STD.ipynb
```

3. Execute cells sequentially:
   - Import libraries and helper functions (first cell)
   - Load test image (circuit.tif by default)
   - Apply built-in edge detection operators:
     - Sobel edge detection
     - Prewitt edge detection
     - Roberts edge detection
     - Canny edge detection
   - Examine visual comparison of all methods

4. For custom implementations:
   - Implement horizontal Sobel (sobel_h) separately
   - Implement vertical Sobel (sobel_v) if required
   - Implement combined Sobel magnitude
   - Compare custom results with library implementations

5. Experimentation:
   - Modify Canny parameters (sigma, low_threshold, high_threshold)
   - Apply preprocessing smoothing before edge detection
   - Test on different image types
   - Analyze edge properties (thickness, continuity, artifacts)

### Expected Outputs

- Side-by-side comparison of input image with all edge detection results
- Clear visualization of detected edges for each algorithm
- Quantitative difference between algorithms
- Effects of parameter variation on Canny detection
- Edge quality assessment for different parameter combinations

## Advanced Topics

### Multi-Scale Edge Detection

Applying edge detection at multiple scales:
- Detect edges at different sigma levels
- Combine results to identify edges at various detail levels
- Use Gaussian image pyramids for efficient multi-scale analysis

### Directional Edge Detection

Detecting edges in specific directions:
- Extract Prewitt or Sobel results for specific angle ranges
- Identify vertical and horizontal structures separately
- Detect diagonal features using rotated kernels

### Edge Linking and Boundary Extraction

Post-processing edge maps:
- Connect nearby edge fragments
- Trace complete boundaries
- Remove isolated spurious detections
- Extract parametric representations of contours

### Difference of Gaussians (DoG)

Alternative edge detection approach:
- Subtract one Gaussian-filtered image from another
- Acts as an edge enhancement filter
- Used in feature detection algorithms
- Provides continuous-valued edge responses

## Real-World Applications

### Object Detection and Recognition
- Extracting object boundaries for shape analysis
- Identifying regions of interest for feature extraction
- Feeding edge maps into machine learning classifiers

### Medical Image Analysis
- Detecting organ and tissue boundaries
- Identifying tumor margins
- Measuring anatomical structures

### Autonomous Vehicles
- Road boundary detection
- Lane marking identification
- Obstacle boundary extraction

### Document Analysis
- Text region boundary identification
- Table structure detection
- Document layout analysis

### Quality Control and Manufacturing
- Defect edge detection for measurement
- Part boundary identification
- Surface inspection

## Laboratory Files

- `Lab_Edge_Detection_STD.ipynb`: Main notebook with all edge detection demonstrations
- `commonfunctions.py`: Helper functions for image display and processing
- `circuit.tif`: Sample integrated circuit image for edge detection testing

## References and Resources

- scikit-image edge detection: https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/edges.py
- Edge filter examples: http://scikit-image.org/docs/0.11.x/auto_examples/plot_edge_filter.html
- Canny edge detection: http://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html
- Gradient and derivative theory
- Convolution and filtering operations

## Important Notes

- Preprocessing with smoothing often improves results by reducing noise sensitivity
- Edge detection is sensitive to image quality, noise, and contrast
- Parameter selection depends heavily on specific images and applications
- No single edge detector is optimal for all scenarios
- Combine edge detection with other techniques for robust feature extraction
- Document parameter choices when reporting results for reproducibility
