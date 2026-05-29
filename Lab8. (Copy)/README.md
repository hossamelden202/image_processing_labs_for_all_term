# Lab 8: Advanced Image Segmentation - Adaptive Thresholding and Techniques

## Overview

Lab 8 presents advanced segmentation techniques, building upon foundational color-based methods to address more complex segmentation scenarios. This laboratory focuses on adaptive thresholding algorithms that adjust locally based on image content, rather than using fixed global thresholds. Adaptive approaches are particularly valuable when images exhibit varying illumination, complex backgrounds, or when segmentation quality must be maintained across diverse image conditions.

The laboratory implements comprehensive segmentation solutions using iterative mean-based algorithms, explores multiple segmentation strategies, and provides practical experience with advanced image analysis. These techniques form the foundation for robust image understanding in real-world applications with challenging conditions.

## Detailed Description

### Fundamental Thresholding Concepts

#### Global vs. Adaptive Thresholding

**Global Thresholding:**
- Uses a single threshold value for the entire image
- Simple and computationally efficient
- Fails when illumination varies across the image
- May incorrectly classify pixels in shadows or brightly lit regions

**Adaptive Thresholding:**
- Computes local threshold values based on neighborhood statistics
- Adapts to local image conditions
- Handles varying illumination effectively
- More computationally intensive but produces superior results

### Iterative Mean-Based Threshold Calculation

#### Algorithm Overview

The iterative mean-based algorithm computes an optimal global threshold by iteratively refining an estimate based on the statistics of foreground and background pixels:

**Mathematical Foundation:**

The algorithm divides the histogram into two regions separated by the current threshold estimate and calculates the mean of each region:

```
T_new = (μ_lower + μ_upper) / 2
```

Where:
- μ_lower: Mean intensity of pixels with value < T_current
- μ_upper: Mean intensity of pixels with value ≥ T_current

**Algorithm Steps:**

1. **Initialization:**
   - Compute the overall image mean: μ_image
   - Set initial threshold T = μ_image

2. **Iterative Refinement:**
   - For each iteration:
     - Identify pixels below and above current threshold
     - Compute mean intensity for pixels below threshold (μ_lower)
     - Compute mean intensity for pixels above threshold (μ_upper)
     - Calculate new threshold: T_new = (μ_lower + μ_upper) / 2
     - Check convergence: If T_new == T_current, algorithm converges
     - Otherwise, set T_current = T_new and repeat

3. **Convergence:**
   - When the threshold stabilizes (stops changing), the algorithm has converged
   - The final threshold value is optimal for this iterative approach

**Convergence Properties:**
- The algorithm typically converges in 3-10 iterations
- Guaranteed convergence for most image histograms
- Produces good results without parameter tuning

#### Histogram Analysis

The threshold calculation depends on image histogram characteristics:

- **Bimodal Histogram**: Clear separation between foreground and background peaks
  - Algorithm performs well
  - Threshold lies near the valley between peaks
  
- **Unimodal Histogram**: No clear separation in histogram
  - Algorithm still produces a result but may be less meaningful
  - Additional preprocessing or manual adjustment may be needed

#### Advantages and Limitations

**Advantages:**
- Automatic calculation requiring no manual threshold selection
- Computationally efficient
- Works well for images with distinct foreground and background
- Robust to illumination variations within regions

**Limitations:**
- May not work optimally for complex histograms or multiple objects
- Requires image conversion to consistent intensity range
- Results depend on histogram shape and object-background contrast
- Single threshold may be insufficient for complex scenes

### Binary Image Creation

After threshold calculation:

```python
def apply_threshold(image, threshold):
    # Convert image to uint8 if necessary
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()
    
    # Apply threshold
    binary_image = np.zeros_like(image_uint8)
    binary_image[image_uint8 >= threshold] = 255
    binary_image[image_uint8 < threshold] = 0
    
    return binary_image
```

### Histogram Computation

The histogram represents the frequency distribution of intensity values:

```python
def calculate_histogram(image):
    # Ensure image is uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Initialize counts array for each intensity level
    counts = np.zeros(256, dtype=int)
    
    # Count pixels at each intensity level
    for intensity in range(256):
        counts[intensity] = np.sum(image == intensity)
    
    return counts
```

## Key Learning Objectives

Upon completion of this laboratory, students will:

1. Understand the concept of adaptive versus global thresholding
2. Implement iterative mean-based threshold calculation algorithms
3. Analyze image histograms to understand intensity distributions
4. Compute optimal threshold values automatically
5. Apply thresholds to create binary segmentation masks
6. Understand convergence properties of iterative algorithms
7. Recognize when adaptive thresholding is appropriate
8. Implement post-processing techniques to improve results
9. Apply segmentation to various image types and conditions
10. Combine threshold-based segmentation with morphological operations

## Technical Implementation Details

### Threshold Calculation Function

Complete implementation of iterative mean-based thresholding:

```python
def getThreshold(image):
    # Convert to uint8 if necessary
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()
    
    # Calculate histogram
    counts = calculate_histogram(image_uint8)
    
    # Initialize threshold as image mean
    grey_levels = np.arange(256)
    total_pixels = np.sum(counts)
    Tinit = round(np.sum(grey_levels * counts) / total_pixels)
    
    threshold = Tinit
    max_iterations = 100
    
    for iteration in range(max_iterations):
        # Calculate mean of lower region
        lower_range = list(range(0, threshold))
        lower_counts = counts[lower_range]
        total_lower = np.sum(lower_counts)
        
        if total_lower > 0:
            mean_lower = np.sum(np.array(lower_range) * lower_counts) / total_lower
        else:
            mean_lower = 0
        
        # Calculate mean of upper region
        upper_range = list(range(threshold, 256))
        upper_counts = counts[upper_range]
        total_upper = np.sum(upper_counts)
        
        if total_upper > 0:
            mean_upper = np.sum(np.array(upper_range) * upper_counts) / total_upper
        else:
            mean_upper = 255
        
        # Calculate new threshold
        new_threshold = round((mean_lower + mean_upper) / 2)
        
        # Check convergence
        if new_threshold == threshold:
            break
        
        threshold = new_threshold
    
    return threshold
```

### Histogram-Based Analysis

Visualization and analysis of histograms:

```python
def plot_histogram(image, title="Image Histogram"):
    counts = calculate_histogram(image)
    plt.figure()
    plt.plot(counts)
    plt.title(title)
    plt.xlabel("Intensity Level")
    plt.ylabel("Frequency")
    plt.show()
```

## How to Run the Laboratory

### Prerequisites

Install required Python packages:

```bash
pip install numpy scikit-image matplotlib scipy
```

### Execution Steps

1. Navigate to the Lab 8 directory:
```bash
cd "Lab8. (Copy)"
```

2. Run the segmentation script:
```bash
python segmentation_lab.py
```

Or execute individual experiments:

3. For bonus experiments:
```bash
python bonus_experiment.py
```

4. To use in Jupyter notebook (if configured):
```bash
jupyter notebook
```

### Expected Outputs

- Histogram plots showing intensity distributions
- Calculated threshold values for test images
- Binary segmentation masks with clear object-background separation
- Comparison of different threshold values and their effects
- Analysis of algorithm convergence
- Visual assessment of segmentation quality
- Results from bonus experiments demonstrating advanced techniques

## Advanced Topics and Extensions

### Otsu's Method

An alternative automatic thresholding technique:
- Minimizes within-class variance of foreground and background
- Often produces better results than simple mean-based methods
- Available in scikit-image: `threshold_otsu()`

### Multi-Level Thresholding

Segmenting into more than two classes:
- Extending iterative methods to handle multiple thresholds
- Creating multi-class segmentation masks
- Applications in color image segmentation

### Local Adaptive Thresholding

Using different thresholds for different image regions:
- Computing local statistics within sliding windows
- Handling complex illumination variations
- Available in scikit-image: `threshold_local()`

### Morphological Post-Processing

Improving binary masks:
- Removing noise with morphological opening
- Filling holes with closing
- Cleaning segmentation artifacts
- Boundary refinement

### Multi-Scale Analysis

Analyzing images at multiple scales:
- Creating image pyramids
- Threshold calculation at different scales
- Combining results for robust segmentation

## Real-World Applications

### Document Analysis
- Binarization of scanned documents
- Text extraction from images
- Historical document restoration and digitization

### Medical Image Analysis
- Tissue segmentation from grayscale medical images
- Tumor detection and boundary identification
- Preparation of masks for surgical planning

### Industrial Quality Control
- Defect detection on manufactured products
- Surface inspection and analysis
- Automated measurement systems

### Satellite and Aerial Imagery
- Land cover classification
- Water body detection
- Urban area mapping

### Video Surveillance
- Moving object detection
- Background subtraction
- Event detection and analysis

## Laboratory Files

- `segmentation_lab.py`: Main Python script implementing segmentation algorithms
- `bonus_experiment.py`: Advanced experiments and additional techniques
- `images/`: Directory containing test images for segmentation
- Sample images for various application domains

## References and Resources

- Otsu's automatic thresholding method
- Histogram theory and applications
- Image segmentation fundamentals
- scikit-image segmentation documentation
- Adaptive thresholding techniques

## Important Notes

- Threshold values are image-dependent; different images may require different thresholds
- Always visualize histograms to understand image characteristics before thresholding
- Iterative algorithms may require different numbers of iterations for different images
- Binary image quality significantly affects downstream processing (morphology, analysis)
- Document threshold values and segmentation parameters for reproducibility
- Consider image preprocessing (normalization, smoothing) before thresholding
- Combination of thresholding with morphological operations often produces superior results
- Validate results visually to ensure correct object extraction
