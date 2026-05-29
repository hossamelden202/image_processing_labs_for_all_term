# Lab 9: Histogram of Oriented Gradients (HOG) - Feature Extraction and Description

## Overview

Lab 9 introduces the Histogram of Oriented Gradients (HOG), a powerful feature descriptor widely used in computer vision for object detection, recognition, and image classification. HOG captures local shape information and edge characteristics by analyzing the distribution of gradient orientations within image regions. This feature representation has proven particularly effective for tasks such as person detection, object recognition, and texture analysis.

HOG descriptors encode structural and shape information about objects in a way that is robust to small translations and illumination variations while remaining sensitive to significant shape changes. The technique bridges low-level image processing and high-level feature representation, making it invaluable for machine learning-based vision systems.

## Detailed Description

### Gradient Fundamentals

#### Image Gradients

Gradients represent the rate of change of pixel intensities:

**Horizontal Gradient (Gx):**
- Represents intensity changes in the x-direction
- Captures vertical edges
- Computed using horizontal derivative filters (Sobel, Prewitt)

**Vertical Gradient (Gy):**
- Represents intensity changes in the y-direction
- Captures horizontal edges
- Computed using vertical derivative filters

**Gradient Magnitude:**
```
M = sqrt(Gx² + Gy²)
```
Represents the strength of intensity change regardless of direction

**Gradient Direction (Orientation):**
```
θ = arctan(Gy / Gx)
```
Represents the direction of steepest intensity change, typically normalized to [0°, 180°) for edge-based analysis

#### Orientation Quantization

HOG quantizes continuous gradient orientations into discrete bins:

**Bin Assignment Process:**
1. Compute gradient direction for each pixel
2. Assign the pixel to the nearest orientation bin
3. Weight the contribution by gradient magnitude

**Typical Bin Configuration:**
- 9 bins for edge-based orientation (0°-180°)
- Each bin covers 20° (180°/9)
- Alternative: 18 bins for non-edge analysis (0°-360°)

**Weighted Histogram:**
Each pixel contributes to its assigned bin with weight equal to its gradient magnitude. This ensures strong edges (high magnitude) have greater influence than weak intensity changes.

### HOG Descriptor Structure

#### Image Decomposition

The HOG process divides the image into a grid of cells:

**Cell Size:**
- Typical 4x4 to 8x8 pixels per cell
- Smaller cells capture fine details; larger cells capture coarse structure
- Affects computational cost and feature sensitivity

**Cell Histogram:**
Each cell computes a histogram of the quantized orientations for all pixels within that cell, weighted by their gradient magnitudes.

#### Block Structure

Cells are grouped into blocks for local normalization:

**Block Characteristics:**
- Typically 2x2 cells per block
- Overlapping blocks increase robustness
- Block stride (spacing between block centers) controls overlap

**Normalization:**
Each block histogram is normalized to have unit length (L2 normalization):
```
norm = sqrt(sum of squared histogram values)
normalized_histogram = histogram / norm
```

This normalization:
- Makes HOG invariant to local contrast changes
- Improves robustness to illumination variations
- Prevents very bright regions from dominating

#### Final Descriptor

The complete HOG descriptor is a concatenation of all block histograms:

**Descriptor Properties:**
- Length: (Blocks_per_image) × (Cells_per_block) × (Orientation_bins)
- Example: Image decomposed into 16×8 blocks of 2×2 cells with 9 bins = 16×8×2×2×9 = 4608 dimensions
- High-dimensional feature representation capturing detailed shape information

### HOG Algorithm Overview

**Phase 1: Image Preprocessing**
- Convert to grayscale if necessary
- Resize to fixed dimensions (128×64 is common)
- Optional Gaussian smoothing to reduce noise

**Phase 2: Gradient Computation**
- Apply horizontal gradient filter to compute Gx
- Apply vertical gradient filter to compute Gy
- Compute magnitude and orientation for each pixel

**Phase 3: Cell Histograms**
- For each cell in the grid:
  - Extract gradient magnitudes and orientations
  - Create histogram of orientations weighted by magnitude
  - Normalize or store for later processing

**Phase 4: Block Normalization**
- For each block of cells:
  - Concatenate cell histograms
  - Apply L2 normalization
  - Store normalized block histogram

**Phase 5: Final Descriptor**
- Concatenate all normalized block histograms
- Result is the final HOG feature vector

### Key Characteristics and Advantages

**Advantages:**
- Robust to small geometric and photometric transformations
- Effective for detecting objects with distinct edges and gradients
- Works well for texture and shape analysis
- Computationally efficient compared to deep learning approaches
- Interpretable features with clear relationship to image content

**Limitations:**
- Requires careful parameter tuning (cell size, block configuration)
- Less effective for objects without clear edge structure
- Not rotation invariant (requires separate handling for rotated objects)
- Computationally more expensive than simpler features like raw pixels
- May not capture long-range structural information effectively

## Key Learning Objectives

Upon completion of this laboratory, students will:

1. Understand gradient computation and orientation concepts
2. Implement custom gradient calculation using filters
3. Compute gradient magnitude and direction
4. Quantize continuous orientations into discrete bins
5. Create cell-based histograms of oriented gradients
6. Understand block-based normalization and its benefits
7. Construct complete HOG descriptors from images
8. Apply HOG to feature-based image classification
9. Recognize parameter effects on descriptor properties
10. Use HOG features for object detection and recognition

## Technical Implementation Details

### Gradient Computation

```python
def compute_gradient(img, grad_filter):
    """Convolve image with gradient filter"""
    # Pad image for border handling
    ts = grad_filter.shape[0]
    padded_img = np.zeros((img.shape[0] + ts - 1, img.shape[1] + ts - 1))
    padded_img[int((ts-1)/2.0):img.shape[0] + int((ts-1)/2.0),
               int((ts-1)/2.0):img.shape[1] + int((ts-1)/2.0)] = img
    
    # Convolution operation
    result = np.zeros_like(padded_img)
    for r in range(int((ts-1)/2.0), img.shape[0] + int((ts-1)/2.0)):
        for c in range(int((ts-1)/2.0), img.shape[1] + int((ts-1)/2.0)):
            region = padded_img[r-int((ts-1)/2.0):r+int((ts-1)/2.0)+1,
                                c-int((ts-1)/2.0):c+int((ts-1)/2.0)+1]
            result[r, c] = np.sum(region * grad_filter)
    
    # Remove padding
    return result[int((ts-1)/2.0):result.shape[0]-int((ts-1)/2.0),
                  int((ts-1)/2.0):result.shape[1]-int((ts-1)/2.0)]
```

### Magnitude and Direction Computation

```python
def compute_gradient_magnitude(gx, gy):
    """Compute gradient magnitude from components"""
    return np.sqrt(np.power(gx, 2) + np.power(gy, 2))

def compute_gradient_direction(gx, gy):
    """Compute gradient direction and normalize to [0, 180)"""
    eps = 1e-5
    direction = np.arctan2(gy, gx + eps)
    # Convert from radians to degrees and normalize
    direction_deg = np.degrees(direction)
    direction_deg[direction_deg < 0] += 180
    return direction_deg
```

### Histogram Creation

```python
def create_cell_histogram(magnitudes, directions, num_bins=9):
    """Create weighted histogram of orientations"""
    hist = np.zeros(num_bins)
    bin_width = 180.0 / num_bins
    
    for mag, direction in zip(magnitudes.flatten(), directions.flatten()):
        bin_idx = int(direction / bin_width) % num_bins
        hist[bin_idx] += mag
    
    return hist
```

## How to Run the Laboratory

### Prerequisites

Install required Python packages:

```bash
pip install numpy scikit-image scipy opencv-python matplotlib
```

### Execution Steps

1. Navigate to the HOG lab directory:
```bash
cd "Lab9-HoG- Lab10 Classification/HoG"
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Lab9-HoG-STD.ipynb
```

3. Execute cells sequentially:
   - Import libraries and helper functions
   - Define custom gradient computation functions
   - Load test image from images/ directory
   - Compute gradients (horizontal and vertical)
   - Calculate gradient magnitude and direction
   - Create histograms for image cells
   - Apply block normalization
   - Construct complete HOG descriptor
   - Visualize HOG features

4. Experimentation and analysis:
   - Modify cell and block sizes to observe effects
   - Change number of orientation bins
   - Apply HOG to different image types
   - Compare HOG representations for similar and different objects
   - Analyze descriptor vectors for interpretability

5. Extension activities:
   - Implement HOG visualization (drawing gradient directions)
   - Create HOG-based classification pipeline
   - Compare with OpenCV HOGDescriptor implementation
   - Apply to pedestrian detection or other tasks

### Expected Outputs

- Computed gradient images (Gx, Gy, magnitude, direction)
- Cell histograms showing orientation distributions
- Normalized block histograms
- Final HOG feature vectors
- Visualization of HOG features and gradients
- Analysis of parameter effects on descriptor characteristics

## Advanced Topics and Extensions

### HOG Visualization

Creating interpretable visualizations:
- Plotting gradient orientations as arrows
- Coloring by magnitude strength
- Overlaying on original image
- Identifying distinctive features

### HOG Normalization Variants

Different normalization strategies:
- L2 normalization (Euclidean)
- L1 normalization (Manhattan)
- L2-sqrt normalization
- Effects on classification performance

### Multi-Scale HOG

Analyzing at multiple scales:
- Computing HOG at different image resolutions
- Image pyramid construction
- Combining multi-scale features
- Improved robustness to scale variations

### Soft Binning

Improved interpolation for orientation assignment:
- Distributing gradient contribution across multiple bins
- Smooth histogram creation
- Reduced aliasing artifacts
- Better gradient representation

### HOG for Rotation Invariance

Handling rotated objects:
- Dominant orientation detection
- Rotating descriptor to canonical orientation
- Creating rotation-invariant representations

## Real-World Applications

### Pedestrian Detection
- Detecting people in images and video
- Security and surveillance systems
- Autonomous vehicle perception

### Object Detection
- Detecting specific object categories
- Counting objects in images
- Part-based detection models

### Action Recognition
- Identifying human activities in images
- Sports analysis
- Surveillance-based event detection

### Texture Analysis
- Material classification
- Surface inspection
- Quality control systems

### Medical Image Analysis
- Tissue characterization
- Abnormality detection
- Organ boundary identification

## Laboratory Files

- `Lab9-HoG-STD.ipynb`: Main notebook with HOG implementation and demonstrations
- `commonfunctions.py`: Helper functions for image processing
- `images/source/`: Source images for HOG computation
- `images/reference/`: Reference images for comparison

## References and Resources

- Histogram of Oriented Gradients: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
- scipy.ndimage filters documentation
- OpenCV HOGDescriptor documentation
- NumPy mathematical operations reference

## Important Notes

- Image preprocessing (resizing, normalization) affects final HOG descriptors
- Cell and block sizes must be chosen based on object size and detail level
- Gradient computation methods (Sobel, Prewitt) may produce slight variations
- Normalization is critical for robust features
- HOG works best for objects with distinctive edge structures
- Parameter values should be consistent for feature comparison across images
- Document parameter choices for reproducibility and comparability
- Consider computational efficiency for real-time applications
