# Lab 12: Scale-Invariant Feature Transform (SIFT) and Harris Corner Detection

## Overview

Lab 12 explores keypoint detection and feature matching, fundamental techniques for object recognition, image alignment, and 3D reconstruction. This laboratory introduces two complementary methods: SIFT (Scale-Invariant Feature Transform) for robust local feature detection and Harris corner detection for corner-based feature identification. These techniques enable matching corresponding points between images despite changes in scale, rotation, illumination, and viewpoint.

SIFT and Harris corner detection form the foundation of modern image matching and are essential for applications including object recognition, panoramic image stitching, and structure-from-motion 3D reconstruction. The laboratory demonstrates practical implementation and evaluation of these powerful feature matching techniques.

## Detailed Description

### Harris Corner Detection

#### Motivation and Intuition

Corners are distinctive image features characterized by:
- Rapid intensity changes in multiple directions
- High information content for recognition
- Geometric significance for object structure
- Relative stability across image variations

**Why Corners?**
- Edges have gradient in one direction only
- Flat regions have no meaningful gradient
- Corners have significant gradients in multiple directions
- Ideal for matching and localization

#### Mathematical Foundation

Harris corner detection is based on the autocorrelation matrix (also called Harris matrix or structure tensor):

**Autocorrelation Matrix:**
```
M = [ Σ(Ix²)    Σ(IxIy)  ]
    [ Σ(IxIy)   Σ(Iy²)   ]
```

Where:
- Ix: Image gradient in x-direction
- Iy: Image gradient in y-direction
- Σ: Sum over a local window (Gaussian-weighted)

**Eigenvalue Analysis:**

The eigenvalues λ₁ and λ₂ of matrix M characterize local image structure:

- **Both small (λ₁ ≈ λ₂ ≈ 0)**: Flat region, no corner
- **One large, one small (λ₁ >> λ₂)**: Edge, gradient in one direction
- **Both large and similar**: Corner, gradients in multiple directions

**Harris Response:**

```
R = det(M) - k * trace(M)²
R = λ₁λ₂ - k(λ₁ + λ₂)²
```

Where k is a tuning parameter (typically 0.04-0.06)

**Interpretation:**
- Positive R with large value: Strong corner
- R near zero: Edge or flat region
- Negative R: Ridge or valley

#### Harris Algorithm Steps

1. **Gradient Computation:**
   - Apply Sobel filters to compute Ix and Iy
   - Optional Gaussian smoothing for noise reduction

2. **Autocorrelation Matrix:**
   - Compute Ix², Iy², and IxIy at each pixel
   - Apply Gaussian weighting and sum in neighborhood
   - Construct Harris matrix M at each location

3. **Response Computation:**
   - Calculate Harris response R at each pixel
   - R indicates corner strength

4. **Non-Maximum Suppression:**
   - For each pixel, check if it's a local maximum of R
   - Suppress non-maximum values to isolate corner peaks
   - Produces single-pixel corner locations

5. **Threshold:**
   - Apply threshold to select significant corners
   - Typically use fraction of maximum R value
   - Example: T = 0.3 × max(R)

6. **Visualization:**
   - Draw rectangles around detected corners
   - Verify detection quality visually

#### Characteristics

**Advantages:**
- Computationally efficient
- Rotation invariant (gradient-based)
- Repeatable detection across different images of same scene
- Well-understood mathematical foundation

**Limitations:**
- Not scale-invariant (corners at same location in image pyramids)
- Less invariant to perspective transformations
- Requires separate detection at multiple scales for scale invariance
- No associated descriptor for matching

### SIFT: Scale-Invariant Feature Transform

#### Motivation and Innovation

SIFT addresses fundamental limitations of simpler corner detectors:
- Detects distinctive features at multiple scales
- Invariant to scale, rotation, and illumination
- Provides rich descriptor for robust matching
- Highly distinctive features suitable for recognition

**Why Scale Invariance?**
- Same object appears at different sizes (distance variation)
- Features must be detected consistently regardless of scale
- Image pyramids enable multi-scale analysis

#### SIFT Algorithm Overview

**Four Main Components:**

1. **Scale-Space Extrema Detection:**
   - Create image pyramid (Gaussian pyramid)
   - Compute Difference of Gaussians (DoG)
   - Search for local extrema in scale and space
   - Identifies candidate keypoints across scales

2. **Accurate Keypoint Localization:**
   - Refine keypoint positions using sub-pixel accuracy
   - Eliminate poorly localized points (low contrast)
   - Filter out edge responses using Harris measure
   - Produce final set of stable keypoints

3. **Orientation Assignment:**
   - Compute gradient directions and magnitudes around keypoint
   - Create histogram of orientations
   - Assign dominant orientation(s) to keypoint
   - Enables rotation invariance

4. **Descriptor Generation:**
   - Sample gradients in region around keypoint
   - Create 4×4 grid of 8-bin orientation histograms
   - Produce 128-dimensional descriptor vector
   - Normalize for illumination robustness

#### Mathematical Details

**Difference of Gaussians (DoG):**

```
D(x,y,σ) = G(x,y,kσ) - G(x,y,σ)
```

Where G is Gaussian with different scales. DoG approximates the Laplacian of Gaussian, an optimal scale-space operator.

**Scale-Space Search:**

- Extrema must be local max or min compared to:
  - 8 neighbors in same scale level
  - 9 neighbors in scale above
  - 9 neighbors in scale below
- Total of 26 comparisons per candidate

**Orientation Histogram:**

- 36 bins covering full 360° (10° per bin)
- Weighted by gradient magnitude and Gaussian
- Multiple peaks treated as separate keypoints

**Descriptor Creation:**

- 4×4 grid of cells in 16×16 window around keypoint
- Each cell: 8-bin orientation histogram
- Final descriptor: 4×4×8 = 128 dimensions
- Normalized to unit length (L2 normalization)

#### Characteristics

**Advantages:**
- Truly scale-invariant through multi-scale processing
- Rotation-invariant through orientation assignment
- Robust to illumination changes through normalization
- Highly distinctive descriptors enable reliable matching
- Extensive validation on diverse applications
- Industry standard for feature matching

**Limitations:**
- Computationally more expensive than Harris
- Many parameters require tuning
- Patented algorithm (licensing considerations)
- Requires sufficient distinctive content for reliable detection

### Feature Matching

#### Distance-Based Matching

The simplest approach to matching:

```python
def match_features_brute_force(descriptors1, descriptors2):
    """Match features using brute-force search"""
    matches = []
    
    for i, desc1 in enumerate(descriptors1):
        # Compute distances to all descriptors in image 2
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        
        # Find two nearest neighbors
        sorted_indices = np.argsort(distances)
        nearest = sorted_indices[0]
        second_nearest = sorted_indices[1]
        
        # Lowe's ratio test: reject ambiguous matches
        if distances[nearest] < 0.7 * distances[second_nearest]:
            matches.append((i, nearest))
    
    return matches
```

**Lowe's Ratio Test:**
- Compares distance to nearest neighbor vs second nearest
- Ratio < 0.7 indicates distinctive match
- Eliminates ambiguous matches from similar features
- Improves matching reliability

#### Visualization of Matches

Matched keypoints can be visualized by:
- Drawing keypoints in both images
- Connecting matched points with lines
- Assessing match quality visually
- Identifying outliers for filtering

## Key Learning Objectives

Upon completion of this laboratory, students will:

1. Understand corner detection principles and applications
2. Implement Harris corner detection algorithm
3. Comprehend Harris response and interpretation
4. Apply non-maximum suppression for corner localization
5. Understand SIFT algorithm and its components
6. Extract SIFT keypoints and descriptors
7. Perform feature matching between images
8. Apply Lowe's ratio test for outlier rejection
9. Visualize and evaluate matching results
10. Recognize applications in image recognition and alignment
11. Compare Harris and SIFT characteristics
12. Implement image matching pipelines

## Technical Implementation Details

### Harris Corner Detection

```python
from scipy.ndimage import maximum_filter, gaussian_filter, sobel

def harris_corners(img, sigma=1.0, k=0.04, threshold=0.3):
    """Detect Harris corners in image"""
    # Gaussian smoothing
    img_smooth = gaussian_filter(img, sigma=sigma)
    
    # Compute gradients
    Ix = sobel(img_smooth, axis=1)
    Iy = sobel(img_smooth, axis=0)
    
    # Compute matrix elements
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # Gaussian weighted sums
    Sx2 = gaussian_filter(Ixx, sigma=1.0)
    Sxy = gaussian_filter(Ixy, sigma=1.0)
    Sy2 = gaussian_filter(Iyy, sigma=1.0)
    
    # Harris response
    det = (Sx2 * Sy2 - Sxy**2)
    trace = Sx2 + Sy2
    R = det - k * trace**2
    
    # Non-maximum suppression
    R_max = maximum_filter(R, size=3)
    corners = (R > threshold * R.max()) & (R == R_max)
    
    return np.argwhere(corners)
```

### SIFT Feature Extraction

Using OpenCV implementation:

```python
import cv2

def extract_sift_features(image):
    """Extract SIFT keypoints and descriptors"""
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors
```

### Feature Matching

```python
def match_sift_features(des1, des2, ratio_test=0.7):
    """Match SIFT descriptors between two images"""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)
    
    return good_matches
```

## How to Run the Laboratory

### Prerequisites

Install required Python packages:

```bash
pip install numpy opencv-contrib-python scikit-image scipy matplotlib
```

Note: opencv-contrib-python is required for SIFT support.

### Execution Steps

1. Navigate to the Lab 12 directory:
```bash
cd Lab12_SIFT-Harris
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Lab_SIFT_HARRIS_Std.ipynb
```

3. Execute cells in sequence:

   **Harris Corner Detection Part:**
   
   - Import libraries and load test image
   - Apply Gaussian smoothing
   - Compute Sobel gradients
   - Calculate Harris response matrix
   - Apply non-maximum suppression
   - Detect and visualize corners
   - Adjust parameters and observe effects

   **SIFT Feature Extraction Part:**
   
   - Load box and scene images
   - Extract SIFT keypoints and descriptors
   - Perform feature matching between images
   - Visualize matches with connecting lines
   - Apply Lowe's ratio test for filtering
   - Analyze matching results

4. Experimentation:
   - Modify Harris parameters (sigma, k, threshold)
   - Observe corner detection robustness
   - Extract SIFT features from different images
   - Test matching on challenging image pairs
   - Evaluate match quality and outlier rejection

5. Extension activities:
   - Implement affine transformation using matching points
   - Create image mosaic/panorama
   - Estimate homography from point correspondences
   - Implement geometric verification (RANSAC)
   - Visualize feature descriptors

### Expected Outputs

- Harris corner detections with rectangles
- SIFT keypoints with scale and orientation visualization
- Matched feature pairs with connecting lines
- Match quality metrics and statistics
- Analysis of feature distinctiveness
- Evaluation of matching robustness

## Advanced Topics and Extensions

### RANSAC (Random Sample Consensus)

Robust outlier rejection:
- Randomly select minimal sample sets
- Fit geometric models
- Count consistent inliers
- Refine model with best inlier set

### Homography Estimation

Computing perspective transformation:
- From matched point pairs
- Using Direct Linear Transform (DLT)
- Applications in image rectification and mosaic creation

### Image Stitching

Creating panoramic images:
- Match features across adjacent images
- Estimate homographies
- Blend overlapping regions
- Create large panoramic composition

### 3D Reconstruction

Building 3D models from matched features:
- Structure-from-motion algorithms
- Triangulation from multiple views
- Bundle adjustment optimization
- Point cloud generation

### Deep Learning Feature Matching

Modern approaches replacing SIFT:
- SuperPoint for keypoint detection
- SuperGlue for descriptor matching
- End-to-end learning
- Improved robustness and performance

## Real-World Applications

### Object Detection and Recognition
- Finding objects in scenes
- Template matching
- Multiple instance detection

### Image Registration
- Medical image alignment
- Change detection in temporal sequences
- Multi-modal image fusion

### 3D Reconstruction
- Structure-from-motion from image sequences
- Photogrammetry
- Environment mapping

### Panoramic Image Stitching
- Creating wide-angle images
- Video stabilization
- Immersive image creation

### Visual Localization
- Place recognition
- Loop closure detection in SLAM
- Navigation and mapping

## Laboratory Files

- `Lab_SIFT_HARRIS_Std.ipynb`: Main notebook with both algorithms
- `commonfunctions.py`: Helper functions for image processing
- `circuit.tif`: Test image for Harris corner detection
- `box.png`: Reference template for SIFT matching
- `box_in_scene.png`: Scene image containing template
- Image files for feature matching examples

## References and Resources

- Harris corner detection: Harris, C. & Stephens, M. (1988)
- SIFT: Lowe, D. G. (2004) "Distinctive Image Features from Scale-Invariant Keypoints"
- OpenCV documentation: https://docs.opencv.org
- scipy.ndimage filters documentation
- Feature matching fundamentals

## Important Notes

- Harris corners are not scale-invariant; detect at multiple scales for multi-scale analysis
- SIFT is computationally more expensive but provides superior features
- Feature matching quality depends on image content distinctiveness
- Parameter tuning is image-dependent; adjust for different applications
- Always apply ratio test for robust matching
- Visualization is essential for validating detection and matching results
- Document parameter choices and methodology for reproducibility
- Consider real-time performance needs when selecting algorithms
- Combine with geometric verification (RANSAC) for robust results
- Modern alternatives (SuperPoint, SuperGlue) offer improved performance
