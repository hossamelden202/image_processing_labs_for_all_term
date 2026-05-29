# Lab 11: Texture Analysis using Gray-Level Co-occurrence Matrices (GLCM)

## Overview

Lab 11 explores texture analysis, a critical component of image understanding that captures the spatial structure and patterns within images. Texture refers to the visual patterns and repetitive structures that characterize surfaces and regions, distinct from shape or color. This laboratory focuses on the Gray-Level Co-occurrence Matrix (GLCM) method, a statistical approach to texture characterization that has been widely used in image analysis, medical imaging, and materials science.

The laboratory demonstrates how to extract quantitative texture descriptors from images, enabling automatic texture classification and segmentation. Students will learn to compute co-occurrence matrices and extract meaningful texture features that distinguish between different material types and surface characteristics.

## Detailed Description

### Texture Fundamentals

#### What is Texture?

Texture is the visual pattern and structure within images that characterizes:
- Surface properties (rough, smooth, bumpy)
- Material types (denim, cotton, fabric)
- Repetitive patterns and structures
- Statistical properties of pixel distributions

**Texture versus Shape:**
- Shape: Overall outline and boundary of objects
- Texture: Fine-grained spatial patterns and structures
- Many objects with same shape have different textures
- Texture recognition requires analyzing local pixel relationships

#### Texture Types

Textures can be categorized as:

1. **Regular/Periodic Textures**: Repeating patterns with consistent structure
   - Cloth weaves
   - Brick patterns
   - Tile arrangements

2. **Random/Stochastic Textures**: No obvious repeating pattern
   - Sand
   - Grass
   - Wood grain

3. **Fractal Textures**: Self-similar patterns at multiple scales
   - Clouds
   - Natural surfaces
   - Some biological textures

### Gray-Level Co-occurrence Matrix (GLCM)

#### Mathematical Concept

The GLCM is a square matrix that counts co-occurrences of gray-level pairs within an image:

**Definition:**
- Matrix dimension: L×L, where L is the number of gray levels (typically 256 for 8-bit images)
- Element (i,j): Number of times a pixel with gray level i is adjacent to a pixel with gray level j
- Adjacency is defined by offset (distance and direction)

**GLCM Computation Process:**

1. **Define Offset**: Specify how adjacent pixels are defined
   - Typical offsets: (1,0) for horizontal, (0,1) for vertical
   - Distance: Usually 1 pixel, can be larger for coarser texture analysis
   - Direction: 0°, 45°, 90°, 135° (or equivalently different offset vectors)

2. **Quantize Gray Levels**: Reduce number of gray levels if needed
   - Original 256 levels → often quantized to 32 or 64
   - Reduces sparsity and improves statistical properties
   - Allows texture analysis at different scales

3. **Scan Image**: For each pixel position:
   - Extract gray level at current position (i)
   - Extract gray level at offset position (j)
   - Increment GLCM element [i,j]

4. **Normalize**: Divide all elements by total co-occurrences
   - Converts counts to probabilities
   - Enables comparison across images of different sizes
   - Results in GLCM where sum of all elements = 1

**Mathematical Formulation:**

For offset (dx, dy), the GLCM element P[i,j] represents the joint probability of finding:
- Gray level i at position (x, y)
- Gray level j at position (x+dx, y+dy)

#### GLCM Properties

The normalized GLCM has important properties:

- **Symmetry**: P[i,j] ≈ P[j,i] (gray level pair order often doesn't matter)
- **Diagonal Elements**: High values indicate repetitive texture (same gray level neighbors)
- **Off-Diagonal Elements**: High values indicate rapid intensity transitions
- **Sparse Structure**: Many zeros for images without certain gray-level combinations

### Texture Features from GLCM

#### Contrast

Measures the local variation in the image:

```
Contrast = Σ Σ (i-j)² * P[i,j]
```

- **High Contrast**: Rapid intensity changes, coarse texture
- **Low Contrast**: Gradual changes, fine texture
- Range: [0, L²] where L is number of gray levels

#### Homogeneity (Inverse Difference Moment)

Measures texture uniformity:

```
Homogeneity = Σ Σ P[i,j] / (1 + (i-j)²)
```

- **High Homogeneity**: Uniform texture, many same-color neighbors
- **Low Homogeneity**: Non-uniform texture, varied color neighbors
- Range: [0, 1]
- Inverse relationship with contrast

#### Energy (Angular Second Moment)

Measures texture orderliness:

```
Energy = Σ Σ P[i,j]²
```

- **High Energy**: Ordered, regular texture
- **Low Energy**: Random, disordered texture
- Range: [0, 1]
- High values indicate strong repeatability

#### Correlation

Measures linear gray-level dependencies:

```
Correlation = Σ Σ ((i-μ_i)(j-μ_j) * P[i,j]) / (σ_i * σ_j)
```

- **High Correlation**: Gray levels are linearly dependent
- **Low Correlation**: Gray levels are independent
- Range: [-1, 1]

### Laboratory Application: Texture-Based Segmentation

The laboratory applies GLCM features to segment an image into regions of different textures (jeans, cotton, background):

**Segmentation Strategy:**

1. **Feature Extraction**: Extract contrast and homogeneity from training patches
2. **Feature Space Construction**: Plot samples in 2D feature space
3. **Classifier Training**: Use feature space distribution to define decision boundaries
4. **Segmentation**: Classify all pixels based on GLCM features
5. **Validation**: Assess segmentation quality

**Feature Space Geometry:**

Different textures occupy different regions in the feature space:
- **Jeans**: Moderate contrast, variable homogeneity
- **Cotton**: Different contrast and homogeneity values
- **Background**: Distinct feature values

Well-separated clusters in feature space enable effective classification.

## Key Learning Objectives

Upon completion of this laboratory, students will:

1. Understand texture analysis principles and applications
2. Comprehend gray-level co-occurrence matrix computation
3. Extract texture features from images
4. Interpret GLCM-based texture descriptors
5. Analyze texture characteristics using feature spaces
6. Segment images based on texture differences
7. Apply GLCM to medical and materials analysis
8. Recognize limitations and extensions of GLCM
9. Implement texture-based classification systems
10. Extract patches and compute GLCM efficiently

## Technical Implementation Details

### GLCM Computation

```python
from skimage.feature import graycomatrix, graycoprops

def compute_glcm(image, distance=1, angle=0, levels=256):
    """Compute Gray-Level Co-occurrence Matrix"""
    # Convert to uint8 if necessary
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=[distance],
        angles=[angle],
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    return glcm
```

### Feature Extraction from GLCM

```python
def get_glcm_features(gray_scale_img):
    """Extract contrast and homogeneity from GLCM"""
    # Compute GLCM with 1 pixel distance and 0° angle
    glcm = graycomatrix(
        gray_scale_img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    # Extract contrast and homogeneity
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return contrast, homogeneity
```

### Batch Processing Patches

```python
def extract_features_from_patches(patch_directory):
    """Extract texture features from all patches in directory"""
    features = []
    filenames = os.listdir(patch_directory)
    
    for filename in filenames:
        if filename.endswith('.jpg'):
            # Load and process image
            img_path = os.path.join(patch_directory, filename)
            img = io.imread(img_path)
            gray_img = rgb2gray(img)
            gray_img = (gray_img * 255).astype(np.uint8)
            
            # Extract features
            contrast, homogeneity = get_glcm_features(gray_img)
            features.append((filename, (contrast, homogeneity)))
    
    return features
```

## How to Run the Laboratory

### Prerequisites

Install required Python packages:

```bash
pip install numpy scikit-image matplotlib scipy opencv-python
```

### Execution Steps

1. Navigate to the Lab 11 directory:
```bash
cd Lab11_Texture
```

2. Open the Jupyter notebook:
```bash
jupyter notebook lab_texture_STD.ipynb
```

3. Execute cells in sequence:

   **Part A - Texture Segmentation:**
   
   - Import libraries and define helper functions
   - Load texture training patches from imgs_patches/
   - Implement get_glcm_features function
   - Extract contrast and homogeneity from samples
   - Plot feature space showing texture cluster distribution
   - Analyze feature space separation
   - Develop segmentation strategy based on feature distribution
   - Apply segmentation to full image

4. Experimentation:
   - Modify GLCM parameters (distance, angles, levels)
   - Extract additional features (energy, correlation)
   - Use different feature combinations for classification
   - Analyze multi-directional GLCM (multiple angles)
   - Apply to different image types

5. Extension activities:
   - Implement multi-scale texture analysis
   - Create texture-based image search
   - Apply machine learning classifiers to texture features
   - Analyze texture evolution in image sequences

### Expected Outputs

- Individual texture patch images and their features
- Feature space plot showing texture cluster positions
- Confusion analysis between similar textures
- Segmented image with texture-based region identification
- Analysis of feature discriminability
- Insights about optimal feature combinations

## Advanced Topics and Extensions

### Multi-Scale GLCM

Analyzing texture at different scales:
- Computing GLCM at multiple distances
- Creating texture profiles
- Scale-invariant texture descriptors

### Multi-Directional GLCM

Analyzing texture isotropy:
- Computing GLCM at multiple angles (0°, 45°, 90°, 135°)
- Comparing directional dependencies
- Detecting oriented patterns

### Additional Texture Features

Extending beyond contrast and homogeneity:
- **Dissimilarity**: Similar to contrast with linear weighting
- **Autocorrelation**: Gray-level dependencies
- **Maximum Probability**: Most frequent co-occurrence

### Local Binary Patterns (LBP)

Alternative texture descriptor:
- Comparing each pixel to its neighbors
- Creating local binary codes
- Computing histograms of binary patterns
- Computationally more efficient than GLCM

### Gabor Filters for Texture

Frequency and orientation-based analysis:
- Filtering at multiple scales and orientations
- Extracting filter responses as features
- Texture characterization in frequency domain

## Real-World Applications

### Medical Image Analysis
- Tissue characterization and classification
- Tumor detection and grade assessment
- Quality control in diagnostic imaging
- Disease progression monitoring

### Materials Science
- Polymer and composite characterization
- Surface quality assessment
- Wear and damage analysis
- Material identification

### Remote Sensing
- Land use and land cover classification
- Texture-based change detection
- Feature identification in satellite imagery
- Urban area analysis

### Industrial Quality Control
- Surface finish inspection
- Defect detection and classification
- Texture-based sorting systems
- Automated visual inspection

### Bioinformatics
- Cell and tissue image analysis
- Microscopy image characterization
- Texture-based disease diagnosis

## Laboratory Files

- `lab_texture_STD.ipynb`: Main notebook with GLCM implementation
- `imgs_patches/`: Directory containing texture training samples
  - Jeans texture patches
  - Cotton texture patches
  - Background patches
- `imgs_we_got_the_patches_from/`: Source images for patch extraction
- Reference texture images for testing

## References and Resources

- GLCM documentation: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.graycomatrix
- GLCM properties: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.graycoprops
- Original GLCM paper: Haralick, R.M., Shanmugam, K., & Dinstein, I.
- Texture analysis theory and applications
- Image feature extraction techniques

## Important Notes

- GLCM computation is sensitive to gray-level quantization; adjust levels based on image quality
- Feature values depend on image statistics; normalization across images is important
- Single offset direction captures directional texture; multiple angles provide more complete description
- GLCM works best for images with distinct texture regions
- Patch size affects feature reliability; ensure sufficient pixel samples
- Different textures may require different parameters for optimal discrimination
- Document parameter choices for reproducibility
- Consider computational efficiency for large-scale analysis
- Combine GLCM features with other descriptors for robust classification
