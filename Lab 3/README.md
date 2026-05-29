# Lab 3: Image Smoothing and Noise Reduction Filters

## Overview

Lab 3 focuses on the fundamental concept of image filtering through smoothing techniques. This laboratory provides a comprehensive exploration of noise reduction in digital images using both custom-implemented algorithms and established scientific image processing libraries. The primary objective is to understand how filtering operations can effectively suppress noise while preserving important image features and structures.

Smoothing filters are essential preprocessing operations in computer vision and image analysis workflows. They reduce noise, smooth sharp transitions, and prepare images for subsequent processing steps such as edge detection or feature extraction. This lab emphasizes the practical implementation of these filters and their mathematical foundations.

## Detailed Description

### Core Concepts

The laboratory demonstrates three fundamental filtering techniques with varying characteristics and use cases:

#### 1. Median Filter

The median filter is a non-linear spatial filtering technique that is particularly effective for removing salt-and-pepper noise (also known as impulse noise) while preserving edge information better than linear filters. The filter operates by:

- Establishing a sliding window of specified dimensions (default 3x3) across the image
- Extracting all pixel values within the window region
- Computing the median value of the extracted values
- Replacing the center pixel with the computed median value

The median filter excels at preserving edge sharpness because median operations inherently maintain transitions between different intensity levels. This makes it superior to simple averaging filters when the goal is to remove noise while keeping edges intact. The implementation includes both a custom algorithm and comparison with scikit-image's optimized version.

#### 2. Gaussian Filter

The Gaussian filter applies a Gaussian (bell-curve) weighted convolution across the image. Unlike the median filter, this is a linear operation that weights nearby pixels more heavily than distant ones according to a Gaussian distribution. The key characteristics include:

- Sigma parameter controls the spread of the Gaussian kernel, directly influencing the degree of smoothing
- Large sigma values (e.g., 8) create strong blurring effects suitable for coarse feature extraction
- Small sigma values (e.g., 0.2) produce minimal smoothing, useful when fine details must be preserved
- The filter is computationally efficient and mathematically elegant

The Gaussian filter is widely used as a preliminary step in image pyramids, multi-scale analysis, and as a general-purpose smoothing operation. It also serves as the foundation for several advanced techniques in computer vision.

### Noise Characteristics

The laboratory specifically uses salt-and-pepper noise to demonstrate filter effectiveness. This noise type:

- Randomly sets some pixels to their minimum (0) or maximum (255) values
- Preserves most of the image while introducing extreme outliers
- Represents a realistic scenario in image acquisition from faulty sensors or transmission errors
- Provides clear visual evidence of noise reduction effectiveness

## Key Learning Objectives

Upon completion of this laboratory, students will:

1. Develop a thorough understanding of sliding window operations in image processing
2. Implement custom median filtering algorithms and understand the underlying mathematics
3. Grasp the concept of parametric smoothing and how parameters affect output quality
4. Recognize the differences between linear (Gaussian) and non-linear (median) filtering approaches
5. Apply proper filtering techniques based on specific noise characteristics and preservation requirements
6. Utilize scientific computing libraries effectively for efficient image processing operations

## Technical Implementation Details

### Median Filter Algorithm

The custom median filter implementation follows these steps:

1. Initialize output image as a copy of the input
2. For each pixel position within valid boundaries (accounting for window overlap):
   - Extract the pixel values in the surrounding window
   - Sort these values to find the median
   - Replace the center pixel with the median value
3. Return the filtered image

The window size parameters (window_height, window_width) determine the neighborhood size. Larger windows produce stronger smoothing but may eliminate fine details.

### Gaussian Filter Implementation

The scikit-image Gaussian filter implementation:

1. Creates a Gaussian kernel of appropriate size based on sigma
2. Normalizes the kernel so weights sum to 1
3. Applies convolution between the kernel and image
4. Returns the filtered image with preserved dimensions

## How to Run the Laboratory

### Prerequisites

Ensure the following Python packages are installed:

```bash
pip install numpy scikit-image matplotlib scipy
```

### Execution Steps

1. Navigate to the Lab 3 directory:
```bash
cd "Lab 3"
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Lab_Smoothing_STD.ipynb
```

3. Execute cells sequentially from top to bottom:
   - The first code cell imports necessary libraries and helper functions
   - Load the bird image and apply salt-and-pepper noise for demonstration
   - Execute the custom median filter implementation
   - Compare with scikit-image's optimized median filter
   - Explore Gaussian filtering with various sigma parameters

4. To run individual filtering operations:
   - Each filter section can be executed independently
   - Modify parameters (window size, sigma value) and re-run to observe effects
   - Experiment with different sigma values to understand smoothing trade-offs

### Expected Outputs

- Visual comparison of noisy image with median-filtered results
- Demonstration of scikit-image's median filter producing similar results
- Multiple Gaussian-filtered versions showing the effect of different sigma parameters
- Clear visualization of how filter parameters affect output quality

## Practical Applications

### Real-World Use Cases

- **Medical Imaging**: Removing sensor noise from X-ray or ultrasound images while preserving diagnostic features
- **Satellite Imagery**: Preprocessing aerial photographs to reduce atmospheric noise before analysis
- **Quality Control**: Smoothing camera feeds in manufacturing environments to improve feature detection
- **Image Restoration**: Recovering detail from degraded historical photographs or compression artifacts

### Filter Selection Guidelines

- Use **median filtering** when dealing with salt-and-pepper noise, impulse noise, or when edge preservation is critical
- Use **Gaussian filtering** for general noise reduction, image pyramid construction, and as a preprocessing step before feature detection
- Combine multiple filtering stages for optimal results in challenging scenarios

## Advanced Considerations

### Computational Efficiency

The custom median filter implementation demonstrates the algorithm's logic clearly but may be slow on large images. For production applications, scikit-image's implementation uses optimized C code and is significantly faster. Considerations include:

- Image dimensions and processing speed trade-offs
- Memory usage for large images
- Window size impact on computational cost

### Filter Kernel Design

Advanced users may explore:

- Custom kernel creation for specific noise profiles
- Bilateral filtering (edge-preserving smoothing)
- Morphological filters for specific structural elements
- Adaptive filtering techniques that adjust locally based on image content

## Laboratory Files

- `Lab_Smoothing_STD.ipynb`: Main Jupyter notebook containing all demonstrations and exercises
- `commonfunctions.py`: Helper module with utility functions like image display
- `bird.jpg`: Sample image used for demonstrations

## References and Resources

- scikit-image documentation: http://scikit-image.org/
- Random noise generation: http://scikit-image.org/docs/0.13.x/api/skimage.util.html#skimage.util.random_noise
- Median filter API: http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median
- Gaussian filter API: http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian

## Notes for Students

- Experiment with different noise amounts and observe how filter effectiveness changes
- Compare visual quality metrics between filtering approaches
- Consider the computational cost when processing large image collections
- Document your findings regarding optimal parameter selection for your specific use cases
