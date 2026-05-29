# Image Processing Laboratory Course - Complete Documentation

## Overview

This comprehensive image processing laboratory course provides hands-on experience with fundamental and advanced techniques in digital image analysis and computer vision. The course is structured as a progressive series of 12 integrated laboratories that build from basic concepts to sophisticated applications.

The laboratory course emphasizes both theoretical understanding and practical implementation. Each laboratory includes complete source code, detailed explanations, and real-world application contexts. Students will implement algorithms from scratch, understand mathematical foundations, and leverage established image processing libraries.

## Course Structure

The laboratories are organized in a carefully designed progression that builds foundational knowledge and skills:

### Foundation Level (Labs 1-2)

These laboratories establish the mathematical and programming foundation necessary for image processing:

- **Lab 1**: Python fundamentals and basic syntax
- **Lab 2**: Convolution and Discrete Fourier Transform

### Core Image Processing (Labs 3-7)

These laboratories introduce fundamental image processing techniques:

- **Lab 3**: Image smoothing and noise reduction filters
- **Lab 4**: Contrast enhancement and intensity transformation
- **Lab 5**: Edge detection and boundary identification
- **Lab 6**: Mathematical morphology and image transformation
- **Lab 7**: Image segmentation using color-based methods

### Advanced Segmentation (Labs 8-10)

These laboratories explore sophisticated segmentation and classification:

- **Lab 8**: Adaptive thresholding and advanced segmentation
- **Lab 9**: Histogram of Oriented Gradients (HOG) feature extraction
- **Lab 10**: Image classification using machine learning

### Feature Analysis (Labs 11-12)

These laboratories examine feature extraction and matching:

- **Lab 11**: Texture analysis using Gray-Level Co-occurrence Matrices
- **Lab 12**: SIFT and Harris corner detection for feature matching

## Detailed Laboratory Descriptions

### Lab 1: Python Fundamentals and Basic Syntax

**Location**: `Lab1/`

**Objectives:**
- Understand Python's dynamic typing system
- Master arithmetic operations and comparisons
- Work with lists and NumPy arrays
- Implement loops and conditional logic
- Define and use functions

**Key Topics:**
- Variable declaration and data types
- Arithmetic operations (division, modulo, exponentiation)
- Boolean logic and conditional statements
- Loop structures (for, while)
- Function definition and keyword arguments
- NumPy array creation and operations
- Conditional indexing with boolean masks

**Content:**
- `Python Basic Syntax.ipynb`: Interactive notebook with Python fundamentals
- `lol1.ipynb`: Additional practice exercises
- Demonstration of essential programming concepts

**How to Run:**
```bash
cd Lab1
jupyter notebook "Python Basic Syntax.ipynb"
```

**Estimated Time**: 3-4 hours

---

### Lab 2: Convolution and Discrete Fourier Transform

**Location**: `Lab2-std/`

**Objectives:**
- Understand convolution operation and spatial filtering
- Learn Fourier Transform fundamentals and frequency-domain analysis
- Apply the Convolution Theorem for efficient filtering
- Manipulate images in frequency domain
- Compare spatial and frequency-domain approaches

**Key Topics:**
- Convolution mathematical foundation and implementation
- Spatial filtering with various kernels
- Discrete Fourier Transform (DFT) concepts
- Frequency-domain representation and interpretation
- Forward and inverse FFT transformations
- Frequency-based filtering operations
- Boundary condition handling

**Content:**
- `lab2-std.ipynb`: Comprehensive notebook on convolution and FFT
- `commonfunctions.py`: Helper functions for visualization
- `imgs/`: Directory with test images

**How to Run:**
```bash
cd Lab2-std
jupyter notebook lab2-std.ipynb
```

**Estimated Time**: 4-5 hours

---

### Lab 3: Image Smoothing and Noise Reduction Filters

**Location**: `Lab 3/`

**Objectives:**
- Understand noise characteristics and filtering effects
- Implement median filter algorithm from scratch
- Apply Gaussian smoothing with various parameters
- Compare linear and non-linear filtering approaches
- Recognize filter selection based on image characteristics

**Key Topics:**
- Median filtering for salt-and-pepper noise removal
- Gaussian filtering with adjustable sigma
- Sliding window operations and convolution
- Edge preservation in filtering
- Kernel size and smoothing strength relationships

**Content:**
- `Lab_Smoothing_STD.ipynb`: Main notebook with filtering demonstrations
- `commonfunctions.py`: Helper functions for image display
- `bird.jpg`: Sample image for demonstrations

**How to Run:**
```bash
cd "Lab 3"
jupyter notebook Lab_Smoothing_STD.ipynb
```

**Estimated Time**: 3-4 hours

---

### Lab 4: Image Contrast Enhancement and Intensity Transformation

**Location**: `Lab 4/`

**Objectives:**
- Apply intensity transformations for image enhancement
- Implement histogram equalization for contrast improvement
- Analyze histogram distributions
- Understand cumulative distribution functions
- Make informed decisions about enhancement strength

**Key Topics:**
- Negative transformation and image inversion
- Histogram computation and analysis
- Histogram equalization algorithm
- Cumulative distribution function (CDF)
- Local versus global contrast enhancement
- Adaptive histogram equalization (CLAHE)

**Content:**
- `Lab4_STD.ipynb`: Contrast enhancement demonstrations
- `commonfunctions.py`: Helper functions
- `extra_task.txt`: Additional challenges and exercises
- Sample images for enhancement

**How to Run:**
```bash
cd "Lab 4"
jupyter notebook Lab4_STD.ipynb
```

**Estimated Time**: 3-4 hours

---

### Lab 5: Edge Detection and Boundary Identification

**Location**: `lab 5/`

**Objectives:**
- Understand gradient-based edge detection principles
- Implement multiple edge detection algorithms
- Apply the Canny edge detector with parameter tuning
- Analyze gradient magnitude and direction
- Recognize algorithm differences and characteristics

**Key Topics:**
- Sobel, Prewitt, and Roberts operators
- Gradient magnitude and direction computation
- Non-maximum suppression and edge thinning
- Canny edge detection algorithm
- Double thresholding and hysteresis
- Parameter selection and effects

**Content:**
- `Lab_Edge_Detection_STD.ipynb`: Edge detection demonstrations
- `commonfunctions.py`: Helper functions
- `circuit.tif`: Test image for edge detection

**How to Run:**
```bash
cd "lab 5"
jupyter notebook Lab_Edge_Detection_STD.ipynb
```

**Estimated Time**: 4-5 hours

---

### Lab 6: Mathematical Morphology and Image Transformation

**Location**: `Lab06/`

**Objectives:**
- Understand morphological operations (erosion, dilation)
- Implement opening and closing for noise removal
- Apply skeletonization for structural analysis
- Select appropriate structuring elements
- Chain multiple operations for image cleaning

**Key Topics:**
- Erosion and dilation operations
- Opening (erosion followed by dilation)
- Closing (dilation followed by erosion)
- Skeletonization and thinning
- Structuring element selection
- Binary image preparation and thresholding
- Contour extraction

**Content:**
- `lab-Morphology-STD.ipynb`: Morphological operations demonstrations
- `commonfunctions.py`: Helper functions
- `img/coins.jpg`: Binary image for processing
- Test images in `img/` directory

**How to Run:**
```bash
cd Lab06
jupyter notebook lab-Morphology-STD.ipynb
```

**Estimated Time**: 4-5 hours

---

### Lab 7: Image Segmentation Part 1 - Color-Based Object Separation

**Location**: `Lab07/`

**Objectives:**
- Analyze color space decomposition (RGB channels)
- Develop color-based segmentation strategies
- Create binary segmentation masks from color information
- Implement multi-channel discrimination
- Extend segmentation to different images and objects

**Key Topics:**
- RGB color space fundamentals
- Channel-based analysis and visualization
- Threshold-based segmentation
- Multi-channel mask creation
- Binary image conversion
- Segmentation quality assessment

**Content:**
- `Lab_Seg_1_STD.ipynb`: Segmentation demonstrations
- `commonfunctions.py`: Helper functions
- `imgs/exp1/golf.jpeg`: Golf course image for color-based segmentation
- `imgs/exp2/`: Additional experiment images

**How to Run:**
```bash
cd Lab07
jupyter notebook Lab_Seg_1_STD.ipynb
```

**Estimated Time**: 3-4 hours

---

### Lab 8: Advanced Image Segmentation - Adaptive Thresholding and Techniques

**Location**: `Lab8. (Copy)/`

**Objectives:**
- Implement iterative mean-based threshold calculation
- Understand adaptive versus global thresholding
- Analyze image histograms for content understanding
- Apply automatic threshold selection
- Implement post-processing for mask improvement

**Key Topics:**
- Global versus adaptive thresholding concepts
- Iterative threshold refinement algorithm
- Histogram computation and interpretation
- Threshold convergence and optimization
- Binary image creation and validation
- Morphological post-processing techniques

**Content:**
- `segmentation_lab.py`: Main Python script with segmentation algorithms
- `bonus_experiment.py`: Advanced experiments and extended techniques
- `images/`: Directory with test images

**How to Run:**
```bash
cd "Lab8. (Copy)"
python segmentation_lab.py
# Or run bonus experiments:
python bonus_experiment.py
```

**Estimated Time**: 3-4 hours

---

### Lab 9: Histogram of Oriented Gradients (HOG) - Feature Extraction and Description

**Location**: `Lab9-HoG- Lab10 Classification/HoG/`

**Objectives:**
- Understand gradient computation and orientation concepts
- Implement custom gradient filters
- Quantize orientations into discrete bins
- Create cell-based histograms
- Apply block normalization for robust features
- Construct complete HOG descriptors

**Key Topics:**
- Gradient magnitude and direction computation
- Orientation quantization and binning
- Cell structure and histogram creation
- Block normalization and L2 normalization
- Descriptor vector concatenation
- Feature dimensionality and interpretation

**Content:**
- `Lab9-HoG-STD.ipynb`: HOG implementation and demonstrations
- `commonfunctions.py`: Helper functions
- `images/source/`: Source images for HOG computation
- `images/reference/`: Reference images for comparison

**How to Run:**
```bash
cd "Lab9-HoG- Lab10 Classification/HoG"
jupyter notebook Lab9-HoG-STD.ipynb
```

**Estimated Time**: 4-5 hours

---

### Lab 10: Image Classification using Machine Learning

**Location**: `Lab9-HoG- Lab10 Classification/Classification/`

**Objectives:**
- Implement complete image classification pipeline
- Extract multiple feature types (HSV histograms, HOG, raw pixels)
- Apply machine learning classifiers (KNN, SVM, Neural Networks)
- Prepare data with proper train-test splitting
- Evaluate and compare classification approaches
- Analyze classification errors and results

**Key Topics:**
- Feature extraction strategies comparison
- HSV histogram-based features
- HOG feature extraction for classification
- Raw pixel features and limitations
- Machine learning classifier implementation
- Data normalization and scaling
- Cross-validation and parameter tuning
- Confusion matrices and error analysis

**Content:**
- `classification-STD.ipynb`: Classification pipeline demonstrations
- `digits_dataset/`: Organized handwritten digit images (0-9)
- `NOTES.txt`: Additional documentation
- `TO-INSTALL.txt`: Package requirements

**How to Run:**
```bash
cd "Lab9-HoG- Lab10 Classification/Classification"
jupyter notebook classification-STD.ipynb
```

**Estimated Time**: 5-6 hours

---

### Lab 11: Texture Analysis using Gray-Level Co-occurrence Matrices (GLCM)

**Location**: `Lab11_Texture/`

**Objectives:**
- Understand texture analysis fundamentals
- Compute Gray-Level Co-occurrence Matrices (GLCM)
- Extract texture descriptors (contrast, homogeneity, energy, correlation)
- Apply texture-based segmentation
- Create and analyze feature spaces

**Key Topics:**
- Texture fundamentals and properties
- GLCM computation and normalization
- Texture feature extraction (contrast, homogeneity)
- Feature space visualization and analysis
- Texture-based classification
- Multi-directional GLCM analysis
- Texture versus shape and color information

**Content:**
- `lab_texture_STD.ipynb`: Texture analysis demonstrations
- `imgs_patches/`: Texture sample patches for training
  - Jeans texture patches
  - Cotton texture patches
  - Background patches
- `imgs_we_got_the_patches_from/`: Source images for patch extraction

**How to Run:**
```bash
cd Lab11_Texture
jupyter notebook lab_texture_STD.ipynb
```

**Estimated Time**: 4-5 hours

---

### Lab 12: Scale-Invariant Feature Transform (SIFT) and Harris Corner Detection

**Location**: `Lab12_SIFT-Harris/`

**Objectives:**
- Implement Harris corner detection algorithm
- Understand scale-space analysis and image pyramids
- Extract SIFT keypoints and descriptors
- Perform feature matching between images
- Apply geometric verification and outlier rejection
- Recognize applications in image recognition

**Key Topics:**
- Harris corner detection principles
- Autocorrelation matrix computation
- Harris response and non-maximum suppression
- SIFT algorithm components
- Scale-space extrema detection
- Orientation assignment and consistency
- Descriptor generation and matching
- Lowe's ratio test for reliable matching
- Feature matching and visualization

**Content:**
- `Lab_SIFT_HARRIS_Std.ipynb`: SIFT and Harris demonstrations
- `commonfunctions.py`: Helper functions
- `circuit.tif`: Test image for Harris corner detection
- `box.png`: Template image for SIFT matching
- `box_in_scene.png`: Scene containing template for SIFT matching

**How to Run:**
```bash
cd Lab12_SIFT-Harris
jupyter notebook Lab_SIFT_HARRIS_Std.ipynb
```

**Estimated Time**: 5-6 hours

---

## Technology Stack

### Required Software

**Python 3.7+**: Modern Python version with extensive library support

**Essential Libraries:**
- **NumPy**: Numerical computing and array operations
- **scikit-image**: Image processing algorithms and tools
- **OpenCV (cv2)**: Computer vision library with optimized implementations
- **Matplotlib**: Visualization and plotting
- **SciPy**: Scientific computing (signal processing, optimization)
- **scikit-learn**: Machine learning algorithms and tools

### Installation Instructions

**Using pip (recommended):**

```bash
pip install numpy scikit-image opencv-contrib-python matplotlib scipy scikit-learn
```

**Using conda (if available):**

```bash
conda install -c conda-forge numpy scikit-image opencv matplotlib scipy scikit-learn
```

**Specific packages for advanced features:**

```bash
# For OpenCV contrib modules (SIFT, extended features)
pip install opencv-contrib-python

# For enhanced visualization
pip install jupyter matplotlib

# For machine learning extensions
pip install scikit-learn imutils
```

### Jupyter Notebook Environment

All laboratories use Jupyter notebooks for interactive development:

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter server
jupyter notebook
```

## Prerequisites and Preparation

### Mathematical Knowledge

Students should have familiarity with:
- Linear algebra (matrices, vectors, transformations)
- Calculus (derivatives, gradients)
- Basic statistics (mean, variance, distributions)
- Signal processing basics (frequency, filtering)

### Programming Experience

Expected prior knowledge:
- Basic programming concepts (variables, loops, functions)
- Working with arrays and matrices
- File input/output operations
- Debugging and testing code

### Time Requirements

**Total Course Duration**: Approximately 40-50 hours

**Breakdown by laboratory:**
- Foundation (Labs 1-2): 7-9 hours
- Core Processing (Labs 3-7): 17-21 hours
- Advanced Segmentation (Labs 8-10): 12-15 hours
- Feature Analysis (Labs 11-12): 9-11 hours

## Using the Laboratories

### General Workflow

1. **Read the Laboratory Documentation**: Each lab includes comprehensive README.md explaining concepts
2. **Examine Sample Code**: Review provided implementations and comments
3. **Run the Notebook**: Execute cells in sequence to understand algorithms
4. **Experiment**: Modify parameters and observe effects
5. **Implement Extensions**: Add custom features or solve additional challenges
6. **Document Findings**: Record observations and insights

### Recommended Approach

**Phase 1: Passive Learning**
- Read documentation thoroughly
- Execute provided code
- Observe and understand outputs
- Study algorithm explanations

**Phase 2: Active Practice**
- Modify parameters and re-run
- Experiment with different images
- Test edge cases
- Compare different approaches

**Phase 3: Creative Application**
- Implement custom features
- Extend algorithms
- Apply to personal images
- Solve new problems

## Real-World Applications

The techniques learned in these laboratories are applied in numerous real-world domains:

### Medical Imaging
- Tumor detection and measurement
- Organ segmentation and analysis
- Image registration and alignment
- Quality control and artifact removal

### Autonomous Vehicles
- Road and lane detection
- Pedestrian and vehicle detection
- Traffic sign recognition
- Obstacle identification

### Document Processing
- Optical Character Recognition (OCR)
- Handwriting analysis
- Document layout analysis
- Historical document restoration

### Robotics
- Visual localization and mapping (SLAM)
- Object detection and recognition
- Navigation and path planning
- Gesture recognition

### Retail and E-Commerce
- Product recognition and classification
- Visual search systems
- Quality control and defect detection
- Inventory management

### Surveillance and Security
- Motion detection and tracking
- Activity recognition
- Face detection and recognition
- Anomaly detection

## Advanced Resources

### Further Learning

**Books:**
- "Digital Image Processing" by Gonzalez and Woods
- "Computer Vision: Algorithms and Applications" by Szelenski
- "Deep Learning" by Goodfellow, Bengio, and Courville

**Online Courses:**
- OpenCV tutorials and documentation
- scikit-image examples and tutorials
- Fast.ai computer vision course

**Research Papers:**
- Original SIFT paper: Lowe (2004)
- Harris corner detection: Harris & Stephens (1988)
- Histogram of Oriented Gradients: Dalal & Triggs (2005)

### Modern Alternatives

Modern deep learning approaches complement traditional techniques:
- Convolutional Neural Networks (CNNs) for feature extraction
- Transfer learning from pre-trained models
- End-to-end learning approaches
- Attention mechanisms and transformers

## Troubleshooting and FAQs

### Common Issues

**Import Errors:**
- Ensure all packages are installed correctly
- Check Python version compatibility
- Verify virtual environment activation

**Image Loading Issues:**
- Verify image file paths are correct
- Check image format support
- Ensure adequate disk space

**Performance Issues:**
- Reduce image size for testing
- Use smaller feature sets initially
- Profile code to identify bottlenecks
- Consider GPU acceleration for large operations

### FAQ

**Q: Can I use these labs with Google Colab or other cloud environments?**
A: Yes, notebooks can be adapted for cloud platforms. Install required packages in cloud environment cells.

**Q: What if I don't have a background in computer vision?**
A: The course is designed for self-contained learning. Prerequisites are minimal; all necessary concepts are explained.

**Q: Can I use these labs for teaching or incorporating into courses?**
A: Yes, the materials are designed for educational use. Adapt as needed for your students.

## Support and Contribution

### Getting Help

- Review detailed README files in each laboratory directory
- Consult referenced documentation and academic papers
- Experiment with modified parameters to understand behavior
- Compare with provided reference implementations

### Contributing Improvements

If you discover errors, improvements, or enhancements:
1. Test your changes thoroughly
2. Document modifications clearly
3. Ensure code follows established style
4. Verify backward compatibility

## Course Summary

This comprehensive image processing course provides:

- 12 progressive laboratories covering fundamental to advanced topics
- Complete working implementations with detailed explanations
- Hands-on experience with industry-standard tools
- Real-world applications and context
- Bridge from classical image processing to modern deep learning

The course emphasizes both understanding and practical ability, ensuring students can implement, modify, and extend image processing systems. Upon completion, students will have strong foundations for advanced computer vision work and practical problem-solving abilities.

## License and Usage

These materials are provided for educational purposes. Use, modification, and sharing are encouraged with appropriate attribution.

---

**Course Created**: Image Processing Laboratory Series
**Last Updated**: 2026
**Total Laboratory Hours**: 40-50 hours
**Recommended Group Size**: Individual or small groups
**Assessment Methods**: Hands-on implementation, parameter exploration, extended projects
