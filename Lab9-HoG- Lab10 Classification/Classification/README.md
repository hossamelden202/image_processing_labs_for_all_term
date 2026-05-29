# Lab 10: Image Classification using Machine Learning

## Overview

Lab 10 addresses the fundamental problem of image classification: automatically assigning category labels to images based on visual content. This laboratory explores multiple feature extraction approaches and machine learning classifiers to solve the practical problem of handwritten digit recognition. Students will learn to extract different types of features from images, prepare data for machine learning algorithms, and evaluate classification systems.

The classification pipeline bridges image processing and machine learning, demonstrating how computer vision features can be transformed into high-level semantic understanding. The laboratory emphasizes practical considerations including feature normalization, train-test splitting, and classifier selection for optimal performance.

## Detailed Description

### Image Classification Problem

#### Task Definition

The core challenge addressed in this laboratory: Given an image of a handwritten digit, automatically determine which digit (0-9) it represents.

**Problem Characteristics:**
- **Multi-class Classification**: 10 possible output categories (digits 0-9)
- **Supervised Learning**: Training data includes correct labels
- **Visual Recognition**: Must learn from visual patterns
- **Practical Application**: Forms the basis for postal code reading, document analysis, and OCR systems

#### Dataset Organization

The digits dataset contains:
- Organized folder structure with images grouped by digit class
- Fixed image size (standardized to 32×32 pixels)
- Black digits on white background
- Training and testing image separation
- Natural variations in handwriting style

### Feature Extraction Strategies

The laboratory implements and compares three fundamentally different feature extraction approaches:

#### 1. HSV Histogram Features

Histogram-based features capture color information and intensity distributions:

**Color Space Conversion:**
- Convert BGR image (OpenCV default) to HSV color space
- HSV separates color information (Hue, Saturation) from brightness (Value)
- More perceptually relevant than RGB for many applications

**Feature Extraction Process:**

```python
def extract_hsv_histogram(img):
    # Resize to fixed dimensions
    img = cv2.resize(img, target_img_size)
    
    # Convert color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Compute 3D histogram
    hist = cv2.calcHist(img, [0, 1, 2], None, 
                        histSize=[8, 8, 8], 
                        ranges=[0, 180, 0, 256, 0, 256])
    
    # Normalize for uniform scale
    cv2.normalize(hist, hist)
    
    # Flatten to 1D vector
    return hist.flatten()
```

**Feature Characteristics:**
- **Dimensionality**: 8×8×8 = 512 bins
- **Information**: Color distribution patterns
- **Advantages**: Robust to small geometric changes, rotation invariant
- **Disadvantages**: Loses spatial information, treats image as unordered color collection

#### 2. Histogram of Oriented Gradients (HOG)

HOG features capture edge orientations and local structure:

**Feature Extraction Configuration:**

```python
def extract_hog_features(img):
    # Resize to fixed dimensions
    img = cv2.resize(img, target_img_size)
    
    # HOG parameter configuration
    win_size = (32, 32)           # Window/image size
    cell_size = (4, 4)             # Cell size in pixels
    block_size_in_cells = (2, 2)   # Block size in cells
    block_size = (block_size_in_cells[1] * cell_size[1], 
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9                       # Orientation bins
    
    # Compute HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                            cell_size, nbins)
    h = hog.compute(img)
    
    return h.flatten()
```

**Feature Characteristics:**
- **Dimensionality**: 256-1024 features (depends on configuration)
- **Information**: Edge directions and local structure
- **Advantages**: Captures shape and form, robust to illumination
- **Disadvantages**: Loses color information, computationally more complex

#### 3. Raw Pixel Features

The simplest approach: using pixel values directly as features:

```python
def extract_raw_pixels(img):
    # Resize to fixed dimensions
    img = cv2.resize(img, target_img_size)
    
    # Flatten 2D image to 1D vector
    return img.flatten()
```

**Feature Characteristics:**
- **Dimensionality**: 32×32 = 1024 features
- **Information**: Raw intensity values
- **Advantages**: Simple, preserves all information, no feature engineering
- **Disadvantages**: High dimensional, sensitive to translation and rotation, requires regularization

### Machine Learning Classifiers

The laboratory explores three classifier types:

#### 1. K-Nearest Neighbors (KNN)

A non-parametric, lazy learning algorithm:

**Algorithm Overview:**
- For each test image, find K nearest training images in feature space
- Assign the test image to the majority class among these K neighbors
- Parameter K controls complexity (larger K = smoother decision boundary)

**Characteristics:**
- **Advantages**: Simple, no training phase, effective baseline
- **Disadvantages**: Slow at test time, sensitive to feature scaling, requires entire training set in memory

#### 2. Support Vector Machine (SVM)

A powerful linear classifier using kernel tricks:

**Algorithm Overview:**
- Finds optimal hyperplane separating classes in feature space
- Kernel tricks enable non-linear boundaries
- Margin-based training minimizes generalization error

**Characteristics:**
- **Advantages**: Robust to high dimensions, effective with various kernel choices
- **Disadvantages**: Training can be slow on large datasets, parameter tuning required

#### 3. Multilayer Perceptron (Neural Network)

An artificial neural network with multiple layers:

**Architecture:**
- Input layer: Feature vector
- Hidden layers: Non-linear transformations
- Output layer: Class probability estimates

**Characteristics:**
- **Advantages**: Can learn complex patterns, flexible architecture
- **Disadvantages**: Requires more parameters, prone to overfitting, slower training

### Classification Pipeline

**Step 1: Data Preparation**
- Load images from dataset directory
- Extract features using chosen method
- Organize into feature matrix (n_samples × n_features)
- Prepare label vector

**Step 2: Data Splitting**
- Divide into training (70-80%) and testing (20-30%) sets
- Maintain random seed for reproducibility
- Ensure balanced class distribution in both sets

**Step 3: Feature Normalization**
- Scale features to zero mean and unit variance
- Prevents high-magnitude features from dominating
- Essential for distance-based classifiers (KNN)
- Important for regularized methods (SVM, neural networks)

**Step 4: Model Training**
- Fit classifier to training data
- Learn decision boundaries
- Hyperparameter tuning using validation data

**Step 5: Evaluation**
- Predict on test set
- Compute accuracy: proportion of correct predictions
- Analyze error patterns and confusion matrices
- Compare different feature-classifier combinations

## Key Learning Objectives

Upon successful completion of this laboratory, students will:

1. Understand the complete image classification pipeline
2. Implement multiple feature extraction strategies
3. Recognize advantages and limitations of different features
4. Apply machine learning classifiers to visual recognition
5. Prepare image data for classification algorithms
6. Normalize and scale features appropriately
7. Split data into training and testing sets
8. Train and evaluate classification models
9. Interpret and analyze classification results
10. Compare different feature and classifier combinations
11. Understand overfitting and generalization concepts
12. Apply classification systems to real-world problems

## Technical Implementation Details

### Feature Extraction Wrapper

```python
def extract_features(img, feature_set='hog'):
    """Extract features based on specified method"""
    if feature_set == 'hsv':
        return extract_hsv_histogram(img)
    elif feature_set == 'hog':
        return extract_hog_features(img)
    elif feature_set == 'raw':
        return extract_raw_pixels(img)
    else:
        raise ValueError("Unknown feature set")
```

### Dataset Loading

```python
def load_dataset(dataset_path):
    """Load images and labels from organized directory structure"""
    images = []
    labels = []
    
    for digit_class in range(10):
        class_dir = os.path.join(dataset_path, str(digit_class))
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png') or img_file.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    images.append(img)
                    labels.append(digit_class)
    
    return np.array(images), np.array(labels)
```

### Classification and Evaluation

```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Feature normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_scaled, y_train)

# Evaluate
accuracy = classifier.score(X_test_scaled, y_test)
predictions = classifier.predict(X_test_scaled)
```

## How to Run the Laboratory

### Prerequisites

Install required Python packages:

```bash
pip install numpy scikit-learn opencv-python matplotlib scipy scikit-image
```

### Execution Steps

1. Navigate to the classification lab directory:
```bash
cd "Lab9-HoG- Lab10 Classification/Classification"
```

2. Open the Jupyter notebook:
```bash
jupyter notebook classification-STD.ipynb
```

3. Execute cells in sequence:
   - Set dataset path and parameters
   - Load digit images from dataset
   - Implement or review feature extraction functions
   - Extract features using different methods
   - Split data into training and testing sets
   - Train classifiers (KNN, SVM, Neural Network)
   - Evaluate accuracy and performance
   - Compare different feature-classifier combinations
   - Visualize results and error analysis

4. Experimentation:
   - Try different numbers of features
   - Modify classifier parameters
   - Test with different random seeds
   - Analyze which digit classes are confused
   - Compare feature extraction methods

5. Extension activities:
   - Implement cross-validation for parameter tuning
   - Analyze feature importance
   - Visualize learned decision boundaries
   - Test on additional datasets

### Expected Outputs

- Feature vectors extracted from images
- Train-test split statistics
- Classification accuracies for different methods
- Confusion matrices showing classification errors
- Comparison tables of feature-classifier performance
- Analysis of which digits are most frequently confused
- Insights about feature effectiveness

## Advanced Topics and Extensions

### Cross-Validation

Improved evaluation using multiple train-test splits:
- k-fold cross-validation
- Stratified splitting for balanced classes
- More robust accuracy estimates

### Hyperparameter Optimization

Systematic parameter tuning:
- Grid search over parameter combinations
- Random search for efficiency
- Bayesian optimization for complex spaces

### Feature Selection

Choosing most discriminative features:
- Removing redundant features
- Reducing computational cost
- Improving interpretability
- Preventing overfitting

### Ensemble Methods

Combining multiple classifiers:
- Voting ensembles
- Boosting (AdaBoost)
- Bagging (Random Forest)
- Gradient Boosting

### Deep Learning Approaches

Neural network-based solutions:
- Convolutional Neural Networks (CNNs)
- Transfer learning from pre-trained models
- End-to-end feature learning

## Real-World Applications

### Document Analysis
- Postal code reading
- Cheque amount recognition
- Form field recognition

### Accessibility Systems
- Converting handwritten notes to digital text
- Document scanning and digitization
- Real-time OCR for assistive technologies

### Security and Verification
- Signature verification
- Document authentication
- Identity verification

### Data Entry Automation
- Invoice digitization
- Medical record processing
- Historical document analysis

## Laboratory Files

- `classification-STD.ipynb`: Main notebook with classification implementations
- `digits_dataset/`: Directory containing organized digit images
  - `0/` to `9/`: Subdirectories for each digit class
- `NOTES.txt`: Additional documentation and instructions
- `TO-INSTALL.txt`: Package installation requirements

## References and Resources

- scikit-learn documentation: https://scikit-learn.org
- OpenCV image processing: https://opencv.org
- NumPy and SciPy documentation
- Machine learning fundamentals
- Image classification best practices

## Important Notes

- Feature normalization is critical for classifier performance
- Random seed should be fixed for reproducible results
- Train-test split must be done before any preprocessing
- Different features and classifiers suit different problems
- No single method is optimal for all scenarios
- Document parameter choices and results for comparability
- Always evaluate on a held-out test set
- Analyze errors to understand classifier behavior
- Consider computational cost for real-time applications
- Combine multiple features or classifiers for improved robustness
