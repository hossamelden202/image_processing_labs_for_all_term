"""
Experiment 3 (Bonus): Improved Thresholding Techniques

This module provides enhanced thresholding methods that improve upon
the basic adaptive thresholding algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology
from skimage.util import img_as_float, img_as_ubyte
from scipy import ndimage

# Import functions from main lab
from segmentation_lab import getThreshold, apply_threshold


def improved_local_threshold_overlapping(image, block_size=64, overlap=0.5):
    """
    Improved local thresholding with overlapping blocks and weighted averaging.
    
    Justification:
    - Overlapping blocks reduce boundary artifacts
    - Weighted averaging creates smooth transitions between regions
    - Smaller blocks adapt better to local variations
    
    Args:
        image: Input grayscale image
        block_size: Size of each block (default: 64)
        overlap: Overlap ratio between blocks (0-1, default: 0.5)
    
    Returns:
        Binary image with improved local thresholding
    """
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype('uint8')
    else:
        image_uint8 = image.copy()
    
    height, width = image_uint8.shape
    step_size = int(block_size * (1 - overlap))
    
    # Create weight matrix for blending
    weight_matrix = np.zeros((height, width), dtype=float)
    result_sum = np.zeros((height, width), dtype=float)
    
    print(f"Processing with block_size={block_size}, overlap={overlap}")
    
    block_count = 0
    for i in range(0, height - block_size + 1, step_size):
        for j in range(0, width - block_size + 1, step_size):
            # Extract block
            block = image_uint8[i:i+block_size, j:j+block_size]
            
            # Calculate threshold for this block
            threshold = getThreshold(block)
            binary_block = apply_threshold(block, threshold)
            
            # Create weight for this block (Gaussian-like)
            y, x = np.ogrid[0:block_size, 0:block_size]
            center_y, center_x = block_size // 2, block_size // 2
            weight = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (block_size/4)**2))
            
            # Accumulate results
            result_sum[i:i+block_size, j:j+block_size] += binary_block * weight
            weight_matrix[i:i+block_size, j:j+block_size] += weight
            
            block_count += 1
    
    print(f"Processed {block_count} overlapping blocks")
    
    # Normalize by weights
    result = np.zeros_like(image_uint8)
    valid_mask = weight_matrix > 0
    result[valid_mask] = (result_sum[valid_mask] / weight_matrix[valid_mask]).astype('uint8')
    
    # Apply final threshold to ensure binary output
    result = np.where(result >= 128, 255, 0).astype('uint8')
    
    return result


def morphological_improvement(binary_image, operation='close', kernel_size=3):
    """
    Apply morphological operations to improve binary image quality.
    
    Justification:
    - Removes small noise (opening)
    - Fills small holes (closing)
    - Smooths boundaries
    
    Args:
        binary_image: Binary input image
        operation: 'open', 'close', or 'both'
        kernel_size: Size of structuring element
    
    Returns:
        Improved binary image
    """
    kernel = morphology.disk(kernel_size)
    
    if operation == 'open':
        result = morphology.opening(binary_image, kernel)
    elif operation == 'close':
        result = morphology.closing(binary_image, kernel)
    elif operation == 'both':
        # Apply opening then closing
        result = morphology.opening(binary_image, kernel)
        result = morphology.closing(result, kernel)
    else:
        result = binary_image
    
    return result


def adaptive_gaussian_threshold(image, block_size=35, C=2):
    """
    Adaptive thresholding using local Gaussian-weighted mean.
    
    Justification:
    - Adapts to local illumination variations
    - Gaussian weighting reduces sensitivity to noise
    - Better for documents with uneven lighting
    
    Args:
        image: Input grayscale image
        block_size: Size of neighborhood (must be odd)
        C: Constant subtracted from mean
    
    Returns:
        Binary image
    """
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype('uint8')
    else:
        image_uint8 = image.copy()
    
    # Apply Gaussian filter to get local mean
    local_mean = ndimage.gaussian_filter(image_uint8.astype(float), 
                                         sigma=block_size/6)
    
    # Threshold: pixel > (local_mean - C)
    result = np.where(image_uint8 > (local_mean - C), 255, 0).astype('uint8')
    
    return result


def otsu_threshold(image):
    """
    Otsu's automatic thresholding method.
    
    Justification:
    - Optimal threshold for bimodal histograms
    - No parameters needed
    - Maximizes between-class variance
    
    Args:
        image: Input grayscale image
    
    Returns:
        threshold_value: Calculated threshold
        binary_image: Thresholded image
    """
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype('uint8')
    else:
        image_uint8 = image.copy()
    
    # Use scikit-image's Otsu implementation
    threshold_value = filters.threshold_otsu(image_uint8)
    binary_image = np.where(image_uint8 >= threshold_value, 255, 0).astype('uint8')
    
    return threshold_value, binary_image


def compare_methods(image_path):
    """
    Compare multiple thresholding methods on a single image.
    """
    print(f"\n{'='*60}")
    print(f"Comparing Methods: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    image = io.imread(image_path)
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    image = img_as_float(image)
    
    # Method 1: Basic global threshold
    print("\n1. Basic Global Threshold:")
    t1 = getThreshold(image)
    result1 = apply_threshold(image, t1)
    
    # Method 2: Basic local threshold (4 quarters)
    print("\n2. Basic Local Threshold (4 quarters):")
    from segmentation_lab import local_adaptive_threshold
    result2 = local_adaptive_threshold(image)
    
    # Method 3: Improved overlapping blocks
    print("\n3. Overlapping Blocks:")
    result3 = improved_local_threshold_overlapping(image, block_size=64, overlap=0.5)
    
    # Method 4: Otsu's method
    print("\n4. Otsu's Method:")
    t4, result4 = otsu_threshold(image)
    print(f"Otsu threshold: {t4}")
    
    # Method 5: Adaptive Gaussian
    print("\n5. Adaptive Gaussian:")
    result5 = adaptive_gaussian_threshold(image, block_size=35, C=2)
    
    # Method 6: Morphological improvement on method 3
    print("\n6. Overlapping + Morphology:")
    result6 = morphological_improvement(result3, operation='both', kernel_size=2)
    
    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(result1, cmap='gray')
    axes[0, 1].set_title(f'Global (T={t1})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result2, cmap='gray')
    axes[0, 2].set_title('Local (4 quarters)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(result3, cmap='gray')
    axes[0, 3].set_title('Overlapping Blocks')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(result4, cmap='gray')
    axes[1, 0].set_title(f'Otsu (T={t4})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(result5, cmap='gray')
    axes[1, 1].set_title('Adaptive Gaussian')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(result6, cmap='gray')
    axes[1, 2].set_title('Overlap + Morphology')
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"comparison_{image_path.split('/')[-1]}", dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'global': result1,
        'local_4': result2,
        'overlapping': result3,
        'otsu': result4,
        'gaussian': result5,
        'morphology': result6
    }


# ==================== Main Execution ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXPERIMENT 3 (BONUS): Improved Thresholding Methods")
    print("="*60)
    
    # Test on book.png (most challenging)
    try:
        results = compare_methods("images/book.png")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        print("""
        Best methods for different scenarios:
        
        1. DOCUMENTS with uneven lighting (like book.png):
           → Overlapping Blocks or Adaptive Gaussian
           → These methods adapt to local illumination
        
        2. CLEAN images with uniform lighting:
           → Global Threshold or Otsu's Method
           → Faster and simpler
        
        3. NOISY images:
           → Add Morphological operations (closing/opening)
           → Removes small artifacts
        
        4. GENERAL purpose:
           → Overlapping Blocks + Morphology
           → Best balance of quality and robustness
        """)
        
    except FileNotFoundError:
        print("Image not found: images/book.png")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Bonus experiment completed!")
    print("="*60)