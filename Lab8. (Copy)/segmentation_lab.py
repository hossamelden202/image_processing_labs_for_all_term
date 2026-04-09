import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import img_as_float, img_as_ubyte

# ==================== Experiment 1: Adaptive Thresholding ====================

def calculate_histogram(image):
    """
    Calculate histogram of a grayscale image.
    Returns counts array with 256 elements (one for each grey level 0-255)
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = img_as_ubyte(image)
    
    # Initialize counts array
    counts = np.zeros(256, dtype=int)
    
    # Count pixels at each grey level
    for grey_level in range(256):
        counts[grey_level] = np.sum(image == grey_level)
    
    return counts


def getThreshold(image):
    """
    Calculate adaptive threshold using iterative mean-based algorithm.
    
    Args:
        image: Input grayscale image (can be float or uint8)
    
    Returns:
        threshold: The calculated threshold value
    """
    # Step 1: Convert image to uint8
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype('uint8')
    else:
        image_uint8 = image.copy()
    
    # Step 2: Get counts array (histogram)
    counts = calculate_histogram(image_uint8)
    
    # Step 3: Get initial threshold (weighted average of all grey levels)
    grey_levels = np.arange(256)
    total_pixels = np.cumsum(counts)[-1]  # Total number of pixels
    
    # Calculate weighted mean: sum(grey_level * count) / total_pixels
    Tinit = round(np.sum(grey_levels * counts) / total_pixels)
    
    threshold = Tinit
    print(f"Initial threshold: {Tinit}")
    
    # Iterative refinement
    iteration = 0
    max_iterations = 100  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        
        # Step 4: Calculate two weighted averages
        # Lower pixels (grey level < threshold)
        lower_range = list(range(0, threshold))
        lower_counts = counts[lower_range]
        total_lower_pixels = np.sum(lower_counts)
        
        if total_lower_pixels > 0:
            mean_lower = round(np.sum(np.array(lower_range) * lower_counts) / total_lower_pixels)
        else:
            mean_lower = 0
        
        # Higher pixels (grey level >= threshold)
        higher_range = list(range(threshold, 256))
        higher_counts = counts[higher_range]
        total_higher_pixels = np.sum(higher_counts)
        
        if total_higher_pixels > 0:
            mean_higher = round(np.sum(np.array(higher_range) * higher_counts) / total_higher_pixels)
        else:
            mean_higher = 255
        
        # Step 5: Update threshold (average of two means)
        new_threshold = round((mean_lower + mean_higher) / 2)
        
        # Step 6: Check for convergence
        if new_threshold == threshold:
            print(f"Converged after {iteration} iterations at threshold: {threshold}")
            break
        
        threshold = new_threshold
    
    return threshold


def apply_threshold(image, threshold):
    """
    Apply threshold to image: pixels < threshold = 0, pixels >= threshold = 255
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype('uint8')
    else:
        image_uint8 = image.copy()
    
    # Apply threshold
    binary_image = np.where(image_uint8 < threshold, 0, 255).astype('uint8')
    
    return binary_image


# ==================== Experiment 2: Local Adaptive Thresholding ====================

def local_adaptive_threshold(image):
    """
    Apply local adaptive thresholding by dividing image into 4 quarters.
    
    Args:
        image: Input grayscale image
    
    Returns:
        result: Image with local thresholds applied
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype('uint8')
    else:
        image_uint8 = image.copy()
    
    height, width = image_uint8.shape
    mid_h = height // 2
    mid_w = width // 2
    
    # Create result image
    result = np.zeros_like(image_uint8)
    
    # Process each quarter
    quarters = [
        (0, mid_h, 0, mid_w, "Top-Left"),
        (0, mid_h, mid_w, width, "Top-Right"),
        (mid_h, height, 0, mid_w, "Bottom-Left"),
        (mid_h, height, mid_w, width, "Bottom-Right")
    ]
    
    for h_start, h_end, w_start, w_end, name in quarters:
        quarter = image_uint8[h_start:h_end, w_start:w_end]
        print(f"\n{name} Quarter:")
        threshold = getThreshold(quarter)
        result[h_start:h_end, w_start:w_end] = apply_threshold(quarter, threshold)
    
    return result


# ==================== Main Execution ====================

def process_image(image_path, apply_local=False):
    """
    Process a single image with global and optionally local thresholding.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    image = io.imread(image_path)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Ensure float format for processing
    image = img_as_float(image)
    
    # Calculate global threshold
    print("\nGlobal Thresholding:")
    global_threshold = getThreshold(image)
    global_result = apply_threshold(image, global_threshold)
    
    # Create figure
    if apply_local:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(global_result, cmap='gray')
        axes[1].set_title(f'Global Threshold (T={global_threshold})')
        axes[1].axis('off')
        
        # Apply local thresholding
        print("\n" + "="*60)
        print("Local Adaptive Thresholding:")
        print("="*60)
        local_result = local_adaptive_threshold(image)
        
        axes[2].imshow(local_result, cmap='gray')
        axes[2].set_title('Local Adaptive Threshold')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(global_result, cmap='gray')
        axes[1].set_title(f'Global Threshold (T={global_threshold})')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"output_{image_path.split('/')[-1]}", dpi=150, bbox_inches='tight')
    plt.show()


# ==================== Run Experiments ====================

if __name__ == "__main__":
    # Experiment 1: Test on multiple images
    print("\n" + "="*60)
    print("EXPERIMENT 1: Global Adaptive Thresholding")
    print("="*60)
    
    test_images = ["images/cameraman.png", "images/cufe.png", 
                   "images/book1.png", "images/book.png"]
    
    for img_path in test_images:
        try:
            process_image(img_path, apply_local=False)
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Experiment 2: Local adaptive thresholding on book.png
    print("\n" + "="*60)
    print("EXPERIMENT 2: Local Adaptive Thresholding on book.png")
    print("="*60)
    
    try:
        process_image("images/book.png", apply_local=True)
    except FileNotFoundError:
        print("Image not found: images/book.png")
    except Exception as e:
        print(f"Error processing book.png: {e}")
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)