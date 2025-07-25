print("Hello, World!")

from PIL import Image
from lang_sam import LangSAM

def save_image_with_masks(image, masks, output_path="output.png"):
    """
    Saves the original image and its masks to a file.
    
    Args:
        image: The original image
        masks: List of masks
        output_path: Path where the image should be saved (default: 'output.png')
    """
    import matplotlib.pyplot as plt
    import os

    num_masks = len(masks)
    
    # Create figure
    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    
    # Handle case where there's only one subplot
    if num_masks == 0:
        axes = [axes]
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Plot masks
    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"Image saved to {output_path}")

# Example usage:
# save_image_with_masks(image, masks, "path/to/output/image.png")


model = LangSAM()

image_pil = Image.open("./images/car.jpg").convert("RGB")
text_prompt = "wheel."
result = model.predict([image_pil], [text_prompt]) # masks, boxes, phrases, logits
print(result)
print(len(result))

save_image_with_masks(image_pil, result[0]['masks'], "images/output.png")