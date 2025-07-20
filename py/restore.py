from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

def restore_and_display_images(json_file):
    
    with open(json_file, "r") as f:
        image_data = json.load(f)
    
    fig, axes = plt.subplots(5, 5, figsize=(10, 16)) 
    axes = axes.flatten()
    
    for idx, (url, image_array) in enumerate(image_data.items()):
        image_array = np.array(image_array, dtype=np.uint8)
        
        restored_image = Image.fromarray(image_array)
        
        axes[idx].imshow(restored_image, cmap='gray')
        axes[idx].set_title(url)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

restore_and_display_images("map_image.json")