from PIL import Image, ImageEnhance
import numpy as np

# Load the image
image_path = "your_image.png"  # Replace with your image path
image = Image.open(image_path).convert("L")  # Convert to grayscale

# Convert the image to a NumPy array for manipulation
image_array = np.array(image)

# Enhance contrast for gray areas (excluding white background)
# Assuming the white background is close to 255
threshold = 240  # Adjust this threshold if needed
contrast_factor = 1.3  # Adjust this factor to increase/decrease contrast

# Apply contrast enhancement only to non-white areas
gray_mask = image_array < threshold
mean_value = image_array[gray_mask].mean()  # Calculate mean of gray areas
enhanced_array = image_array.copy()
enhanced_array[gray_mask] = np.clip(
    mean_value + contrast_factor * (image_array[gray_mask] - mean_value), 0, 255
)

# Convert back to an image
enhanced_image = Image.fromarray(enhanced_array.astype("uint8"))

# Save the result
output_path = "enhanced_image.png"  # Replace with desired output path
enhanced_image.save(output_path)

# Show the result
# enhanced_image.show()
