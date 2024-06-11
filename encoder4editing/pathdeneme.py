import os
image_path = '/home/izma/Stylemc-Fork/uploads/adam.png'
print(f"Trying to open image from path: {image_path}")
if not os.path.exists(image_path):
    print(f"Error: The image at path {image_path} does not exist.")