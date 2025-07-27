import os
import json
from PIL import Image
import comfy.sd  # Import any necessary modules from ComfyUI
from nodes import MAX_RESOLUTION  # Assuming this constant is defined elsewhere

# Utility function for handling whitespace
def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

# Node class definition
class MyCustomNode:
    RETURN_TYPES = ()  # This can be customized based on your node's return types
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "MyCustomCategory"  # A category for the node, adjust as needed

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts
        return {
            "required": {
                "input_image": ("IMAGE", ),  # Adjust as per your needs
                "filename": ("STRING", {"default": "output_image", "multiline": False}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100}),
            },
        }

    OUTPUT_NODE = True  # If this is an output node (e.g., saves or returns images)
    
    def process(self, input_image, filename, quality):
        # Implement the logic to process your input image
        output_path = f"/path/to/save/{filename}.png"
        
        # Convert image and save (this is just a simple example)
        img = Image.fromarray(input_image)
        img.save(output_path, quality=quality)

        # You can return the output path or other details
        return {"output": output_path}

NODE_CLASS_MAPPINGS = {
    "Extractable Text Node": MyCustomNode,  # Replace "My Custom Node" with the name you want to show in ComfyUI
}