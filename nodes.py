import os
import hashlib
from datetime import datetime
import json
import piexif
import piexif.helper
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION



def make_filename(filename):
    return "image" if filename == "" else filename

# Utility function for handling whitespace
def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

# Node class definition
class ExtractableTextNode:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": f'image', "multiline": False}),
                "path": ("STRING", {"default": f'', "multiline": False}),
                "description": ("STRING", {"multiline": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("fullpath",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Extractable Nodes"  # A category for the node, adjust as needed
    LABEL = "Extractable Text Node"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, images, description, path, filename_prefix):
        # Force re-evaluation of the node
            return float("NaN")

    
    def process(self, images, description, path, filename_prefix):
        # Ensure the input is treated as text
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)  

        self.save_images(images, output_path, filename_prefix, description)
        return(output_path.strip(),)


    def save_images(self, images, output_path, filename_prefix, description) -> list[str]:
        img_count = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if images.size()[0] > 1:
                filename_prefix += "_{:02d}".format(img_count)

            filename = f"{filename_prefix}.png"
            metadata = PngInfo()
            metadata.add_text("description", description)
            img.save(os.path.join(output_path, filename), pnginfo=metadata, optimize=True)
            paths.append(filename)
            img_count += 1


# Node class definition
class SaveImgToFolder:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.output_path = ""
        self.file_path = ""

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": f'image', "multiline": False}),
                "path": ("STRING", {"default": f'', "multiline": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("fullpath",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Extractable Nodes"  # A category for the node, adjust as needed
    LABEL = "Save Image To Folder"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, images, description, path, filename_prefix):
        # Force re-evaluation of the node
            return float("NaN")

    
    def process(self, images, path, filename_prefix):
        # Ensure the input is treated as text
        output_path = os.path.join(self.output_dir, path)
        self.output_path = output_path

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)  

        self.save_images(images, output_path, filename_prefix)
        return (output_path.strip(),)


    def save_images(self, images, output_path, filename_prefix) -> list[str]:
        img_count = 1
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if images.size()[0] > 1:
                filename_prefix += "_{:02d}".format(img_count)

            filename = f"{filename_prefix}.png"
            metadata = PngInfo()
            img.save(os.path.join(output_path, filename), pnginfo=metadata, optimize=True)
            img_count += 1

# Register the node in ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Extractable Text Node": ExtractableTextNode,  # The name that will show in the UI
    "Save Image To Folder": SaveImgToFolder,  # The name that will show in the UI
}
