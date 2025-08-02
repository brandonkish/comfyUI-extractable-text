import os
import hashlib
from datetime import datetime
import json
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION
import base64
import json
import math
import numpy as np
import torch
import random
from nodes import SaveImage
try:
    import piexif.helper
    import piexif
    from .exif.exif import read_info_from_image_stealth
    piexif_loaded = True
except ImportError:
    piexif_loaded = False



def make_filename(filename):
    return "image" if filename == "" else filename

# Utility function for handling whitespace
def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

# Node class definition
class SaveImageWithDescription:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": f'image', "multiline": False}),
                "path": ("STRING", {"default": f'', "multiline": False}),
                "include_workflow": ("BOOLEAN", {"default": True, "tooltip": "If true will save a copy of the workflow into the PNG, Else will not."}),
                "description": ("STRING", {"multiline": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("STRING","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("folderpath", "filepath")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Descriptive Images"  # A category for the node, adjust as needed
    LABEL = "Save Image With Description"  # Default label text
    OUTPUT_NODE = True


    
    def process(self, images, description, path, filename, include_workflow, extra_pnginfo=None):
        # Ensure the input is treated as text
        output_path = os.path.join(self.output_dir, path)


        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)  

        self.save_images(images, output_path, filename, description, extra_pnginfo, include_workflow)
        return (output_path.strip(),output_path.strip() + "\\" + filename.strip() + ".png")


    def save_images(self, images, output_path, filename_prefix, description, extra_pnginfo, include_workflow) -> list[str]:
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

            if include_workflow is True:
                if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            
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
                "filename": ("STRING", {"default": f'image', "multiline": False}),
                "path": ("STRING", {"default": f'', "multiline": False}),
                "include_workflow": ("BOOLEAN", {"default": True, "tooltip": "If true will save a copy of the workflow into the PNG, Else will not."}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("STRING","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("folderpath", "filepath")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Descriptive Images"  # A category for the node, adjust as needed
    LABEL = "Save Image To Folder"  # Default label text
    OUTPUT_NODE = True

    
    def process(self, images, path, filename, include_workflow, extra_pnginfo=None):
        # Ensure the input is treated as text
        output_path = os.path.join(self.output_dir, path)
        self.output_path = output_path

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)  

        self.save_images(images, output_path, filename,extra_pnginfo, include_workflow)
        return (output_path.strip(),output_path.strip() + "\\" + filename.strip() + ".png")


    def save_images(self, images, output_path, filename_prefix,extra_pnginfo, include_workflow) -> list[str]:
        img_count = 1
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if images.size()[0] > 1:
                filename_prefix += "_{:02d}".format(img_count)

            filename = f"{filename_prefix}.png"
            metadata = PngInfo()

            if include_workflow is True:
                if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            img.save(os.path.join(output_path, filename), pnginfo=metadata, optimize=True)
            img_count += 1


# Node class definition
class LoadImageWithDescription:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required":
                    {"image": (sorted(files), {"image_upload": True})},
        }

    
    RETURN_TYPES = ("IMAGE","STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("image","name","description")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Descriptive Images"  # A category for the node, adjust as needed
    LABEL = "Load Image With Description"  # Default label text
    OUTPUT_NODE = True


    
    def process(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        with open(image_path,'rb') as file:
            img = Image.open(file)
            extension = image_path.split('.')[-1]
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

        parameters = ""
        if extension.lower() == 'png':
            try:
                parameters = img.info['description']
            except:
                parameters = ""
                print("WARN: No description found in PNG")
        return(image, image_name, parameters)
               
# Node class definition
class LoadImageWithDescriptionByPath:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        return {
            "required":
                    {
                        "image_name": ("STRING", {"default": f'', "multiline": False}),
                        "folder": ("STRING", {"default": f'', "multiline": False}),
                    },
        }

    
    RETURN_TYPES = ("STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("description","name",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Descriptive Images"  # A category for the node, adjust as needed
    LABEL = "Load Image With Description"  # Default label text
    OUTPUT_NODE = True


    
    def process(self, folder, image_name):
        parameters = ""
        if folder and image_name:
            image_path = os.path.join(self.output_dir, folder, f"{image_name}.png")
            with open(image_path, 'rb') as file:
                img = Image.open(file)
                img.load()  # Force loading image and metadata before file is closed
                extension = image_path.split('.')[-1]
                
                if extension.lower() == 'png':
                    try:
                        parameters = img.info.get('description', '')
                    except Exception as e:
                        parameters = ""
                        print("WARN: No description found in PNG:", e)

                img.close()  # Explicitly release resources
        else:
            folder = ""
            image_name = ""

        return (image_name, parameters)

class AIVisionPreview:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "description": ("STRING", {"default": "image", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "description",)
    FUNCTION = "process"
    CATEGORY = "Descriptive Images"
    LABEL = "AI Vision Preview"
    OUTPUT_NODE = True

    def process(self, images, description, prompt=None, extra_pnginfo=None):
        image_preview = [{
            "image": images[0],
            "type": self.type,
            "filename": f"{description}.png",
            "subfolder": "AIVisionPreview",
        }]
        return (
            images,
            description,
            {"ui": {"images": image_preview}}
        )

  

# Register the node in ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Save Image With Description": SaveImageWithDescription,  # The name that will show in the UI
    "Save Image To Folder": SaveImgToFolder,  # The name that will show in the UI
    "Load Image With Description": LoadImageWithDescription,  # The name that will show in the UI
    "AI Vision Preview": AIVisionPreview,
    "Load Image With Description By Path (Under Construction)": LoadImageWithDescriptionByPath,
}
