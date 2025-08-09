import os
import hashlib
from datetime import datetime
import json
import torch
from PIL import Image, ImageOps, ImageSequence, ExifTags, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION
import base64
import math
import random
from nodes import SaveImage
import cv2
import comfy.utils
import re
import node_helpers
from pathlib import Path

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


class JSONExtractor:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "ollama_connectivity": ("OLLAMA_CONNECTIVITY", {
                "multiline": False,
                "default": ""}),
            }}

    RETURN_TYPES = ("STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("model","url","keep_alive","keep_alive_unit")

    FUNCTION = "process_text"
    CATEGORY = "BKNodes"
    LABEL = "Ollama Connectivity Data"  # Default label text



    def process_text(self, ollama_connectivity):
        model = ''
        url = ''
        keep_alive = ''
        keep_alive_unit = ''

        try:
            # Step 1: Convert Python object to a JSON string
            json_string = json.dumps(ollama_connectivity)

            # Step 2: Convert JSON string back to Python dictionary
            data = json.loads(json_string)

            # Step 3: Search for the key in the dictionary

            model = json_value(data, "model")
            url = json_value(data, "url")
            keep_alive = json_value(data, "keep_alive")
            keep_alive_unit = json_value(data, "keep_alive_unit")
        except json.JSONDecodeError:
            pass

        
        # Step 4: Ensure we return the result properly as a full string
        return (model, url, keep_alive, keep_alive_unit)
    
def json_value(json_obj, name):
    result = "NotFound"
    if name in json_obj:
        result = json_obj[name]
    return result
    
# Node class definition
class SaveImageEasy:
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
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("folderpath", "filepath",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "Save Image Easy"  # Default label text
    OUTPUT_NODE = True


    
    def process(self, images, path, filename, include_workflow, extra_pnginfo=None):
        # Ensure the input is treated as text
        output_path = os.path.join(self.output_dir, path)


        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)  

        self.save_images(images, output_path, filename, None, extra_pnginfo, include_workflow)
        return (output_path.strip(),output_path.strip() + "\\" + filename.strip() + ".png")


    def save_images(self, images, output_path, filename_prefix, extra_pnginfo, include_workflow) -> list[str]:
        img_count = 1
        paths = list()
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
            paths.append(filename)
            img_count += 1


# Node class definition
class CreateOverwriteTxtFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text": ("STRING", {"default": f'', "multiline": False}),
                "path": ("STRING", {"default": f'', "multiline": False}),
                "filename": ("STRING", {"default": f'file', "multiline": False}),
                "extension":  ("STRING", {"default": f'.txt', "multiline": False}),
            },
        }
    
    RETURN_TYPES = ("STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("folderpath", "text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "Save\Overwrite Text File"  # Default label text
    OUTPUT_NODE = True


    
    def process(self, text, path, filename, extension):
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)  

        filepath = combine_path(output_path.strip(), filename.strip(), extension)
    # Open the file in write mode ('w'), which will overwrite the file if it exists
        with open(filepath, 'w') as file:
            file.write(text.strip())
        print(f"Content written to {filepath}")
        return (output_path, text)



def combine_path(folder_path, file_name, extension):
    # Ensure the folder path ends without a trailing separator
    folder_path = Path(folder_path).resolve()
    
    # Clean up the extension to ensure it only has one leading period
    if extension.startswith('.'):
        extension = extension[1:]  # Remove the leading period if it exists
    # Combine the folder path with the file name and the cleaned extension
    full_path = folder_path / f"{file_name}.{extension}"
    
    return str(full_path)

# Node class definition
class OverwriteImage:
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
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
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
class LoadImageWithPath:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required":
                    {
                        "image": (sorted(files), {"image_upload": True})
                    },
        }

    
    RETURN_TYPES = ("IMAGE","MASK","STRING", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("image","mask","name", "folder_path")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "Load Image With Path"  # Default label text
    OUTPUT_NODE = True


    
    def process(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        folder_path = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return(output_image, output_mask, image_name, folder_path)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
    
#NOTE The following code is not my own. It is copied from skfoo/ComfyUI-Coziness. All credit goes to him. I am only modifying it slightly to serve my own purposes.
# Node class definition
class LoRANameGenerator:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "subfolder": ("STRING", {
                    "multiline": False,
                    "default": ""}),
                "name_prefix": ("STRING", {
                    "multiline": False,
                    "default": ""}),
                "name_suffix": ("STRING", {
                    "multiline": False,
                    "default": ""}),
                "extension": ("STRING", {
                    "multiline": False,
                    "default": ".safetensors"}),
                "start": ("INT", {"default": 1,"tooltip": "The starting number of the lora you want to use."}),
                "range": ("INT", {"default": 1,"tooltip": "The range of loras from the starting number to use (start+range)."}),
                "idx": ("INT", {"default": 1,"tooltip": "Total number of loaders being used."}),
                "total_loaders": ("INT", {"default": 1,"tooltip": "Total number of loaders being used."}),
                "lora_step": ("INT", {"default": 1,"tooltip": "This is the number the loras increment by. So if the lora's values increment by 5, but a 5 here."}),
                "zero_padding": ("INT", {"default": 1,"tooltip": "number of zeros to pad the number with."}),
                "increment_seed": ("INT", {"default": 0,"forceInput":True,"tooltip": "Connect to a seed number that increments by 1 each run and start it from 0."}),
                
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("lora_name", "lora_path", "starting_at")
    FUNCTION = "generate_names"  # The function name for processing the inputs
    CATEGORY = "BKNodes/LoRA Testing"  # A category for the node, adjust as needed
    LABEL = "LoRA Name Generator"  # Default label text
    OUTPUT_NODE = True


    
    def generate_names(self, name_prefix, name_suffix, extension, start, range, idx, total_loaders, zero_padding, lora_step, subfolder, increment_seed):        # Ensure the input is treated as text
        lora_number = 0
        step = start + (range * increment_seed)
        if total_loaders != 0:
            intermediate_value = abs(range * (idx / total_loaders) + step)
            lora_number = int(lora_step * (intermediate_value // lora_step))
        padded_integer_string = f"{lora_number:0{zero_padding}d}"
        lora_path = subfolder + name_prefix + padded_integer_string + name_suffix + extension
        lora_name = name_prefix + padded_integer_string + name_suffix
        start += range
        return(lora_name, lora_path, step)
    

class LoRATestingNode:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "subfolder": ("STRING", {
                "multiline": False,
                "default": ""}),
            "name_prefix": ("STRING", {
                "multiline": False,
                "default": ""}),
            "name_suffix": ("STRING", {
                "multiline": False,
                "default": ""}),
            "extension": ("STRING", {
                "multiline": False,
                "default": ".safetensors"}),
            "start": ("INT", {"default": 1,"tooltip": "The starting number of the lora you want to use."}),
            "range": ("INT", {"default": 1,"tooltip": "The range of loras from the starting number to use (start+range)."}),
            "idx": ("INT", {"default": 1,"tooltip": "Total number of loaders being used."}),
            "total_loaders": ("INT", {"default": 1,"tooltip": "Total number of loaders being used."}),
            "lora_step": ("INT", {"default": 1,"tooltip": "This is the number the lora files increment by."}),
            "zero_padding": ("INT", {"default": 1,"tooltip": "number of zeros to pad the number with."}),
            "increment_seed": ("INT", {"default": 0,"forceInput":True,"tooltip": "Connect to a seed number that increments by 1 each run and start it from 0."}),
         }}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "lora_name", "lora_path")
    FUNCTION = "get_lora"
    CATEGORY = "BKNodes/LoRA Testing"  # A category for the node, adjust as needed
    LABEL = "LoRA Testing Node"  # Default label text

    def get_lora(self, model, clip, name_prefix, name_suffix, extension, start, range, idx, total_loaders, zero_padding, lora_step, subfolder, increment_seed):
        result = (model, clip,"","")
        lora_number = 0
        current_start = start + (range * increment_seed)
        if total_loaders != 0:
            intermediate_value = abs(range * (idx / total_loaders) + current_start)
            lora_number = int(lora_step * (intermediate_value // lora_step))
        padded_integer_string = f"{lora_number:0{zero_padding}d}"
        lora_path = subfolder + name_prefix + padded_integer_string + name_suffix + extension
        lora_name = name_prefix + padded_integer_string + name_suffix
        
        lora_items = self.selected_loras.updated_lora_items_with_text(lora_path)

        if len(lora_items) > 0:
            for item in lora_items:
                result = item.apply_lora(result[0], result[1])
            
        return(result[0],result[1],lora_name, lora_path, current_start)    
    
class SingleLoRATestNode:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "subfolder": ("STRING", {
                "multiline": False,
                "default": ""}),
            "name_prefix": ("STRING", {
                "multiline": False,
                "default": ""}),
            "name_suffix": ("STRING", {
                "multiline": False,
                "default": ""}),
            "extension": ("STRING", {
                "multiline": False,
                "default": ".safetensors"}),
            
            "idx": ("INT", {"default": 2,"tooltip": "Total number of loaders being used."}),
            "max": ("INT", {"default": 3000,"tooltip": "The maxiumum number the loras go up to"}),
            "min": ("INT", {"default": 2,"tooltip": "The minimum number the loras go down to."}),
            "lora_step": ("INT", {"default": 2,"tooltip": "This is the number the lora files increment by."}),
            "zero_padding": ("INT", {"default": 9,"tooltip": "number of zeros to pad the number with."}),
         }}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "INT")
    RETURN_NAMES = ("MODEL", "CLIP", "lora_name", "lora_path", "lora_number")
    FUNCTION = "get_lora"
    CATEGORY = "BKNodes/LoRA Testing"  # A category for the node, adjust as needed
    LABEL = "LoRA Testing Node"  # Default label text

    def get_lora(self, clip, model, subfolder, name_prefix, name_suffix, extension, idx, max, min, lora_step, zero_padding):
        result = (model, clip,"","",0)
        lora_number = get_nearest_step_value(min, max, lora_step, idx)
        padded_integer_string = f"{lora_number:0{zero_padding}d}"
        lora_path = subfolder + name_prefix + padded_integer_string + name_suffix + extension
        lora_name = name_prefix + padded_integer_string + name_suffix
        
        lora_items = self.selected_loras.updated_lora_items_with_text(lora_path)

        if len(lora_items) > 0:
            for item in lora_items:
                result = item.apply_lora(result[0], result[1])
            
        return(result[0],result[1],lora_name, lora_path, lora_number) 
    

def get_line_by_index(index: int, text: str) -> str:
    """
    Returns the line at the specified index from a multiline string.
    
    Args:
        index (int): The zero-based index of the line to return.
        text (str): The multiline string input.
        
    Returns:
        str: The line at the given index, or an error message if the index is out of range.
    """
    lines = text.splitlines()
    
    if 0 <= index < len(lines):
        return lines[index]
    else:
        return f"Index {index} out of range. Text has {len(lines)} lines."
    
def count_lines(text: str) -> int:
    """
    Returns the number of lines in a multiline string.
    
    Args:
        text (str): The multiline string input.
        
    Returns:
        int: The total number of lines.
    """
    return len(text.splitlines())

def has_next_line(index: int, text: str) -> bool:
    """
    Checks if there is a next line after the given index in a multiline string.

    Args:
        index (int): The current line index (0-based).
        text (str): The multiline string input.

    Returns:
        bool: True if there is a next line, False otherwise.
    """
    lines = text.splitlines()
    return (index + 1) < len(lines)

def parse_int(s: str) -> int:
    """
    Attempts to parse a string into an integer.

    Args:
        s (str): The input string.

    Returns:
        int: The parsed integer.

    Raises:
        ValueError: If the string cannot be converted to an integer.
    """
    return int(s)
    
class MultiLoRATestNode:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "while_loop_idx": ("INT", {"default": 0,"tooltip": "Manually add a +1 to a value in the while loop and use that to increment this value."}),
            "subfolder": ("STRING", {
                "multiline": False,
                "default": ""}),
            "name_prefix": ("STRING", {
                "multiline": False,
                "default": ""}),
            "name_suffix": ("STRING", {
                "multiline": False,
                "default": ""}),
            "extension": ("STRING", {
                "multiline": False,
                "default": ".safetensors"}),
            "lora_list": ("STRING", {
                "multiline": True,
                "default": ""}),
            
            "lora_step": ("INT", {"default": 2,"tooltip": "This is the number the lora files increment by."}),
            "zero_padding": ("INT", {"default": 9,"tooltip": "number of zeros to pad the number with."}),
         }}

    RETURN_TYPES = ("MODEL", "CLIP","BOOLEAN", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("MODEL", "CLIP","has_next", "lora_name", "lora_path", "lora_number")
    FUNCTION = "get_lora"
    CATEGORY = "BKNodes/LoRA Testing"  # A category for the node, adjust as needed
    LABEL = "LoRA Testing Node"  # Default label text

    def get_lora(self, clip, model, subfolder, name_prefix, name_suffix, extension, lora_list, while_loop_idx, lora_step, zero_padding):
        result = (model, clip,"","",0, False)
        has_next = has_next_line(while_loop_idx, lora_list)
        current_lora_string = get_line_by_index(while_loop_idx, lora_list)
        idx = parse_int(current_lora_string)
        
        lora_number = get_nearest_step(lora_step, idx)
        padded_integer_string = f"{lora_number:0{zero_padding}d}"
        lora_path = subfolder + name_prefix + padded_integer_string + name_suffix + extension
        lora_name = name_prefix + padded_integer_string + name_suffix
        
        lora_items = self.selected_loras.updated_lora_items_with_text(lora_path)

        if len(lora_items) > 0:
            for item in lora_items:
                result = item.apply_lora(result[0], result[1])
            
        return(result[0],result[1],lora_name, lora_path, lora_number, has_next) 
    
def get_nearest_step_value(min_val, max_val, step_value, starting_number):
    # Step 1: If the starting_number is above the max value, keep subtracting max_val until it's within range
    while starting_number > max_val:
        starting_number -= max_val
    
    # Step 2: If the starting_number is below the min value, keep adding max_val until it's within range
    while starting_number < min_val:
        starting_number += max_val
    
    # Step 3: Find the nearest multiple of the step value
    nearest_multiple = round(starting_number / step_value) * step_value
    return nearest_multiple

def get_nearest_step(step_value, starting_number):
    # Step 3: Find the nearest multiple of the step value
    nearest_multiple = round(starting_number / step_value) * step_value
    return nearest_multiple

# maintains a list of lora objects made from a prompt, preserving loaded loras across changes
class SelectedLoras:
    def __init__(self):
        self.lora_items = []

    # returns a list of loaded loras using text from LoraTextExtractor
    def updated_lora_items_with_text(self, text):
        available_loras = self.available_loras()
        self.update_current_lora_items_with_new_items(self.items_from_lora_text_with_available_loras(text, available_loras))
        
        for item in self.lora_items:
            if item.lora_name not in available_loras:
                raise ValueError(f"Unable to find lora with name '{item.lora_name}'")
            
        return self.lora_items

    def available_loras(self):
        return folder_paths.get_filename_list("loras")
    
    def items_from_lora_text_with_available_loras(self, lora_text, available_loras):
        return LoraItemsParser.parse_lora_items_from_text(lora_text, self.dictionary_with_short_names_for_loras(available_loras))
    
    def dictionary_with_short_names_for_loras(self, available_loras):
        result = {}
        
        for path in available_loras:
            result[os.path.splitext(os.path.basename(path))[0]] = path
        
        return result

    def update_current_lora_items_with_new_items(self, lora_items):
        if self.lora_items != lora_items:
            existing_by_name = dict([(existing_item.lora_name, existing_item) for existing_item in self.lora_items])
            
            for new_item in lora_items:
                new_item.move_resources_from(existing_by_name)
            
            self.lora_items = lora_items

class LoraItemsParser:

    @classmethod
    def parse_lora_items_from_text(cls, lora_text, loras_by_short_names = {}, default_weight=1, weight_separator=":"):
        return cls(lora_text, loras_by_short_names, default_weight, weight_separator).execute()

    def __init__(self, lora_text, loras_by_short_names, default_weight, weight_separator):
        self.lora_text = lora_text
        self.loras_by_short_names = loras_by_short_names
        self.default_weight = default_weight
        self.weight_separator = weight_separator
        self.prefix_trim_re = re.compile("\A<(lora|lyco):")
        self.comment_trim_re = re.compile("\s*#.*\Z")
    
    def execute(self):
        return [LoraItem(elements[0], elements[1], elements[2])
            for line in self.lora_text.splitlines()
            for elements in [self.parse_lora_description(self.description_from_line(line))] if elements[0] is not None]
    
    def parse_lora_description(self, description):
        if description is None:
            return (None,)
        
        lora_name = None
        strength_model = self.default_weight
        strength_clip = None
        
        remaining, sep, strength = description.rpartition(self.weight_separator)
        if sep == self.weight_separator:
            lora_name = remaining
            strength_model = float(strength)
            
            remaining, sep, strength = remaining.rpartition(self.weight_separator)
            if sep == self.weight_separator:
                strength_clip = strength_model
                strength_model = float(strength)
                lora_name = remaining
        else:
            lora_name = description
        
        if strength_clip is None:
            strength_clip = strength_model
        
        return (self.loras_by_short_names.get(lora_name, lora_name), strength_model, strength_clip)

    def description_from_line(self, line):
        result = self.comment_trim_re.sub("", line.strip())
        result = self.prefix_trim_re.sub("", result.removesuffix(">"))
        return result if len(result) > 0 else None
    

class LoraItem:
    def __init__(self, lora_name, strength_model, strength_clip):
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        self._loaded_lora = None
    
    def __eq__(self, other):
        return self.lora_name == other.lora_name and self.strength_model == other.strength_model and self.strength_clip == other.strength_clip
    
    def get_lora_path(self):
        return folder_paths.get_full_path("loras", self.lora_name)
        
    def move_resources_from(self, lora_items_by_name):
        existing = lora_items_by_name.get(self.lora_name)
        if existing is not None:
            self._loaded_lora = existing._loaded_lora
            existing._loaded_lora = None

    def apply_lora(self, model, clip):
        if self.is_noop:
            return (model, clip)
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, self.lora_object, self.strength_model, self.strength_clip)
        return (model_lora, clip_lora)

    @property
    def lora_object(self):
        if self._loaded_lora is None:
            lora_path = self.get_lora_path()
            if lora_path is None:
                raise ValueError(f"Unable to get file path for lora with name '{self.lora_name}'")
            self._loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        return self._loaded_lora

    @property
    def is_noop(self):
        return self.strength_model == 0 and self.strength_clip == 0



# Register the node in ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Save Image Easy": SaveImageEasy,  # The name that will show in the UI
    "Save\Overwrite Image": OverwriteImage,  # The name that will show in the UI
    "Load Image Easy": LoadImageWithPath,  # The name that will show in the UI
    "Ollama Connectivity Data": JSONExtractor,
    "LoRA Testing Node": LoRATestingNode,
    "Single LoRA Test Node": SingleLoRATestNode,
    "Save\Overwrite Text File": CreateOverwriteTxtFile,
    "Multi LoRA Test Node": MultiLoRATestNode,
}
