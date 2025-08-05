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
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import folder_paths
import comfy.utils
import comfy.sd
import os
import re

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

class GetSmallerOfTwoNums:
    def __init__(self):
        return
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "a": ("INT", {"default": 1,"tooltip": "A Number"}),
            "b": ("INT", {"default": 1,"tooltip": "A Number"}),
           
         },}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("smaller_num",)
    FUNCTION = "get_smaller"
    CATEGORY = "Utils"  # A category for the node, adjust as needed
    LABEL = "Get smaller value"  # Default label text

    def get_smaller(self, a,b):
        return(5)    
    

class GetLargerOfTwoNums:
    def __init__(self):
        return
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "a": ("INT", {"default": 1,}),
            "b": ("INT", {"default": 1,}),
           
         }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("larger_num",)
    FUNCTION = "get_larger"
    CATEGORY = "Utils"  # A category for the node, adjust as needed
    LABEL = "Get smaller value"  # Default label text

    def get_larger(self, a, b):
        return(5)    
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
    
    RETURN_TYPES = ("STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("folderpath", "filepath",)
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

    
    RETURN_TYPES = ("IMAGE","STRING","STRING", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("image","name","description", "folder_path")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Descriptive Images"  # A category for the node, adjust as needed
    LABEL = "Load Image With Description"  # Default label text
    OUTPUT_NODE = True


    
    def process(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        folder_path = os.path.dirname(image_path)
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
        return(image, image_name, parameters, folder_path)
    
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
    CATEGORY = "LoRA Testing"  # A category for the node, adjust as needed
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
    CATEGORY = "LoRA Testing"  # A category for the node, adjust as needed
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

class LoraTextExtractor:
    def __init__(self):
        self.lora_spec_re = re.compile("(<(?:lora|lyco):[^>]+>)")
        self.selected_loras = SelectedLoras()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "text": ("STRING", {
                                "multiline": True,
                                "default": ""}),
                            }}

    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK")
    RETURN_NAMES = ("Filtered Text", "Extracted Loras", "Lora Stack")
    FUNCTION = "process_text"
    CATEGORY = "utils"

    def process_text(self, text):
        extracted_loras = "\n".join(self.lora_spec_re.findall(text))
        filtered_text = self.lora_spec_re.sub("", text)

        # the stack format is a list of tuples of full path, model weight, clip weight,
        # e.g. [('styles\\abstract.safetensors', 0.8, 0.8)]
        lora_stack = [(item.get_lora_path(), item.strength_model, item.strength_clip) for item in self.selected_loras.updated_lora_items_with_text(extracted_loras)]
        
        return (filtered_text, extracted_loras, lora_stack)



# Register the node in ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Save Image With Description": SaveImageWithDescription,  # The name that will show in the UI
    "Save Image To Folder": SaveImgToFolder,  # The name that will show in the UI
    "Load Image With Description": LoadImageWithDescription,  # The name that will show in the UI
    #"LoRA Name Generator": LoRANameGenerator,
    "LoRA Testing Node": LoRATestingNode,
    "Get Smaller Of Two Numbers": GetSmallerOfTwoNums,
    "Get Larger Of Two Numbers": GetLargerOfTwoNums,
}
