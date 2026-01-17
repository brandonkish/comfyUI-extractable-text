from email.mime import image
from torch import Tensor
import os
import scipy.ndimage as ndi
from io import StringIO
import hashlib
import torch
import torch.nn.functional as F
import shutil
import pandas as pd
import itertools
from datetime import datetime
import unicodedata
import json
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
import latent_preview
from comfy.samplers import KSampler
import csv
import glob

try:
    import piexif.helper
    import piexif
    from .exif.exif import read_info_from_image_stealth
    piexif_loaded = True
except ImportError:
    piexif_loaded = False

HEADER_LENGTH = 100

def print_debug_header(is_debug, name):
    if is_debug:
        name = f" DEBUG {name} "
        sides = (HEADER_LENGTH - len(name)) // 2
        print(f"{'#' * sides}{name}{'#' * sides}")

def print_debug_bar(is_debug):
    if is_debug:
        print(f"{'#' * HEADER_LENGTH}")

def make_filename(filename):
    return "image" if filename == "" else filename

# Utility function for handling whitespace
def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

class BKAddToPath:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "file_name": ("STRING", {"": f'', "multiline": False, "tooltip": "The folder path you wish to add a subfolder to."}),
                "folder_path": ("STRING", {"": f'', "multiline": False, "tooltip": "Name of the file you wish to append the subfolder name to."}),
                "add_to_filename": ("BOOLEAN", {"default": True, "tooltip": "If true, this will append the name of the folder to the filename output."}),
                "add_as_subfolder": ("BOOLEAN", {"default": True, "tooltip": "If true, will add this as a subfolder to the path."}),
                "name": ("STRING", {"": f'', "multiline": False, "tooltip": "Name of the subfolder you wish to add."}),
            },
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("updated_filename","updated_path","updated_full_path",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Add To Path"  # Default label text
    OUTPUT_NODE = True

    def process(self, file_name, folder_path, add_to_filename, name, add_as_subfolder):


        if add_as_subfolder:
            updated_folder_path = f"{folder_path.rstrip("\\")}\\{name.rstrip("\\")}"
        else:
            updated_folder_path = f"{folder_path.rstrip("\\")}"

        if add_to_filename:
            updated_filename = f"{file_name.rstrip("\\")}_{name.rstrip("\\")}"
        else:
            updated_filename = f"{file_name.rstrip("\\")}"
        
        updated_fullpath = f"{updated_folder_path}\\{updated_filename}"

        return (updated_filename, updated_folder_path, updated_fullpath)

class BKLoopStatusText:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "idx": ("INT", {"default": 0,"tooltip": "Current loop index."}),
                "total": ("INT", {"default": 0,"tooltip": "Total number of images to be generated."}),
      
            },
        }
    
    RETURN_TYPES = ("STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("status_text")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Loop Status Text"  # Default label text
    OUTPUT_NODE = True

    def process(self, idx, total):
        nidx:str = str(idx + 1)
        ntotal:str = str(total)
        current_time = datetime.now().strftime('%I:%M:%S %p')
        status_text:str = str(f"Genrating image {nidx} of {ntotal} @ [{current_time}].")
        print(f"DEBUG: {status_text}")  # Check what is printed
        return (status_text,)
    
class BKLineCounter:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
            "lines": ("STRING", {"default": f'', "multiline": True, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, lines):
        return float("nan")
    
    RETURN_TYPES = ("INT","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("num_of_lines","text")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Line Counter"  # Default label text
    OUTPUT_NODE = True

    def process(self, lines):
        # Split the input string into a list of lines
        line_list = lines.split('\n')
        line_count = len(line_list)
        return (line_count, lines)
    
class BKReplaceEachTagRandom:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "prompt":("STRING", {"default": f'', "multiline": False, "tooltip": "Prompt with tags in it."}),
                "tag": ("STRING", {"default": f'', "multiline": False, "tooltip": "Tag to replace."}),
                "lines": ("STRING", {"default": f'', "multiline": True, "tooltip": "Each tag found in the prompt is replaced by a randomly selected line in the line list. So if there are 4 tags, each tag will get a different randomly selected line."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, prompt, tag, lines):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING", "STRING","INT")  # This specifies that the output will be text
    RETURN_NAMES = ("prompt_with_tag_replaced","all_lines")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Replace Each Tag Random"  # Default label text
    OUTPUT_NODE = True

    def process(self, prompt, tag, lines):
        # Ensure we have a list of possible lines
        line_list = [line for line in lines.split('\n') if line.strip()]
        
        # Split the prompt on each occurrence of the tag
        parts = prompt.split(tag)
        
        # Rebuild the prompt, inserting a random line between parts
        prompt_w_tag_replaced = ''
        for i, part in enumerate(parts):
            prompt_w_tag_replaced += part
            if i < len(parts) - 1:
                prompt_w_tag_replaced += random.choice(line_list)
        
        return (prompt_w_tag_replaced, lines)
    
class BKReplaceEachTagRandomByFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "prompt":("STRING", {"default": f'', "multiline": False, "tooltip": "Prompt with tags in it."}),
                "tag": ("STRING", {"default": f'', "multiline": False, "tooltip": "Tag to replace."}),
                "lines_file_path": ("STRING", {"default": f'', "multiline": False, "tooltip": "Each tag found in the prompt is replaced by a randomly selected line in the line file. So if there are 4 tags, each tag will get a different randomly selected line."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, prompt, tag, lines):
        return float("nan")
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("prompt_with_tag_replaced",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Replace Each Tag Random By File"  # Default label text
    OUTPUT_NODE = True

def process(self, prompt, tag, lines_file_path):
    # Load lines from the specified file
    try:
        with open(lines_file_path, 'r') as file:
            line_list = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        raise ValueError(f"The file at '{lines_file_path}' was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the file: {e}")

    # Split the prompt on each occurrence of the tag
    parts = prompt.split(tag)
    
    # Rebuild the prompt, inserting a random line between parts
    prompt_w_tag_replaced = ''
    for i, part in enumerate(parts):
        prompt_w_tag_replaced += part
        if i < len(parts) - 1:
            prompt_w_tag_replaced += random.choice(line_list)
    
    return (prompt_w_tag_replaced,)

class BKReadTextFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text_file_path": ("STRING", {"default": f'', "multiline": False, "tooltip": "File path of text file to read."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, text_file_path):
        return float("nan")
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Read Text File"  # Default label text
    OUTPUT_NODE = True

    def process(self, text_file_path):
        # Remove leading and trailing double quotes from the file path
        text_file_path = text_file_path.strip('"')
        text_file_path = text_file_path.strip('\'')

        if not os.path.isabs(text_file_path):
            # If relative, construct the path with the output directory
            text_file_path = os.path.join(self.output_dir.strip(), text_file_path.strip())
        
        try:
            with open(text_file_path, 'r') as file:
                text = file.read()  # Read the entire content of the file
        except FileNotFoundError:
            raise ValueError(f"The file at '{text_file_path}' was not found.")
        except Exception as e:
            raise ValueError(f"An error occurred while reading the file: {e}")

        return (text,)
    
class BKMultiReadTextFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text_file_paths": ("STRING", {"default": f'', "multiline": True, "tooltip": "File path of text file to read."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, text_file_paths):
        return float("nan")
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Read Text File"  # Default label text
    OUTPUT_NODE = True


    def process(self, text_file_paths):
        # Initialize an empty string to store all the concatenated content
        all_text = ""

        # Split the string into file paths (assuming each file path is on a new line)
        file_paths = text_file_paths.splitlines()

        for file_path in file_paths:
            # Remove leading and trailing quotes (both single and double)
            clean_path = file_path.strip('"').strip("'")

            # If relative, construct the path with the output directory
            if not os.path.isabs(clean_path):
                clean_path = os.path.join(self.output_dir.strip(), clean_path.strip())

            # Check if the path is a directory
            if os.path.isdir(clean_path):
                try:
                    # Get all .txt files in the directory
                    for filename in os.listdir(clean_path):
                        if filename.endswith('.txt'):
                            file_path = os.path.join(clean_path, filename)
                            try:
                                # Try opening the file with UTF-8 encoding
                                with open(file_path, 'r', encoding='utf-8-sig') as file:
                                    all_text += file.read().strip() + "\n"
                            except UnicodeDecodeError:
                                # Fallback to a more lenient encoding
                                with open(file_path, 'r', encoding='latin1') as file:
                                    all_text += file.read().strip() + "\n"
                            except FileNotFoundError:
                                raise ValueError(f"The file at '{file_path}' was not found.")
                            except Exception as e:
                                raise ValueError(f"An error occurred while reading the file '{file_path}': {e}")
                except Exception as e:
                    raise ValueError(f"An error occurred while reading the directory '{clean_path}': {e}")
            else:
                try:
                    # Try opening the file with UTF-8 encoding
                    with open(clean_path, 'r', encoding='utf-8-sig') as file:
                        all_text += file.read().strip() + "\n"
                except UnicodeDecodeError:
                    # Fallback to a more lenient encoding
                    with open(clean_path, 'r', encoding='latin1') as file:
                        all_text += file.read().strip() + "\n"
                except FileNotFoundError:
                    raise ValueError(f"The file at '{clean_path}' was not found.")
                except Exception as e:
                    raise ValueError(f"An error occurred while reading the file '{clean_path}': {e}")

        # Return the concatenated content as a single string
        return (all_text.strip(),)




'''
    def process(self, text_file_paths):
        # Initialize an empty string to store all the concatenated content
        all_text = ""

        # Split the string into file paths (assuming each file path is on a new line)
        file_paths = text_file_paths.splitlines()

        for file_path in file_paths:
            # Remove leading and trailing quotes (both single and double)
            clean_path = file_path.strip('"').strip("'")

            try:
                # Open and read the content of each file
                with open(clean_path, 'r') as file:
                    all_text += file.read().strip() + "\n"  # Append the content and add a newline between files
            except FileNotFoundError:
                raise ValueError(f"The file at '{clean_path}' was not found.")
            except Exception as e:
                raise ValueError(f"An error occurred while reading the file '{clean_path}': {e}")



        # Return the concatenated content as a single string
        return (all_text.strip(),)

'''
class BKComboTag:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "prompt":("STRING", {"default": f'', "multiline": False, "tooltip": "Prompt with tags in it."}),
                "tag": ("STRING", {"default": f'', "multiline": False, "tooltip": "Tag to replace."}),
                "iteration": ("INT", {"default": 0,"tooltip": "the value of the line number to select the line if 'randomize' is false. If the line number is larger than the number of lines, then the line number will loop around."}),
                "separator": ("STRING", {"default": f'', "multiline": False, "tooltip": "This string will be used to determine where to split the line."}),
                "randomize": ("BOOLEAN", {"default": False, "tooltip": "If true will randomly select a line instead of using the provided line number."}),
                "exclude_reordered": ("BOOLEAN", {"default": False, "tooltip": "If true, will skip combinations where the only difference is the words being re-rodered."}),
                "lines": ("STRING", {"default": f'', "multiline": True, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, prompt, tag, iteration, randomize, lines, separator, exclude_reordered):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","INT", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("prompt_w_tags_replaced","line","iteration_num", "lines")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Combo Tag"  # Default label text
    OUTPUT_NODE = True

    def process(self, prompt, tag, iteration, randomize, lines, separator, exclude_reordered):
        # Clean input lines and ensure list of strings
        if isinstance(lines, str):
            # Handle case where lines might be a single string with newlines
            lines = [l.strip() for l in lines.splitlines() if l.strip()]
        else:
            lines = [str(line).strip() for line in lines if str(line).strip()]

        n = len(lines)
        if n == 0:
            return (prompt, "", iteration, lines)

        # Compute how many total combinations exist
        if exclude_reordered:
            total_combos = sum(math.comb(n, r) for r in range(1, n + 1))
        else:
            total_combos = sum(math.perm(n, r) for r in range(1, n + 1))

        # Wrap around iteration index
        idx = (iteration - 1) % total_combos

        # Randomize order of input lines (not chars)
        if randomize:
            shuffled_lines = lines[:]  # make a copy
            random.shuffle(shuffled_lines)
        else:
            shuffled_lines = lines

        # Iterate lazily to the desired combination
        counter = 0
        for r in range(1, n + 1):
            if exclude_reordered:
                combos = itertools.combinations(shuffled_lines, r)
            else:
                combos = itertools.permutations(shuffled_lines, r)

            for combo in combos:
                if counter == idx:
                    line = separator.join(combo)
                    prompt_w_tag_replaced = prompt.replace(tag, line)
                    return (prompt_w_tag_replaced, line, iteration, lines)
                counter += 1

        # Fallback (shouldn’t reach)
        return (prompt, "", iteration, lines)
    

class BKCaptionFileParser:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "list_of_image_paths": ("STRING", {"default": f'', "multiline": False, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
                "seed": ("INT", {"default": 0,"min":0,"tooltip": "This is really the index for the row to extract, I put it as a seed so you can have it auto increment without a new node. But think of it as the idx value."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, prompt, tag, iteration, randomize, lines, separator, exclude_reordered):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","INT", "INT", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("image_name","image_directory","idx", "total_images", "status_text")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Combo Tag"  # Default label text
    OUTPUT_NODE = True

    def process(self, list_of_image_paths, seed):
        # Split the images_missing_captions string back into a list of paths
        image_paths = list_of_image_paths.split("\n")
        
        # Get the total number of images
        total_images = len(image_paths)
        
        # Ensure the seed is within the valid range
        if seed >= total_images or seed < 0:
            raise IndexError("Seed index out of range.")
        
        # Extract the image path for the given seed
        image_path = image_paths[seed]
        
        # Split the image path into directory and file name (without extension)
        image_directory, image_filename = os.path.split(image_path)
        image_name, _ = os.path.splitext(image_filename)
        
        # Create the status message
        current_time = datetime.now().strftime('%I:%M:%S %p')
        status_text = f"Processing [{seed + 1} of {total_images}] @ [{current_time}]: {image_path}"
        
        # Return the image name, directory, seed, total images, and status text
        return (image_name, image_directory, seed, total_images, status_text)
    

class BKCaptionFileScanner:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "folder_path": ("STRING", {"default": f'', "multiline": False, "tooltip": ""}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, prompt, tag, iteration, randomize, lines, separator, exclude_reordered):
        return float("nan")
    
    RETURN_TYPES = ("INT","STRING", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("num_total_missing","folder_path","list_of_image_paths")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Combo Tag"  # Default label text
    OUTPUT_NODE = True

    def process(self, folder_path):
        updated_image_list = []
        images_missing_captions = ""
        
        # Step 1: Get list of all PNG files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                # Construct the absolute path of the PNG file
                image_path = os.path.join(folder_path, filename)
                
                # Check if a corresponding text file exists
                txt_file = os.path.splitext(filename)[0] + ".txt"
                txt_file_path = os.path.join(folder_path, txt_file)
                
                # If the text file does not exist, add the image path to the list
                if not os.path.exists(txt_file_path):
                    updated_image_list.append(image_path)
        
        # If the updated_image_list is empty, raise an error
        if not updated_image_list:
            raise ValueError("All images have caption files.")
        
        # Step 2: Create a string of paths with each path on a new line
        images_missing_captions = "\n".join(updated_image_list)
        
        # Return the total number of missing images, folder path, and the list of missing captions
        num_total_missing = len(updated_image_list)
        return (num_total_missing, folder_path, images_missing_captions)



class BKWriteTextFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text": ("STRING", {"default": f'', "multiline": True, "tooltip": "Text to save to the text file."}),
                "text_file_path": ("STRING", {"default": f'', "multiline": False, "tooltip": "File path of text file to write."}),
                "save": ("BOOLEAN", {"default": False, "tooltip": "If true will save the file, else will not save the file."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, text_file_path, text, save):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","INT","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("prompt_with_tag_replaced", "line_text", "idx", "lines_text")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Write Text File"  # Default label text
    OUTPUT_NODE = True

    def process(self, text_file_path, text, save):
        if save:
            # Remove leading and trailing double quotes from the file path
            text_file_path = text_file_path.strip('"').strip("'")

            try:
                # Open the file in write mode ('w') with UTF-8 encoding to handle special characters
                with open(text_file_path, 'w', encoding='utf-8') as file:
                    file.write(text)  # Write the content to the file
            except Exception as e:
                raise ValueError(f"An error occurred while writing to the file: {e}")
        return (text_file_path,)


class BKTSVFileLoader:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "overwrite": ("BOOLEAN", {"default": True, "tooltip": "If true, will overwrite images that already exist in the folder, if false, will not overwrite images that already exist."}),
                "create": ("BOOLEAN", {"default": True, "tooltip": "If true, will add images that do not exist to the list of images to generate, if false, will not add any images that do not already exist."}),
                "images_path": ("STRING", {"default": f'', "multiline": False, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
                "tsv_file_path": ("STRING", {"default": f'', "multiline": False, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, overwrite, create, images_path, tsv_file_path):
        return float("nan")
    
    RETURN_TYPES = ("INT", "STRING", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("total_images","images_path","tsv_string")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK TSV File Loader"  # Default label text
    OUTPUT_NODE = True

    def process(self, overwrite, create, images_path, tsv_file_path):
        # Trim leading/trailing quotes from paths
        tsv_file_path = tsv_file_path.strip('"').strip("'")
        images_path = images_path.strip('"').strip("'")

        # Initialize the updated list and total image count
        updated_list = []
        total_imgs_count = 0

        try:
            # Open and read the TSV file
            with open(tsv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')
                rows = list(reader)

                # Check if TSV file is empty or has no data
                if len(rows) == 0:
                    raise ValueError("TSV file is empty")

                # Check if required headers are present
                required_columns = ['FILENAME', 'LORA_STRING', 'CAPTION', 'PROMPT']
                for column in required_columns:
                    if column not in reader.fieldnames:
                        raise ValueError(f"No columns with the header {column} in the first row found in the tsv. Ensure there is a FILENAME, LORA_STRING, CAPTION, and PROMPT columns in the tsv, and their first row includes their name as a header.")

                # Process each row in the TSV
                for row in rows:
                    filename = row['FILENAME'].strip().replace('"', '').replace("'", "")
                    file_path = os.path.join(self.output_dir, images_path, filename + '.png')

                    # Check if the image file exists
                    image_exists = os.path.exists(file_path)

                    # Handle overwrite and create flags
                    if overwrite and image_exists:
                        updated_list.append({
                            'FILENAME': filename,
                            'PROMPT': row['PROMPT'],
                            'LORA_STRING': row['LORA_STRING'],
                            'CAPTION': row['CAPTION']
                        })
                        total_imgs_count += 1
                    elif create and not image_exists:
                        updated_list.append({
                            'FILENAME': filename,
                            'PROMPT': row['PROMPT'],
                            'LORA_STRING': row['LORA_STRING'],
                            'CAPTION': row['CAPTION']
                        })
                        total_imgs_count += 1

                # Error handling for empty updated list
                if total_imgs_count == 0:
                    if not overwrite and create:
                        raise ValueError("all images already exist in folder, nothing to create")
                    if overwrite and not create:
                        raise ValueError("no existing images found to overwrite. Did you mean to set create_images to true?")
                    if not overwrite and not create:
                        raise ValueError("Both overwrite_images and create_images are set to false, please enable at least one to generate images.")

                # Generate the TSV string from updated list
                tsv_as_string = "\t".join(['FILENAME', 'PROMPT', 'LORA_STRING', 'CAPTION']) + '\n'
                for entry in updated_list:
                    tsv_as_string += "\t".join([entry['FILENAME'], entry['PROMPT'], entry['LORA_STRING'], entry['CAPTION']]) + '\n'

                return total_imgs_count,  images_path, tsv_as_string

        except FileNotFoundError:
            raise ValueError(f"File {tsv_file_path} not found.")
        except Exception as e:
            raise e
        

class BKTSVPromptReader:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "output_headers": ("STRING", {"default": f'', "multiline": False, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
                "seed": ("INT", {"default": 0,"min":0,"tooltip": "This is really the index for the row to extract, I put it as a seed so you can have it auto increment without a new node. But think of it as the idx value."}),
                "tsv_file_paths": ("STRING", {"default": f'', "multiline": True, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
                
            },
        }
    
    @classmethod
    def IS_CHANGED(self, tsv_file_paths, seed, output_headers):
        return float("nan")
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING", "INT")  # This specifies that the output will be text
    RETURN_NAMES = ("pos_prompt", "neg_prompt", "seed", "tsv_folder", "tsv_name", "header", "total")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK TSV Prompt Reader"  # Default label text
    OUTPUT_NODE = True

    def process(self, tsv_file_paths, seed, output_headers):
        # Split output_headers into a list of headers to track
        output_headers = [header.strip() for header in output_headers.split(',')]
        
        master_list = []  # This will hold the combined data for the output_headers columns
        neg_prompt = []  # Holds the complementary data for each column (like BUG_N for BUG)

        last_file_info = None  # To store the last file's information (tsv_folder, tsv_name)



        #tsv_file_paths_list = tsv_file_paths.splitlines()

        # Iterate through the list of file paths
        for file_path in tsv_file_paths.splitlines():

            file_path = file_path.strip()  # Remove leading/trailing whitespace
            if not file_path:
                continue  # Skip empty lines

            # Handle the path to the file (absolute or relative)
            file_path = os.path.normpath(file_path)  # Normalize the path to remove redundant slashes
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.output_dir, file_path)  # Prepend output_dir for relative paths


            # Extract folder and filename info for later use
            tsv_folder = os.path.dirname(file_path)
            tsv_name = os.path.splitext(os.path.basename(file_path))[0]
            last_file_info = (tsv_folder, tsv_name)


            # Try to open and read the TSV file
            try:
                
                with open(file_path, 'r', newline='', encoding='utf-8') as tsv_file:

                    tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
                    temp_master = {header: [] for header in output_headers}
                    temp_neg_prompt = {header: [] for header in output_headers}

                    # Read each row and extract the relevant data
                    for row in tsv_reader:

                        for header in output_headers:      

                              
                            main_column = header.strip()  # The main column to look for (e.g., BUG)
                            

                            # Check both possible complementary columns (e.g., BUG_N and BUG_n)
                            comp_column_upper = f"{main_column}_N"  # Uppercase _N
                            comp_column_lower = f"{main_column}_n"  # Lowercase _n

                            # Get the values from the row
                            main_value = str(row.get(main_column, "") or "").strip()
                            comp_value = row.get(comp_column_upper, "").strip() if comp_column_upper in row else \
                                         row.get(comp_column_lower, "").strip() if comp_column_lower in row else ""

                            # If the main value has data, add both the main and complementary values
                            if main_value:
                                temp_master[main_column].append(main_value)
                                temp_neg_prompt[main_column].append(comp_value if comp_value else "")
                            # If there's no main value, don't add the row (neither the main nor complementary column)
                            # Therefore, do nothing (skip adding to temp_master and temp_neg_prompt)
                    # Add the extracted data to the master lists
                    for header in output_headers:
                        print("HERE5")
                        master_list.extend(temp_master[header])
                        neg_prompt.extend(temp_neg_prompt[header])


            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue
            except Exception as e:
                print(f"Error processing file {file_path}: {e} ")
                continue

        if len(master_list) == 0:
            return ("NONE", "NONE", 0, "NA", "NA", "NA", 0,)

        # Ensure the seed value is within bounds of the master list
        seed_idx = seed % len(master_list) if len(master_list) > 0 else 0

        # Fetch the relevant data for the prompt at the seed index
        pos_prompt = master_list[seed_idx]
        neg_prompt_value = neg_prompt[seed_idx]
        
        # Retrieve the last valid file's folder and name
        tsv_folder, tsv_name = last_file_info if last_file_info else ("", "")

        # Determine the header (use the first header in the list by default)
        header = output_headers[0]

        # Return the results as required
        return (pos_prompt, neg_prompt_value, seed, tsv_folder, tsv_name, header, len(master_list),)
    
class  BKIsVerticalImage:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "width": ("INT", {"default": 0,"min":0,"tooltip": "Image width."}),
                "height": ("INT", {"default": 0,"min":0,"tooltip": "Image height."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, width, height):
        return float("nan")
    
    RETURN_TYPES = ("BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("is_a_greater_than_b",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Print To Console With Border"  # Default label text
    OUTPUT_NODE = True

    def process(self, width, height):
        return ((height > width),)
    
class  BKIsAGreaterThanBINT:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "a": ("INT", {"default": 0,"min":0,"tooltip": "A"}),
                "b": ("INT", {"default": 0,"min":0,"tooltip": "B"}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, a, b):
        return float("nan")
    
    RETURN_TYPES = ("BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("is_a_greater_than_b",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Is A Greater Than B INT"  # Default label text
    OUTPUT_NODE = True

    def process(self, a, b):
        return ((a > b),)

class  BKIsAGreaterThanOrEqualToBINT:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "a": ("INT", {"default": 0,"min":0,"tooltip": "A"}),
                "b": ("INT", {"default": 0,"min":0,"tooltip": "B"}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, a, b):
        return float("nan")
    
    RETURN_TYPES = ("BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("is_a_greater_than_or_equal_to_b",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Is A Greater Than Or Equal To B INT"  # Default label text
    OUTPUT_NODE = True

    def process(self, a, b):
        return ((a >= b),)
    
class  BKIsALessThanOrEqualToBINT:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "a": ("INT", {"default": 0,"min":0,"tooltip": "A"}),
                "b": ("INT", {"default": 0,"min":0,"tooltip": "B"}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, a, b):
        return float("nan")
    
    RETURN_TYPES = ("BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("is_a_less_than_or_equal_to_b",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Is A Less Than Or Equal To B INT"  # Default label text
    OUTPUT_NODE = True

    def process(self, a, b):
        return ((a <= b),)
    
class  BKIsALessThanBINT:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "a": ("INT", {"default": 0,"min":0,"tooltip": "A"}),
                "b": ("INT", {"default": 0,"min":0,"tooltip": "B"}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, a, b):
        return float("nan")
    
    RETURN_TYPES = ("BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("is_a_less_than_b",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Print To Console With Border"  # Default label text
    OUTPUT_NODE = True

    def process(self, a, b):
        return ((a < b),)

class  BKPrintToConsoleWithBorder:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text": ("STRING", { "default": f'', "multiline": False, "tooltip": "This is the text to go inbetween the boarders"}),
                "char": ("STRING", { "default": f'#', "multiline": False, "tooltip": "This is the character that will be used to create the boarder."}),
                "length": ("INT", {"default": 75,"min":0,"tooltip": "This is the number of times the character will be repeated to create the boarder."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, text, char, length):
        return float("nan")
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Print To Console With Border"  # Default label text
    OUTPUT_NODE = True

    def process(self, text, char, length):

        border = ""

        for i in range(length):
            border += char  # Add the string 's' to the result

        result = f"{border}\n{char} {text}\n{border}"

        print(result)

        return (result,)
    
class  BKRemoveAnySentencesWithText:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text": ("STRING", { "default": f'', "multiline": False, "tooltip": "This is the text to go inbetween the boarders"}),
                "searchfor": ("STRING", { "default": f'#', "multiline": False, "tooltip": "This is the character that will be used to create the boarder."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, text, searchfor):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("text","removed_text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Remove Any Sentences With Text"  # Default label text
    OUTPUT_NODE = True

    def process(self, text, searchfor):
        # Split the text into sentences using a simple regex for sentence ending punctuation.
        sentences = re.split(r'(?<=\.)\s+', text)

        # Initialize lists to hold the result and removed sentences
        result_sentences = []
        removed_sentences = []

        # Iterate through each sentence and check if the keyword is in it
        for sentence in sentences:
            if searchfor.lower() in sentence.lower():
                removed_sentences.append(sentence)  # Add to removed sentences if keyword is found
            else:
                result_sentences.append(sentence)  # Add to result if keyword is not found

        # Join the remaining sentences to form the result text
        result = ' '.join(result_sentences)

        # Join the removed sentences to form the removed_text
        removed_text = ' '.join(removed_sentences)

        # Return the result and the removed_text
        return (result, removed_text)



class  BKTSVStringParser:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "seed": ("INT", {"default": 0,"min":0,"tooltip": "This is really the index for the row to extract, I put it as a seed so you can have it auto increment without a new node. But think of it as the idx value."}),
                "tsv_string": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
                "debug_to_console": ("BOOLEAN", {"default": False, "tooltip": "If true, will overwrite images that already exist in the folder, if false, will not overwrite images that already exist."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, seed, tsv_string, debug_to_console):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","STRING","INT","STRING", "INT","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("prompt","caption","lora_string","idx","filename", "total_imgs_count", "status_text")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK TSV String Parser"  # Default label text
    OUTPUT_NODE = True

    def process(self, seed, tsv_string, debug_to_console):
        # Split the TSV string into rows
        rows = []

        if not tsv_string.strip():
            raise ValueError("The TSV string is empty or malformed.")
        
        try:
            reader = csv.DictReader(tsv_string.splitlines(), delimiter='\t')
            rows = list(reader)


            if debug_to_console:
                # Print the raw TSV string for debugging purposes
                print("Raw TSV string:")
                print(repr(tsv_string))  # Use repr to show any hidden characters

            # Validate the seed against the available rows
            if seed >= len(rows):
                raise ValueError(f"Seed value is greater than the number of images to generate, please reset the seed between 0 and {len(rows)-1}")

            # Extract the row based on the seed index
            row = rows[seed]
            prompt = row['PROMPT']
            caption = row['CAPTION']
            filename = row['FILENAME']
            lora_string = row['LORA_STRING']

            current_time = datetime.now().strftime('%I:%M:%S %p')
            status_text = f"Processing [{seed+1} of {len(rows)}] @ [{current_time}]: {filename}"

            return prompt, caption, lora_string, seed, filename, len(rows), status_text

        except Exception as e:
            raise ValueError(f"Error processing TSV string: {e}")
        
class  BKTSVTagReplacer:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "Prompt to replace tags in."}),
                "seed": ("INT", {"default": 0,"min":0,"tooltip": "This is really the index for the row to extract, I put it as a seed so you can have it auto increment without a new node. But think of it as the idx value."}),
                "tsv_string": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "These are the lines that will be used to replace the tag by line number."}),
                "debug_to_console": ("BOOLEAN", {"default": False, "tooltip": "If true, will overwrite images that already exist in the folder, if false, will not overwrite images that already exist."}),
            },
            "optoinal":{
                "negative": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "An optional negative prompt to append to."}),
                "caption": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "Caption text."}),
                "status": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "Status text."}),
            }
        }
    
    @classmethod
    def IS_CHANGED(self, prompt, seed, tsv_string, debug_to_console, negative = "", caption = "", status=""):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","INT","STRING", "INT")  # This specifies that the output will be text
    RETURN_NAMES = ("prompt","caption","status","negative prompt","lora_string","idx","filename", "total_imgs_count")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK TSV Tag Replacer"  # Default label text
    OUTPUT_NODE = True

    def process(self, prompt, seed, tsv_string, debug_to_console, negative = "", caption="", status = ""):
        # Split the TSV string into rows
        rows = []

        if not tsv_string.strip():
            raise ValueError("The TSV string is empty or malformed.")
        
        try:
            # Read the TSV string as a CSV with tab delimiter
            reader = csv.DictReader(tsv_string.splitlines(), delimiter='\t')
            rows = list(reader)

            # Validate if the rows are empty
            if len(rows) == 0:
                raise ValueError("No data found in the TSV string.")

            # If debugging, print the raw TSV string
            if debug_to_console:
                print("Raw TSV string:")
                print(repr(tsv_string))

            # Get the header (tags) from the first row
            header = rows[0].keys()

            # Handle seed wrapping (if seed exceeds the available rows, wrap around)
            seed = seed % len(rows)

            # Extract the row based on the seed index
            row = rows[seed]

            # Initialize status text with basic information
            status.strip()
            current_time = datetime.now().strftime('%I:%M:%S %p')
            status += f"\nProcessing [{seed + 1} of {len(rows)}] @ [{current_time}]"

            # Initialize an empty dictionary to store tag values
            tag_values = {}

            # Replace the tags in the prompt with corresponding values from the row
            for tag in header:
                if tag and tag.strip() and tag.upper() not in ['CAPTION', 'FILENAME', 'LORA_STRING']:  # Skip special cases and empty tags
                    tag_value = row.get(tag, '')
                    if tag_value:  # Only replace if the tag has a non-empty value
                        prompt = prompt.replace(tag, tag_value)
                        tag_values[tag] = tag_value
                        # Append this tag and its value to the status text
                        status += f"\n{tag}:[{tag_value}]"

            # Extract necessary data for caption, filename, and lora_string (empty if not found)
            caption_value = row.get('CAPTION', '')  # Return empty string if 'CAPTION' column is missing
            filename = row.get('FILENAME', '')  # Return empty string if 'FILENAME' column is missing
            lora_string = row.get('LORA_STRING', '')  # Return empty string if 'LORA_STRING' column is missing
            negative_value = row.get('NEGATIVE', '')  # Return empty string if 'NEGATIVE' column is missing

            # If caption_value is not empty, append it to the caption
            if caption_value:
                caption += f", {caption_value}"
            
            negative += f"{negative_value}"

            # Append FILENAME to status_text
            if filename:
                status += f"\nFILENAME:[{filename}]"

            # Return the processed data
            return prompt, caption, status, negative, lora_string, seed, filename, len(rows)

        except Exception as e:
            raise ValueError(f"Error processing TSV string: {e}")


class  BKAITextParser:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "ai_text": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "Prompt to replace tags in."}),
                "tsv_text": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "Prompt to replace tags in."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, ai_text, tsv_text = ""):
        return float("nan")
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("tsv_text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK AI Text Parser"  # Default label text
    OUTPUT_NODE = True

    def process(self, ai_text, tsv_text = ""):
        # --- Parse ai_text into headers, keeping only first occurrence ---
        ai_headers = {}
        for line in ai_text.splitlines():
            if ":" not in line:
                continue
            header, value = line.split(":", 1)
            header = header.strip().upper()
            value = value.strip()
            if header not in ai_headers:  # only keep first occurrence
                ai_headers[header] = value

        # --- Parse TSV into rows ---
        if tsv_text:
            rows = [r.split("\t") for r in tsv_text.strip().split("\n")]
        else:
            rows = [[]]

        header_row = rows[0]

        # --- Ensure all ai_text headers exist in TSV header row ---
        for h in ai_headers.keys():
            if h not in header_row:
                header_row.append(h)

        # --- Prepare new row with correct number of columns ---
        num_cols = len(header_row)
        new_row = [""] * num_cols

        # For each header in the TSV, fill from ai_headers or empty
        for idx, h in enumerate(header_row):
            if h in ai_headers:
                new_row[idx] = ai_headers[h]
            else:
                new_row[idx] = ""

        # --- Append new row ---
        rows.append(new_row)

        # --- Reconstruct TSV ---
        new_tsv = "\n".join("\t".join(r) for r in rows)

        return (new_tsv,)

class  BKTSVHeaderFormatter:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "tsv_text": ("STRING", {"forceInput": True, "default": f'', "multiline": False, "tooltip": "Prompt to replace tags in."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(self, tsv_text):
        return float("nan")
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("tsv_text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK TSV Header Formatter"  # Default label text
    OUTPUT_NODE = True

    def process(self, tsv_text):
        # Split the TSV text into rows
        rows = [r.split("\t") for r in tsv_text.strip().split("\n")]

        # If the TSV is empty, initialize with an empty row
        if not rows:
            rows = [[]]

        # Get the header row (first row)
        header_row = rows[0]

        # Predefined columns that should not have '<' and '>' around them
        required_columns = ["NEGATIVE", "LORA_STRING", "FILENAME", "CAPTION"]

        # --- Add '<' and '>' to headers except for the predefined columns ---
        updated_header_row = []
        for header in header_row:
            # If it's one of the predefined columns, don't add '<' and '>'
            if header in required_columns:
                updated_header_row.append(header)
            else:
                updated_header_row.append(f"<{header}>")

        # --- Ensure that the predefined columns are added if they are not already present ---
        for col in required_columns:
            if col not in updated_header_row:
                updated_header_row.append(col)

        # --- Update the rows with the same number of columns ---
        updated_rows = [updated_header_row]  # First row is the updated header row

        # Now we will loop through each of the rows and ensure each row matches the updated header row
        for row in rows[1:]:  # Skip the first row since it's the header
            new_row = row + [""] * (len(updated_header_row) - len(row))  # Pad with empty strings if needed
            updated_rows.append(new_row)

        # --- Reconstruct the updated TSV text ---
        new_tsv = "\n".join("\t".join(r) for r in updated_rows)

        return (new_tsv,)

        
class BKReplaceAllTags:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "prompt":("STRING", {"forceInput": True,}),
                "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": False}),
                "tag": ("STRING", {"default": f'', "multiline": False, "tooltip": "Tag to replace."}),
                "is_ignore_tag_case": ("BOOLEAN", {"default": True, "multiline": True, "tooltip": "If true, will ignore the case of the tag when parsing. Useful for AI generated text."}),
                "lines": ("STRING", {"default": f'', "multiline": True, "tooltip": "These are the lines that will be used to replace the tag by line number."}), 
            },
            "optional":{
                "caption":("STRING", {"forceInput": True,}),
                "status":("STRING", {"forceInput": True,}),
            }
        }
    
    @classmethod
    def IS_CHANGED(self, prompt, tag, lines, seed, is_ignore_tag_case, status = "", caption = ""):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","STRING","INT","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("prompt","caption","status", "line_number", "line_text")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Replace All Tags"  # Default label text
    OUTPUT_NODE = True

    def case_insensitive_replace(self, text, tag, replacement):
        start = 0
        while start < len(text):
            # Find the next occurrence of the tag, ignoring case
            start = text.lower().find(tag.lower(), start)
            
            if start == -1:
                break
            
            # Replace that occurrence with the replacement, preserving the original case
            end = start + len(tag)
            text = text[:start] + replacement + text[end:]
            
            # Move the start index to just after the replacement
            start += len(replacement)
        
        return text

    def process(self, prompt, tag, lines, seed, is_ignore_tag_case, status = "", caption = ""):
        # Split the input string into a list of lines
        line_list = lines.split('\n')

        if not status:
            status = ""

        # Calculate the valid index based on the rules
        if seed >= len(line_list):
            idx = seed % len(line_list)
        else:
            idx = seed

        # Extract the appropriate line
        line = line_list[idx]

        # Replace the tag in the prompt with the extracted line
        if is_ignore_tag_case:
            updated_prompt = self.case_insensitive_replace(prompt, tag, line)
            updated_caption = self.case_insensitive_replace(caption, tag, line)
        else:
            updated_prompt = prompt.replace(tag, line)
            updated_caption = caption.replace(tag,line)

        status += f"{tag}:[{line}]\n"

        return (updated_prompt,updated_caption,status,idx,line)
    
##################################################################################################################
# BK TSV RANDOM PROMPT
##################################################################################################################
class BKTSVRandomPrompt:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.name = ""
        self.is_debug = False
        self.name_column_suffix = "_name"
        self.negative_column_suffix = "_negative"

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "tsv_file_path": ("STRING",),
                "prompt_column":("STRING", {"default": "prompts"}),
                "prompt_name_search": ("STRING",{"default": "", "tooltip":"Will select prompts that start with the name, and randomly choose one. If a unique name is found, it will use just the prompt with that unique name. If it fails to find the prompt by name, will fall back to using the index."}),
                "seed": ("INT",),
                "advanced_control": ("STRING", {"multiline": True, "default": f"", "tooltip": "<header>:set:<string> => Sets all instances of the header to a specific value. <header>:idx:<int> => sets all instances of header to use a specific idx in column, <header>:match:<header> => use same random idx when replacing header tags in the prompt for both columns (will roll over if idx is larger than items in column)"})
            },
            "optional":{
            }
        }
    
    @classmethod
    def IS_CHANGED(self, tsv_file_path, prompt_column, seed, advanced_control = None, prompt_name = None):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","STRING","INT","INT")  # This specifies that the output will be text
    RETURN_NAMES = ("pos_prompt", "neg_prompt", "name", "seed", "idx")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK TSV Random Prompt"  # Default label text
    OUTPUT_NODE = True

    def is_a_name_column(self, value: str) -> bool:

        if not isinstance(value, str):
            return False

        return value.endswith(self.name_column_suffix)

    def sanatize_all_name_columns_for_case_insensitve_searching(self, dataframe):
        # Check if dataframe is a pandas DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected input to be a pandas DataFrame, but got a {}.".format(type(dataframe)))
        
        # Loop through columns and apply string methods only to the relevant ones
        for column in dataframe.columns:
            if self.is_a_name_column(column):
                if dataframe[column].dtype == 'object':  # Check if the column contains string data
                    dataframe[column] = dataframe[column].str.strip().str.lower()
        
        return dataframe
        
    def sanatize_name_for_case_insensitive_searching(self, name):
        return name.strip().lower()


    def process(self, tsv_file_path, prompt_column, seed, advanced_control=None, prompt_name_search = None):
        self.name = ""

        if not tsv_file_path:
            raise ValueError("TSV file path is empty.")
        
        if not prompt_column:
            raise ValueError("No prompt column name specified. Need to know which column the prompts are in.")

        tsv_file_path = self.convert_relative_path_to_abs(self.output_dir, tsv_file_path)

        # Now load the new file with pandas
        dataframe = self.read_file_to_dataframe(tsv_file_path)

        if dataframe.empty:
            raise ValueError("The file loaded as empty. Is there data in the file?")

        # sanatize names for case insenstive compare
        if prompt_name_search:
            prompt_name_search = self.sanatize_name_for_case_insensitive_searching(prompt_name_search)
        prompt_column = self.sanatize_name_for_case_insensitive_searching(prompt_column)

        # Ensure prompt_column exists in the TSV file
        if prompt_column not in dataframe.columns:
            raise ValueError(f"The prompt column '{prompt_column}' was not found in TSV file [{tsv_file_path}].")
        
        dataframe = self.sanatize_headers_for_case_insensitve_searching(dataframe)
        dataframe = self.sanatize_all_name_columns_for_case_insensitve_searching(dataframe)

        

        prompts = None
        idx = -1

        # If a name was provided get the prompt by name
        if prompt_name_search:
            prompt_name_column = self.get_name_column_name(prompt_column)
            indexes_of_names = self.get_idx_list_of_names_starting_with_name(dataframe, prompt_name_column, prompt_name_search)
            
            if indexes_of_names.any():
                idx_list_of_prompts_that_start_with_name = len(indexes_of_names)
                idx = self.convert_seed_to_idx(idx_list_of_prompts_that_start_with_name, seed)
                idx_of_prompt_that_starts_with_name = indexes_of_names[idx]
                prompts = self.get_dataset_for_name_at_idx(dataframe, prompt_column, idx_of_prompt_that_starts_with_name)
            
            else:
                self.print_debug(f"No prompts found that start with the name [{prompt_name_search}], defaulting to using the seed selection.")

        if not prompts:
            num_of_rows = self.get_total_num_of_rows_in_column(dataframe, prompt_column)
            self.print_debug(f"num_of_rows[{num_of_rows}]")
            idx = self.convert_seed_to_idx(num_of_rows, seed)
            prompts = self.get_dataset_for_name_at_idx(dataframe, prompt_column, idx)
        
        # TODO: Make a toggleable option to select if the name of the tags should be output in the name as well or not
        # TODO: Have it so that any tags in the negative prompt are replaced by what is in the <TAG>_negative. So if the prompt_negative has a tag in it, that it will use the negative for that tag
        # TODO: if the negative tag is also in the postive tag, then it will use the same idx for both, if the negative tag is NOT in the positive prompt then it will select the negative tag by idx
        # TODO: Have it throw an error if it can not parse a line on the advanced section with the line number and text displayed
        # TODO: Make option to handle mode if name not found, either default to first image, use seed, or throw error
        # TODO: Add ability to select prompts with specific tags (Implement as Noneable feild), Also allow exclusion of tag field
        # TODO: Allow selection by "Random tag" vs "Output tag". Random tags are tags that will be randomized before the output is produced and are not present in the output, "Output" tag will be output in the prompt after all random tags are replaced
        # TODO: Tag parameters should have be 4D [value, negative, name, tag] this way we can manipulate / search for it by tag also
        # TODO: Prompt array should also have a "Output" and "Random" tag list that is generated for it [prompt, negative, name, [random], [output]]
        # TODO: (Depreciated) Output tags will be generated by using the input "output" tags list and comparing it to the prompt to see if it contains any, if it does, it is added to the list? If this can be calculated with the excel sheet, should it really be a column maybe not. the time it would take to process a list, would probably be the same as just using the headers in the sheet.
        # TODO: Filter by keywords, allow for searching of keywords in the prompt such as "pink" or "standing"

        pos_prompt, neg_prompt, name  = prompts

        return pos_prompt, neg_prompt, name, seed, idx
    
    def value(self, data_set):
        return data_set[2]

    def name(self, data_set):
        return data_set[0]
    
    def negative(self, data_set):
        return data_set[1]

    def strip_suffixs_from_name(self, name_column):
        value_column_name = name_column.removesuffix(self.name_column_suffix)
        value_column_name = name_column.removesuffix(self.negative_column_suffix)
        return value_column_name
    
    def get_value_column_name(self, name):
        return self.strip_suffixs_from_name(name)
    
    def get_negative_column_name(self, name):
        stripped_name = self.strip_suffixs_from_name(name)
        return f"{stripped_name}{self.negative_column_suffix}"
    
    def get_name_column_name(self, name):
        stripped_name = self.strip_suffixs_from_name(name)
        return f"{stripped_name}{self.name_column_suffix}"
    

    
    def get_dataset_for_name_at_idx(self, dataframe, name_any, idx):
        value = self.get_required_column_value_by_idx(dataframe, self.get_value_column_name(name_any), idx)
        negative = self.get_optional_column_value_by_idx(dataframe, self.get_negative_column_name(name_any), idx)
        name = self.get_optional_column_value_by_idx(dataframe, self.get_name_column_name(name_any), idx)
        return (value, negative, name)

    def get_idx_list_of_names_starting_with_name(self, dataframe, name_column, name):


        self.print_debug(f"name_column[{name_column}]")
        self.print_debug(f"name[{name}]")
        # Returns a 1D array of boolean where only matching values (starting with `name`)
        # in the column return a true value

        self.print_debug(f"dataframe[name_column[{dataframe[name_column]}]")
        all_matching_rows_in_column = dataframe[name_column].str.startswith(name, na=False)

        self.print_debug(f"all_matching_rows_in_column[{all_matching_rows_in_column}]")
        
        # Returns a list of indices of the rows where the value starts with `name`
        list_of_matching_idx_for_name_column = dataframe[all_matching_rows_in_column].index
        return list_of_matching_idx_for_name_column
        
    def convert_seed_to_idx(self, length, seed):
        # Use modulus to ensure the seed is within the bounds of the prompt_column length

        self.print_debug(f"length[{length}]")
        self.print_debug(f"seed[{seed}]")

        if seed == 0:
            return 0

        return seed % length  # Adjust seed to be within valid range for the column

    
    def convert_relative_path_to_abs(self, base_folder, tsv_file_path):
       
        tsv_file_path = tsv_file_path.strip('"')

        # Ensure the TSV file path is absolute
        if not os.path.isabs(tsv_file_path):
            tsv_file_path = os.path.join(base_folder, tsv_file_path)
        
        return tsv_file_path
        
    def read_file_to_dataframe(self, tsv_file_path):
        return pd.read_csv(self.read_file_to_utf8_string(tsv_file_path), sep='\t')
    
    def get_total_num_of_rows_in_column(self, dataframe, prompt_column):
        self.print_debug(f"prompt_column[{prompt_column}]")
        return len(dataframe[prompt_column].dropna())


    def get_required_column_value_by_idx(self, dataframe, column, idx):
        return dataframe.iloc[idx][column]


    def get_optional_column_value_by_idx(self, dataframe, column, idx):
        if column in dataframe.columns and 0 <= idx < len(dataframe):
            value = dataframe.iloc[idx][column]
            return "" if pd.isna(value) else value
        return ""

    def sanatize_headers_for_case_insensitve_searching(self, dataframe):
        # Loop through all columns in the DataFrame and sanitize the header one by one
        sanitized_columns = []
        for column in dataframe.columns:
            # Sanitize: strip whitespace and convert to lowercase
            sanitized_column = column.strip().lower()
            sanitized_columns.append(sanitized_column)
        
        # Reassign sanitized column names back to the DataFrame
        dataframe.columns = sanitized_columns
        return dataframe
    
    def print_debug(self, string):
        if self.is_debug:
            print (f"{string}")

    def open_file_as_read_only(self, tsv_file_path):
        content = None
        # Read the file in binary mode
        try:
            with open(tsv_file_path, 'rb') as f:
                content = f.read()
        except Exception as e:
            raise Exception(f"Failed to open the TSV file. Is it open in a program? Reason: {e}")

        return content

    def read_file_to_utf8_string(self, tsv_file_path):
        file_content = self.open_file_as_read_only(tsv_file_path)

        # Decode the content with 'ignore' for invalid characters
        return StringIO(file_content.decode('utf-8', errors='ignore'))

    
    def append_to_name(self, df, column_name, idx):

        # Generate the column name with "_name" suffix
        column_name_with_suffix = f"{column_name}_name"
        
        # Check if the column exists (case insensitive check)
        matching_columns = [col for col in df.columns if col.lower() == column_name_with_suffix.lower()]
        
        # If the column does not exist, return without doing anything
        if not matching_columns:
            return
        
        # Get the actual column name (the matched one, case-insensitive)
        matching_column = matching_columns[0]
        
        # Get the value at the specified idx in the "_name" column
        value_to_append = df.iloc[idx][matching_column]
        
        if pd.notna(value_to_append):  # Only proceed if the value is not NaN
            # Append the value to self.name with an underscore in between
            self.name = f"{self.name}_{value_to_append}"

        # Strip any leading/trailing spaces or underscores from self.name
        self.name = self.name.strip("_").strip()

    def _process_advanced_control(self, advanced_control, df, pos_prompt, seed):
        commands = advanced_control.splitlines()
        affected_columns = []  # List to track columns affected by advanced control

        for command in commands:
            parts = command.split(':')
            if len(parts) != 3:
                continue

            column_name, instruction, value = parts

            if instruction == "set":
                # Override the replacement for the column
                pos_prompt = pos_prompt.replace(f"{column_name}", value)
                affected_columns.append(column_name)  # Mark column as affected

            elif instruction == "idx":
                # Replace with value at specific index
                try:
                    idx = int(value) % len(df[column_name].dropna())
                    pos_prompt = pos_prompt.replace(f"{column_name}", str(df.iloc[idx][column_name]))
                    self.append_to_name(df,column_name, idx)
                    affected_columns.append(column_name)  # Mark column as affected
                except KeyError:
                    print(f"Error: Column '{column_name}' not found for 'idx' operation.")
                except IndexError:
                    print(f"Error: Index out of range for column '{column_name}' in 'idx' operation.")

            elif instruction == "match":
                # Match the index of one column with another
                other_column = value
                try:
                    random_index = random.randint(0, len(df[column_name]) - 1)
                    idx = self._get_index_for_column(df, column_name, random_index)
                    other_idx = self._get_index_for_column(df, other_column, idx)
                    pos_prompt = pos_prompt.replace(f"{column_name}", str(df.iloc[idx][column_name]))
                    pos_prompt = pos_prompt.replace(f"{other_column}", str(df.iloc[other_idx][other_column]))
                    self.append_to_name(df,column_name, idx)
                    self.append_to_name(df,other_column, other_idx)
                    affected_columns.append(column_name)  # Mark column as affected
                    affected_columns.append(other_column)  # Mark column as affected
                except KeyError:
                    print(f"Error: Column '{other_column}' not found for 'match' operation.")
                except IndexError:
                    print(f"Error: Index out of range for column '{other_column}' in 'match' operation.")

        print(f"pos_prompt[{pos_prompt}]")
        return pos_prompt, affected_columns

    def _get_index_for_column(self, df, column_name, seed):
        try:
            return seed % len(df[column_name].dropna())
        except KeyError:
            print(f"Error: Column '{column_name}' not found.")
            return -1

    def _get_related_column_value(self, df, prompt_column, seed, suffix, default_value):
        related_column = self._get_related_column(df, prompt_column, suffix)
        if related_column:
            try:
                # Return the value at the seed index if available, else empty string
                return df.iloc[seed][related_column] if pd.notna(df.iloc[seed][related_column]) else ""
            except IndexError:
                print(f"Error: Index out of range for column '{related_column}'.")
        return default_value

    def _get_related_column(self, df, prompt_column, suffix):
        # Case-insensitive search for a column name with suffix (_neg or _name)
        for column in df.columns:
            if column.lower() == f"{prompt_column.lower()}{suffix.lower()}":
                return column
        return None
    


##################################################################################################################
# BK MAX SIZE
##################################################################################################################
class BKMaxSize:
    def __init__(self):
        self.output_dir = folder_paths.output_directory


    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "image_width": ("INT", {"default": 512,"tooltip": "Current width of image."}),
                "image_height": ("INT", {"default": 512,"tooltip": "Current height of image."}),
                "max_width": ("INT", {"default": 512,"tooltip": "Maximum desired width."}),
                "max_height": ("INT", {"default": 512,"tooltip": "Maximum desired height."}),
      
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("INT","INT","FLOAT",)  # This specifies that the output will be text
    RETURN_NAMES = ("calc_width", "calc_height", "percent")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Max Size"  # Default label text
    OUTPUT_NODE = True

    def process(self, image_height, image_width, max_height, max_width, extra_pnginfo=None):
        # Calculate the aspect ratio
        aspect_ratio = image_width / image_height

        # If the image's height is greater than the maximum allowed height, adjust it
        if image_height > max_height:
            calc_height = max_height
            calc_width = int(calc_height * aspect_ratio)  # Recalculate width to maintain the aspect ratio
        else:
            calc_height = image_height
            calc_width = image_width

        # If the image's width is greater than the maximum allowed width, adjust it
        if calc_width > max_width:
            calc_width = max_width
            calc_height = int(calc_width / aspect_ratio)  # Recalculate height to maintain the aspect ratio

        scale_percent = float(calc_width / image_width)

        return (calc_width, calc_height, scale_percent)
    
class BKFileSelector:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "seed": ("INT", {"default": 0,"tooltip": "The index of the file you want to extract."}),
                "search_pattern": ("STRING", {"default": "*.png","tooltip": "The search pattern to use when searching for files in the folder. Use '*' as a wild card. If you want to search an extension do '*.ext', or for a file that starts with a name 'prefix*', or a prefix and an extension 'prefix*.png'. You can use more than one '*' as a wild card. (Uses Python Glob)"}),
                "source_path": ("STRING", {"default": "","tooltip": "The absolute path of the folder you wish to search through. (Can be anywhere.)"}),
            }
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","INT","STRING","STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("filename","folder_path", "extension", "full_path", "total_files_found","file_list", "file_list_w_idx", "status",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK File Selector"  # Default label text
    OUTPUT_NODE = True

    def process(self, seed, search_pattern, source_path):

        if not os.path.isabs(source_path):
            # If relative, construct the path with the output directory
            source_path = f"{self.output_dir.strip()}\\{source_path}"

        # Construct the full search pattern (adding folder_path)
        search_pattern = os.path.join(source_path, search_pattern)

        # Find all files that match the search pattern
        matching_files = glob.glob(search_pattern)

        # Sort the files alphabetically
        matching_files.sort()

        # Get the total number of files found
        total_files_found = len(matching_files)

        # Check if the index is valid
        if seed < 0 or seed >= total_files_found:
            raise IndexError("Index is out of range")

        # Extract the file at the specified index
        full_path = matching_files[seed]
        filename_with_extension = os.path.basename(full_path)
        filename, extension = os.path.splitext(filename_with_extension)

        # Create the list of files as a string (one filename per line, without extension)
        file_list = "\n".join([os.path.splitext(os.path.basename(file))[0] for file in matching_files])

        # Create the list of files with index as a string (index:filename without extension)
        file_list_w_idx = "\n".join([f"{i}:{os.path.splitext(os.path.basename(file))[0]}" for i, file in enumerate(matching_files)])

        current_time = datetime.now().strftime('%I:%M:%S %p')
        status = f"Processing [{seed + 1} of {total_files_found}] @ [{current_time}]: {filename}"

        # Return the requested information
        return (filename, source_path, extension, full_path, total_files_found, file_list, file_list_w_idx, status)

##################################################################################################################
# BK Move File
##################################################################################################################

class BKMoveOrCopyFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.is_debug = True

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "mode": (["Skip if exists", "Overwrite", "Rename if exists"], {"default": "Skip if exists" ,"tooltip": "How to handle if the destination file already exists."}),
                "operation": (["Move", "Copy"], {"default": "Copy", "tooltip": "Whether to move or copy the file."}),
                "src_folder": ("STRING", {"default": "","tooltip": "Absolute folder path, or will apply relative path to inside of output folder."}),
                "src_filename": ("STRING", {"default": "filename","tooltip": "Filename of source file."}),
                "src_extension": ("STRING", {"default": "png","tooltip": "If empty, will keep same extension as source file. Else will change to this extension."}),
                "dest_folder": ("STRING", {"default": "moved","tooltip": "Absolute folder path, or will apply relative path to inside of src folder path."}),
                "change_filename_to": ("STRING", {"default": "","tooltip": "Filename of destination file."}),
                "change_extension_to": ("STRING", {"default": "","tooltip": "If empty, will keep same extension as source file. Else will change to this extension."}),
                "is_move_file": ("BOOLEAN",{"default": True}),
                
            }
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("dest_filename","dest_absolute_folder", "dest_extension","dest_file_path", "status")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Move Or Copy File"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, src_filename, src_folder, src_extension, change_filename_to, dest_folder, change_extension_to, is_move_file, operation = "Copy", mode = "Overwrite"):
        return float("nan")
    
    def get_dest_file_name(self, filename, src_filename):
        if not filename:
            return src_filename
        else:
            return filename
    
    def process(self, src_filename, src_folder, src_extension, change_filename_to, dest_folder, change_extension_to, is_move_file, operation = "Copy", mode = "Overwrite"):
        print_debug_header(self.is_debug, "BK MOVE OR COPY FILE")
        
        if not src_filename:
            raise ValueError("Source filename is empty. Please provide a valid source filename.")
        
        if not src_extension:
            raise ValueError("Source extension is empty. Please provide a valid source file extension.")
        
        if not src_folder:
            raise ValueError("Source folder is empty. Please provide a valid source folder path.")


        src_folder = os.path.normpath(src_folder)
        if dest_folder:
            dest_folder = os.path.normpath(dest_folder)

        src_abs_folder = self.get_abs_folder(src_folder, self.output_dir)
        src_file_path = self.get_file_path(src_abs_folder, src_filename, src_extension)

        change_filename_to = self.get_dest_file_name(change_filename_to, src_filename)

        dest_extension = self.get_dest_ext(src_extension, change_extension_to)

        dest_abs_folder = self.get_abs_folder(dest_folder, src_abs_folder)
        dest_file_path = self.get_file_path(dest_abs_folder, change_filename_to, dest_extension)

        status = f"Moved file {src_file_path} to {dest_file_path}."

        if not is_move_file:
            status = self.set_status("is_move_file is set to False, skipping move operation.")
            return (change_filename_to, dest_abs_folder, dest_extension,dest_file_path, status)
        
        # Check if the source file exists before attempting to move it
        if self.is_file_missing(src_file_path):
            raise Exception(f"Source file {src_file_path} not found. Please check the path.")
        
        if self.is_file_exists(dest_file_path):
            if mode == "Skip if exists":
                status = self.set_status(f"Destination file {dest_file_path} already exists. Mode set to 'Skip if exists'. Skipping move operation.")
                return (change_filename_to, dest_abs_folder, dest_extension,dest_file_path, status)
            
            elif mode == "Rename if exists":
                new_dest_file_path = self.rename_file_if_exists(dest_file_path)
                status = self.set_status(f"Destination file [{dest_file_path}] already exists. Mode set to 'Rename if exists'. Renaming to [{new_dest_file_path}].")
                dest_file_path = new_dest_file_path
            
            else:
                status = self.set_status(f"Destination file [{dest_file_path}] already exists. Mode set to 'overwrite'. Overwriting file.")
                dest_file_path = dest_file_path


        try:
            if dest_file_path == src_file_path:
                status = self.set_status(f"Source file and destination file are the same [{src_file_path}]. No move needed.")
                return (change_filename_to, dest_abs_folder, dest_extension, dest_file_path, status)

            self.create_folder_if_missing(dest_abs_folder)

            if operation == "Copy":
                self.copy_file(src_file_path, dest_file_path)
            elif operation == "Move":
                self.move_file(src_file_path, dest_file_path)
            else:
                raise ValueError(f"Invalid operation mode: {mode}. Supported modes are 'Move' and 'Copy'.")

            print_debug_bar(self.is_debug)
            return (change_filename_to, dest_abs_folder, dest_extension, dest_file_path, status)
        except FileNotFoundError as e:
            raise Exception(f"Source file {src_file_path} not found. - [{e}]")
        except PermissionError as e:
            raise Exception(f"Permission denied while moving {src_file_path}. - [{e}]")
    
    def change_file_extension(self, filename, new_extension):
        base = os.path.splitext(filename)[0]
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension
        return base + new_extension
    
    def set_relative_path_to_output_folder(self, path):
        return os.path.join(self.output_dir, path)
    
    def get_absolute_folder_path(self, root_folder, path):
        return os.path.join(root_folder, path)

    def is_relative_path(self, path):
        return not os.path.isabs(path)
    
    def get_file_path(self, folder, filename, extension):
        return os.path.join(folder, f"{filename.strip(".")}.{extension.lower().strip(".")}")
    
    def is_dest_ext_specified(self, dest_extension):
        return dest_extension not in ("", None)

    def get_dest_ext(self, src_ext, dest_ext):
        if self.is_dest_ext_specified(dest_ext):
            return dest_ext.lower().strip(".")
        else:
            return src_ext.lower().strip(".")
    
    def get_abs_folder(self, path, root_folder):
        if self.is_relative_path(path):
            abs_folder = self.get_absolute_folder_path(root_folder, path)
        else:
            abs_folder = path

        return abs_folder

    def print_debug(self, string):
        if self.is_debug:
            print (f"{string}")

    def is_file_missing(self, file_path):
        return not os.path.isfile(file_path)
    
    def is_file_exists(self, file_path):
        return os.path.isfile(file_path)
    
    def set_status(self, status):
        print(f"BKMoveFile: {status}")
        return status
    
    def get_folder_path(self, file_path):
        return os.path.dirname(file_path)
    
    def create_folder_if_missing(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            self.print_debug(f"Created missing folder: {folder_path}")

    def move_file(self, src_path, dest_path):
        os.rename(src_path, dest_path)
        self.print_debug(f"Moved file from [{src_path}] to [{dest_path}]")

    def copy_file(self, src_path, dest_path):
        shutil.copy(src_path, dest_path)
        self.print_debug(f"Copied file from [{src_path}] to [{dest_path}]")

    def rename_file_if_exists(self, dest_file_path):
        base, extension = os.path.splitext(dest_file_path)
        counter = 1
        new_dest_file_path = f"{base}_{counter}{extension}"
        while os.path.isfile(new_dest_file_path):
            counter += 1
            new_dest_file_path = f"{base}_{counter}{extension}"
        return new_dest_file_path

    

        
class BKFileSelectNextMissing:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "search_pattern": ("STRING", {"default": "*.png","tooltip": "The search pattern to use when searching for files in the folder. Use '*' as a wild card. If you want to search an extension do '*.ext', or for a file that starts with a name 'prefix*', or a prefix and an extension 'prefix*.png'. You can use more than one '*' as a wild card. (Uses Python Glob)"}),
                "source_path": ("STRING", {"default": "","tooltip": "The source path is the relative path to the folder with the files you wish to modify. Can use absolute or relative path, if relative, starts in output folder."}),
                "dest_path": ("STRING", {"default": "","tooltip": "The node will scan the source path folder for any of the specified files, it will then return the next file that is in the source path, but not in the destination path. ex. If the source path has a file called 'image.png', and the destination folder does not, it will output the info for the 'image.png' file found in the source folder. It will keep doing this until all files in the source folder are found in the destination folder. Can use absolute or relative path, if relative, starts in output folder."}),
            }
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","INT","STRING","STRING","STRING","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("filename","folder_path", "extension", "full_path", "num_of_remaining_files","file_list", "file_list_w_idx", "status", "dest_folder")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK File Select Next Missing"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, search_pattern, source_path, dest_path):
        return float("nan")

    def process(self, search_pattern, source_path, dest_path):

        # Check if the provided path is absolute or relative
        if not os.path.isabs(source_path):
            # If relative, construct the path with the output directory
            source_path = f"{self.output_dir.strip()}\\{source_path}"

                # Check if the provided path is absolute or relative
        if not os.path.isabs(dest_path):
            # If relative, construct the path with the output directory
            dest_path = f"{self.output_dir.strip()}\\{dest_path}"

        # Construct the full search pattern (adding folder_path)
        search_pattern = os.path.join(source_path, search_pattern)

        # Find all files that match the search pattern
        matching_files = glob.glob(search_pattern)

        if not os.path.exists(dest_path.strip()):
            print(f'The path `{dest_path.strip()}` specified doesn\'t exist! Creating directory.')
            os.makedirs(dest_path, exist_ok=True) 

        #ADD CODE HERE
        # Get the list of file names in the dest_folder
        dest_files = os.listdir(dest_path)
        
        # Extract the base names (without extensions) from the files in dest_folder
        dest_base_names = [os.path.splitext(file)[0] for file in dest_files]

        # Check each file in matching_files and remove it if its name starts with any in dest_base_names
        matching_files = [
            file for file in matching_files
            if not any(dest_base_name.startswith(os.path.splitext(os.path.basename(file))[0]) for dest_base_name in dest_base_names)
        ]

        # If no images are found, raise an error
        if not matching_files:
            raise ValueError("No files found in the source folder that are not found in the destination folder.")

        # Sort the files alphabetically
        matching_files.sort()

        # Get the total number of files found
        total_files_found = len(matching_files)

        # Extract the file at the specified index
        full_path = matching_files[0]
        filename_with_extension = os.path.basename(full_path)
        filename, extension = os.path.splitext(filename_with_extension)

        # Create the list of files as a string (one filename per line, without extension)
        file_list = "\n".join([os.path.splitext(os.path.basename(file))[0] for file in matching_files])

        # Create the list of files with index as a string (index:filename without extension)
        file_list_w_idx = "\n".join([f"{i}:{os.path.splitext(os.path.basename(file))[0]}" for i, file in enumerate(matching_files)])

        current_time = datetime.now().strftime('%I:%M:%S %p')
        status = f"Processing [{filename}] {total_files_found - 1} remaining. [{current_time}]"

        # Return the requested information
        return (filename, source_path, extension, full_path, (total_files_found - 1), file_list, file_list_w_idx, status, dest_path)
    
class BKFileSelectNextUnprocessed:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "search_pattern": ("STRING", {"default": "*.png","tooltip": "The search pattern to use when searching for files in the folder. Use '*' as a wild card. If you want to search an extension do '*.ext', or for a file that starts with a name 'prefix*', or a prefix and an extension 'prefix*.png'. You can use more than one '*' as a wild card. (Uses Python Glob)"}),
                "source_path": ("STRING", {"default": "","tooltip": "The source path is the relative path to the folder with the files you wish to modify. Can use absolute or relative path, if relative, starts in output folder."}),
                "dest_path": ("STRING", {"default": "","tooltip": "The node will scan the source path folder for any of the specified files, it will then return the next file that is in the source path, but not in the destination path. ex. If the source path has a file called 'image.png', and the destination folder does not, it will output the info for the 'image.png' file found in the source folder. It will keep doing this until all files in the source folder are found in the destination folder. Can use absolute or relative path, if relative, starts in output folder."}),
            }
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","INT","STRING","STRING","STRING","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("filename","folder_path", "extension", "full_path", "num_of_remaining_files","file_list", "file_list_w_idx", "status", "dest_folder")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK File Select Next Unprocessed"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, search_pattern, source_path, dest_path = ""):
        return float("nan")

    def process(self, search_pattern, source_path, dest_path = ""):

        # Check if the provided path is absolute or relative
        if not os.path.isabs(source_path):
            # If relative, construct the path with the output directory
            source_path = f"{self.output_dir.strip()}\\{source_path}"

        # Construct the full search pattern (adding folder_path)
        search_pattern = os.path.join(source_path, search_pattern)

        # Find all files that match the search pattern
        matching_files = glob.glob(search_pattern)

        # If a destination path is specified...
        if dest_path.strip():
            # Check if the provided path is absolute or relative
            if not os.path.isabs(dest_path):
                # If relative, construct the path with the output directory
                dest_path = f"{self.output_dir.strip()}\\{dest_path}"

            if not os.path.exists(dest_path.strip()):
                print(f'The path `{dest_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(dest_path, exist_ok=True)

            # Get the list of file names in the dest_folder
            dest_files = os.listdir(dest_path)
            
            # Extract the base names (without extensions) from the files in dest_folder
            dest_base_names = [os.path.splitext(file)[0] for file in dest_files]

            # Check each file in matching_files and remove it if its name starts with any in dest_base_names
            matching_files = [
                file for file in matching_files
                if not any(dest_base_name.startswith(os.path.splitext(os.path.basename(file))[0]) for dest_base_name in dest_base_names)
            ]

        # If no images are found, raise an error
        if not matching_files:
            raise ValueError("No files found in the source folder that are not found in the destination folder.")

        # Sort the files alphabetically
        matching_files.sort()

        log_name = "processLog.txt"

        if dest_path.strip():
            reviewed_file = os.path.join(dest_path, log_name)
        else:
            reviewed_file = os.path.join(source_path, log_name)
        
        print(f"reviewed_file: [{reviewed_file}]")

        # If "reviewed.txt" exists in dest_path, read it
        reviewed_files = set()
        if os.path.exists(reviewed_file):
            with open(reviewed_file, 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if line:  # Ignore empty lines
                        reviewed_files.add(line.lower())  # Store filenames in lowercase

        # Remove files from "matching_files" that are in "reviewed_files" (ignoring file extensions)
        matching_files = [
            file for file in matching_files
            if os.path.splitext(os.path.basename(file))[0].lower() not in reviewed_files
        ]

        # If no files left to process, raise an error
        if not matching_files:
            raise ValueError(f"No files to process after removing reviewed files. If you want to reset review previously reviewed notes, simply delete or edit the '{log_name}' file in the destination folder.")
        
        # Get the total number of files still left in the list after existing, and reviewed files have been removed.
        total_files_found = len(matching_files)

        # Extract the file at the specified index
        full_path = matching_files[0]
        filename_with_extension = os.path.basename(full_path)
        filename, extension = os.path.splitext(filename_with_extension)

        # Append the "filename" to "reviewed.txt" while locking the file during write
        with open(reviewed_file, 'a') as file:
            file.write(f"{filename}\n")

        # Create the list of files as a string (one filename per line, without extension)
        file_list = "\n".join([os.path.splitext(os.path.basename(file))[0] for file in matching_files])

        # Create the list of files with index as a string (index:filename without extension)
        file_list_w_idx = "\n".join([f"{i}:{os.path.splitext(os.path.basename(file))[0]}" for i, file in enumerate(matching_files)])

        current_time = datetime.now().strftime('%I:%M:%S %p')
        status = f"Processing [{filename}] {total_files_found - 1} remaining. [{current_time}]"

        # Return the requested information
        return (filename, source_path, extension, full_path, (total_files_found-1), file_list, file_list_w_idx, status, dest_path)
    
class BKNextUnprocessedFileInFolder:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "search_pattern": ("STRING", {"default": "*.png","tooltip": "The search pattern to use when searching for files in the folder. Use '*' as a wild card. If you want to search an extension do '*.ext', or for a file that starts with a name 'prefix*', or a prefix and an extension 'prefix*.png'. You can use more than one '*' as a wild card. (Uses Python Glob)"}),
                "folder": ("STRING", {"default": "","tooltip": "The node will scan the source path folder for any of the specified files, it will then return the next file that is in the source path, but not in the destination path. ex. If the source path has a file called 'image.png', and the destination folder does not, it will output the info for the 'image.png' file found in the source folder. It will keep doing this until all files in the source folder are found in the destination folder. Can use absolute or relative path, if relative, starts in output folder."}),
            }
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","INT","STRING","STRING","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("filename","folder_path", "extension", "full_path", "num_of_remaining_files","file_list", "file_list_w_idx", "status")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Next Unprocessed File In Folder"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, search_pattern, folder):
        return float("nan")

    def process(self, search_pattern, folder):

        # Check if the provided path is absolute or relative
        if not os.path.isabs(folder):
            # If relative, construct the path with the output directory
            folder = f"{self.output_dir.strip()}\\{folder}"

        if not os.path.exists(folder.strip()):
                ValueError(f'The path `{folder.strip()}` specified doesn\'t exist!')

        # Construct the full search pattern (adding folder_path)
        search_pattern = os.path.join(folder, search_pattern)

        # Find all files that match the search pattern
        matching_files = glob.glob(search_pattern)

        # If no images are found, raise an error
        if not matching_files:
            raise ValueError(f"No files found in `{folder.strip()}` folder.")

        # Sort the files alphabetically
        matching_files.sort()

        log_name = "processLog.txt"

        reviewed_file = os.path.join(folder, log_name)
        
        print(f"reviewed_file: [{reviewed_file}]")

        # If "reviewed.txt" exists read it
        reviewed_files = set()
        if os.path.exists(reviewed_file):
            with open(reviewed_file, 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if line:  # Ignore empty lines
                        reviewed_files.add(line.lower())  # Store filenames in lowercase

        # Remove files from "matching_files" that are in "reviewed_files" (ignoring file extensions)
        matching_files = [
            file for file in matching_files
            if os.path.splitext(os.path.basename(file))[0].lower() not in reviewed_files
        ]

        # If no files left to process, raise an error
        if not matching_files:
            raise ValueError(f"No files to process after removing reviewed files. If you want to reset review previously reviewed notes, simply delete or edit the '{reviewed_file}' file in the destination folder.")
        
        # Get the total number of files still left in the list after existing, and reviewed files have been removed.
        total_files_found = len(matching_files)

        # Extract the file at the specified index
        full_path = matching_files[0]
        filename_with_extension = os.path.basename(full_path)
        filename, extension = os.path.splitext(filename_with_extension)

        # Append the "filename" to "reviewed.txt" while locking the file during write
        with open(reviewed_file, 'a') as file:
            file.write(f"{filename}\n")

        # Create the list of files as a string (one filename per line, without extension)
        file_list = "\n".join([os.path.splitext(os.path.basename(file))[0] for file in matching_files])

        # Create the list of files with index as a string (index:filename without extension)
        file_list_w_idx = "\n".join([f"{i}:{os.path.splitext(os.path.basename(file))[0]}" for i, file in enumerate(matching_files)])
        remaining_files = total_files_found - 1

        current_time = datetime.now().strftime('%I:%M:%S %p')
        status = f"Processing [{filename}] {remaining_files} remaining. [{current_time}]"

        # Return the requested information
        return (filename, folder, extension, full_path, remaining_files, file_list, file_list_w_idx, status)
    
class BKNextUnprocessedImageInFolder:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "folder": ("STRING", {"default": "","tooltip": "The node will scan the source path folder for any of the specified files, it will then return the next file that is in the source path, but not in the destination path. ex. If the source path has a file called 'image.png', and the destination folder does not, it will output the info for the 'image.png' file found in the source folder. It will keep doing this until all files in the source folder are found in the destination folder. Can use absolute or relative path, if relative, starts in output folder."}),
                "auto_reset": ("BOOLEAN",),
            }
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","INT","STRING","STRING","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("filename","folder_path", "extension", "full_path", "remaining_files","file_list", "file_list_w_idx", "status")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Next Unprocessed Image In Folder"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, folder, auto_reset):
        return float("nan")

    def process(self, folder, auto_reset):

        # Check if the provided path is absolute or relative
        if not os.path.isabs(folder):
            # If relative, construct the path with the output directory
            folder = f"{self.output_dir.strip()}\\{folder}"

        if not os.path.exists(folder.strip()):
            raise ValueError(f'The path `{folder.strip()}` specified doesn\'t exist!')

        # Define the common image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        
        # Construct the full search pattern (adding folder_path) for image extensions
        search_pattern = os.path.join(folder, "*")  # This will match all files in the folder
        matching_files = []

        # Search for image files with supported extensions
        for ext in image_extensions:
            matching_files.extend(glob.glob(search_pattern + ext))

        # If no images are found, raise an error
        if not matching_files:
            raise ValueError(f"No image files found in `{folder.strip()}` folder.")

        # Sort the files alphabetically
        matching_files.sort()

        log_name = "ImgProcessLog.txt"

        reviewed_file = os.path.join(folder, log_name)

        print(f"reviewed_file: [{reviewed_file}]")

        # If "reviewed.txt" exists read it
        reviewed_files = set()
        if os.path.exists(reviewed_file):
            with open(reviewed_file, 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if line:  # Ignore empty lines
                        reviewed_files.add(line.lower())  # Store filenames in lowercase

        # Remove files from "matching_files" that are in "reviewed_files" (ignoring file extensions)
        matching_files = [
            file for file in matching_files
            if os.path.splitext(os.path.basename(file))[0].lower() not in reviewed_files
        ]

        # If no files left to process
        if not matching_files:
            # And the user has selected auto_reset
            if auto_reset:
                print(f"All files processed, attempting to auto reset.")
                try:
                    # Attempt to delete the file
                    os.remove(reviewed_file)
                    print(f"File '{reviewed_file}' has been deleted successfully.")

                    # Search for image files with supported extensions
                    for ext in image_extensions:
                        matching_files.extend(glob.glob(search_pattern + ext))

                    # Sort the files alphabetically
                    matching_files.sort()
                except Exception as e:
                    
                    # If an error occurs, print the error message
                    raise Exception(f"Failed to reset, could no delete file '{reviewed_file}': {str(e)}")
            
            # And the user has not selected auto reset
            else:
                raise ValueError(f"No files to process after removing reviewed files. If you want to reset review previously reviewed notes, simply delete or edit the '{reviewed_file}' file in the destination folder.")

        # Get the total number of files still left in the list after existing, and reviewed files have been removed.
        total_files_found = len(matching_files)

        # Extract the file at the specified index
        full_path = matching_files[0]
        filename_with_extension = os.path.basename(full_path)
        filename, extension = os.path.splitext(filename_with_extension)

        # Append the "filename" to "reviewed.txt" while locking the file during write
        with open(reviewed_file, 'a') as file:
            file.write(f"{filename}\n")

        # Create the list of files as a string (one filename per line, without extension)
        file_list = "\n".join([os.path.splitext(os.path.basename(file))[0] for file in matching_files])

        # Create the list of files with index as a string (index:filename without extension)
        file_list_w_idx = "\n".join([f"{i}:{os.path.splitext(os.path.basename(file))[0]}" for i, file in enumerate(matching_files)])

        remaining_files = total_files_found - 1

        current_time = datetime.now().strftime('%I:%M:%S %p')
        status = f"Processing [{filename}] {remaining_files} remaining. [{current_time}]"

        # Return the requested information
        return (filename, folder, extension, full_path, remaining_files, file_list, file_list_w_idx, status)


##################################################################################################################
# BK Get Next Caption File
##################################################################################################################


class BKGetNextCaptionFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.processed_captions = []

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "folder": ("STRING", {"default": "","tooltip": "The node will scan the source path folder for any of the specified files, it will then return the next file that is in the source path, but not in the destination path. ex. If the source path has a file called 'image.png', and the destination folder does not, it will output the info for the 'image.png' file found in the source folder. It will keep doing this until all files in the source folder are found in the destination folder. Can use absolute or relative path, if relative, starts in output folder."}),
                "auto_reset": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("caption Text", "filename", "folder_path", "image_extension", "remaining_files", "status")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Get Next Caption File"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, folder, auto_reset):
        return float("nan")
    
    # ---------- Main process ----------

    def process(self, folder, auto_reset):
        # Resolve folder path
        folder_path = self.resolve_folder_path(folder)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Locate caption log
        log_path = self.get_caption_log_path(folder_path)

        # Read existing log
        logged_files = self.read_caption_log(log_path)

        # Find valid caption/image pairs
        valid_files = self.find_text_files_with_matching_images(folder_path)

        # Select next unprocessed file
        filename, image_extension = self.select_next_unlogged_file(
            valid_files, logged_files
        )

        if filename is None:
            if auto_reset:
                self.clear_caption_log(log_path)

            return (
                "",
                "",
                folder_path,
                "",
                0,
                "No remaining files to process.",
            )

        # Count remaining files
        remaining_files = self.count_remaining_unlogged_files(
            valid_files, logged_files, filename
        )

        # Read caption text
        caption_text = self.read_caption_text_file(folder_path, filename)

        # Update caption log
        self.append_to_caption_log(log_path, filename)

        # Auto-reset only when nothing remains
        if auto_reset and remaining_files == 0:
            self.clear_caption_log(log_path)

        # Build status
        status = self.build_status_text(filename, remaining_files)

        return (
            caption_text,
            filename,
            folder_path,
            image_extension,
            remaining_files,
            status,
        )

    # ---------- Path handling ----------

    def resolve_folder_path(self, folder):
        """Determine whether the folder path is absolute or relative."""
        if os.path.isabs(folder):
            return os.path.normpath(folder)
        return os.path.normpath(os.path.join(self.output_dir, folder))

    def get_caption_log_path(self, folder_path):
        """Return the full path to the caption log file."""
        return os.path.join(folder_path, "captionlog.caplog")

    # ---------- File discovery ----------

    def get_supported_image_extensions(self):
        """Return supported image file extensions."""
        return {
            ".png", ".jpg", ".jpeg", ".bmp",
            ".tiff", ".tif", ".webp"
        }

    def find_text_files_with_matching_images(self, folder_path):
        """
        Find all .txt files that have an image file
        with the same base name.
        """
        valid_files = []
        image_extensions = self.get_supported_image_extensions()

        for entry in os.listdir(folder_path):
            if not entry.lower().endswith(".txt"):
                continue

            base_name = os.path.splitext(entry)[0]

            for ext in image_extensions:
                image_path = os.path.join(folder_path, base_name + ext)
                if os.path.isfile(image_path):
                    valid_files.append((base_name, ext))
                    break

        return sorted(valid_files, key=lambda x: x[0])

    # ---------- Caption log handling ----------

    def read_caption_log(self, log_path):
        """Read processed filenames from captionlog.caplog."""
        if not os.path.isfile(log_path):
            return []

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            return [line.strip() for line in f if line.strip()]

    def append_to_caption_log(self, log_path, filename):
        """Append a filename to the caption log."""
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(filename + "\n")

    def clear_caption_log(self, log_path):
        """Clear the caption log file."""
        open(log_path, "w").close()

    # ---------- Selection logic ----------

    def select_next_unlogged_file(self, valid_files, logged_files):
        """Select the next file not found in the caption log."""
        for base_name, img_ext in valid_files:
            if base_name not in logged_files:
                return base_name, img_ext
        return None, None

    def count_remaining_unlogged_files(self, valid_files, logged_files, current_filename):
        """Count remaining unlogged files excluding the current one."""
        return sum(
            1 for base_name, _ in valid_files
            if base_name not in logged_files and base_name != current_filename
        )

    # ---------- File reading ----------

    def read_caption_text_file(self, folder_path, filename):
        """Read the caption text file in read-only mode."""
        txt_path = os.path.join(folder_path, filename + ".txt")
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ---------- Status ----------

    def build_status_text(self, filename, remaining_files):
        """Build the status text."""
        return f"Processing file [{filename}.txt] with [{remaining_files}] reamaining."

    


    

##################################################################################################################
# BK AI Text Cleaner
##################################################################################################################

class BKAITextCleaner:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text": ("STRING", {"default": "","tooltip": "Output text from AI generation."}),
                "exclude": ("STRING", {"default": "", "tooltip": "Any text placed here will not be `cleaned` or removed from the text automatically. This is useful for tags or placeholders."})
            }
        }
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("cleaned_text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK AI Text Cleaner"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, text, exclude):
        return float("nan")
    
    def sanitize_excluded_tags_to_list(self, exclude):
        if exclude:
            excluded_tags_list = [tag for tag in exclude.split(',') if tag]
        else:
            excluded_tags_list = []
        return excluded_tags_list
    
    def has_not_reached_end_of_text(self, idx, text):
        return idx < len(text)

    def is_tag_found_at_index(self, text, idx, tag):
        return text[idx:idx + len(tag)] == tag

    def move_index_past_tag(self, idx, tag):
        return idx + len(tag)
    
    def is_tag_not_found_at_index(self, matched):
        return not matched
    
    def find_matching_tag(self, text, idx, excluded_tags_list):
        for tag in excluded_tags_list:
            if self.is_tag_found_at_index(text, idx, tag):
                return tag
        return None
    
    def handle_found_tag(self, tag, idx, current_part, parts):
        if current_part:
            parts.append(current_part)

        parts.append(tag)
        current_part = ""
        idx = self.move_index_past_tag(idx, tag)

        return idx, current_part
    
    def handle_regular_character(self, text, idx, current_part):
        current_part += text[idx]
        idx += 1
        return current_part, idx



    def process(self, text, exclude):
        print_debug_header(self.is_debug, "BK AI TEXT CLEANER")
        
        excluded_tags_list = self.sanitize_excluded_tags_to_list(exclude)

        # Create a list of text parts
        parts = []
        current_part = ""
        
        # Use regex to match the excluded tags and split the text
        idx = 0

        self.print_debug(f"len(text): {len(text)}")

        while self.has_not_reached_end_of_text(idx, text):
            tag = self.find_matching_tag(text, idx, excluded_tags_list)

            if tag:
                idx, current_part = self.handle_found_tag(tag, idx, current_part, parts)
            else:
                current_part, idx = self.handle_regular_character(text, idx, current_part)

        # If there's any remaining non-matching part, append it
        if current_part:
            parts.append(current_part)

        # Now process only the non-tag parts
        for i in range(len(parts)):
            if parts[i] not in excluded_tags_list:
                # Remove text inside special characters (e.g., '*' '<' '>')
                parts[i] = re.sub(r'([^\w\s,\-\*])\S*([^\w\s,\-\*])', '', parts[i])

                # Remove unwanted characters that are not letters, digits, or common punctuation
                parts[i] = re.sub(r'[^a-zA-Z0-9\s.!?\'",-—]', '', parts[i])

                # Normalize to ensure UTF-8 compatibility
                parts[i] = unicodedata.normalize('NFKD', parts[i]).encode('utf-8', 'ignore').decode('utf-8')

        # Reassemble the parts back into a single cleaned text
        cleaned_text = ''.join(parts).strip()

        # Remove all new lines from the cleaned text
        cleaned_text = cleaned_text.replace('\n','')

        print_debug_bar(self.is_debug)
        # Return the cleaned text as a tuple
        return (cleaned_text,)
    
    def print_debug(self, string):
        if self.is_debug:
            print(string)

   
class BKRemoveMaskAtIdx:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "mask": ("MASK",),
                "remove_idx": ("INT", {"default": 0, "min": 0, "step": 1,"tooltip": "Only removes a mask at the index if the mask count is greater than 2. This prevents a null mask from being output."}),
            }
        }
    
    RETURN_TYPES = ("MASK", "INT",)  # This specifies that the output will be text
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Remove Mask At Idx"  # Default label text
    OUTPUT_NODE = True

    def process(self, mask: torch.Tensor, remove_idx: int):

        # Check if remove_idx is valid
        if remove_idx < 0 or remove_idx >= mask.size(0):
            sub_mask = mask
            return (sub_mask, sub_mask.size(0))

        # Remove the index `remove_idx` from the `mask` tensor
        sub_mask = torch.cat((mask[:remove_idx], mask[remove_idx + 1:]), dim=0)

        # Return the sub_mask and its size
        return (sub_mask, sub_mask.size(0))
    
class BKRemoveMaskAtIdxSAM3:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "mask": ("MASK",),
                "remove_idx": ("INT", {"default": 0, "min": 0, "step": 1,"tooltip": "Only removes a mask at the index if the mask count is greater than 2. This prevents a null mask from being output."}),
            }
        }
    
    RETURN_TYPES = ("MASK", "INT",)  # This specifies that the output will be text
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Remove Mask At Idx SAM3"  # Default label text
    OUTPUT_NODE = True

    def process(self, mask: torch.Tensor, remove_idx: int):

        # Check if remove_idx is valid
        if mask.size(0) <= 1:
            sub_mask = mask
            return (sub_mask, sub_mask.size(0))

        # Check if remove_idx is valid
        if remove_idx < 0 or remove_idx >= mask.size(0):
            sub_mask = mask
            return (sub_mask, sub_mask.size(0))

        # Remove the index `remove_idx` from the `mask` tensor
        sub_mask = torch.cat((mask[:remove_idx], mask[remove_idx + 1:]), dim=0)

        # Return the sub_mask and its size
        return (sub_mask, sub_mask.size(0))

class BKGetNextImgWOCaption:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "folder_path": ("STRING", {"default": "","tooltip": "The absolute path of the folder you wish to search through. (Can be anywhere.)"}),
            }
        }
    
    
    @classmethod
    def IS_CHANGED(self, folder_path):
        return float("nan")
    
    RETURN_TYPES = ("STRING","STRING","STRING","INT","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("filename","folder_path", "extension", "remaining", "status",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Get Next Img WO Caption"  # Default label text
    OUTPUT_NODE = True

    def process(self, folder_path):
        # List of image file extensions we are interested in
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        
        # List to store image file names
        images = []
        images_missing_txt = []
        
        # Loop through the folder and get all image files
        for file in os.listdir(folder_path):
            file_name, file_extension = os.path.splitext(file)
            # Check if the file is an image based on its extension
            if file_extension.lower() in image_extensions:
                images.append(file)
                # Check if the corresponding .txt file exists
                txt_file = os.path.join(folder_path, f"{file_name}.txt")
                if not os.path.exists(txt_file):
                    images_missing_txt.append(file)
        
        # If no images are found, raise an error
        if not images:
            raise ValueError("No image files found in the specified folder.")
        
        # If there are images without corresponding .txt files
        if images_missing_txt:
            # Get the first image without a corresponding .txt file
            first_missing_image = images_missing_txt[0]
            base_name, extension = os.path.splitext(first_missing_image)
            remaining = len(images_missing_txt)
            # Create the status message
            current_time = datetime.now().strftime('%I:%M:%S %p')
            status = f"Processing [{base_name}] with [{remaining}] remaining. [{current_time}]"
            return (base_name, folder_path, extension, remaining, status)
        
        # If all images have corresponding .txt files
        return ("All images have captions, no missing captions found.", folder_path, "", 0, "All images have captions, no missing captions found.")
        
   

class BKLoopPathBuilder:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "output_type": ("STRING", {"root": f'', "multiline": False, "tooltip": "This is the root folder. It keeps the file types organized in the output folder. i.e. images, videos, prompts, etc."}),
                "project_name": ("STRING", {"folder": f'', "multiline": False, "tooltip": "This folder will describe the process being applied so that content generated with the same project are all in the same spot. i.e. 'text 2 video'"}),
                "subject_name": ("STRING", {"folder": f'', "multiline": False, "tooltip": "This is the name of the specific subject of the content you are creating. i.e. 'wendy' if you creating images of a girl named wendy."}),
                "file_seed": ("INT", {"default": 0,"tooltip": "Add the seed to the filename for reference."}),
                "loop_idx": ("INT", {"default": 0,"tooltip": "Adds loop idx to file name."}),
                "add_nested_seed_folder": ("BOOLEAN", {"default": False, "tooltip": "If true will generate subfolders inside of the subect folder with the name of the specified seed. This is useful if you are generating multiple files for one seed."}),
                "use_absolute_paths": ("BOOLEAN", {"default": False, "tooltip": "If true will include the output folder path in the returned paths and the drive letter. Some nodes require this, but most of the time you will want to use relative. Returns paths as absolute paths instead of relative.)"}),
                "add_seed_to_name": ("BOOLEAN", {"default": True, "tooltip": "This is generally good for reference, especially with mutiple files. On by default but can be turned off.)"}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("file_name", "folder_path", "full_path",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Loop Path Builder"  # Default label text
    OUTPUT_NODE = True

    def process(self, output_type, project_name, loop_idx, subject_name, file_seed, add_nested_seed_folder, use_absolute_paths, add_seed_to_name, extra_pnginfo=None):
        # Ensure paths are not ending with a backslash, to avoid duplication
        output_type = output_type.rstrip("\\")
        project_name = project_name.rstrip("\\")
        relative_folder_path = f"{output_type}\\{project_name}\\{subject_name}"

        # Add seed to filename if add_seed_to_name is true
        if add_seed_to_name:
            built_file_name = f"{subject_name}_{file_seed}_{loop_idx}"
        else:
            built_file_name = subject_name

        # Generate folder_path based on conditions
        if use_absolute_paths:
            folder_path = f"{self.output_dir}\\{relative_folder_path}"
        else:
            folder_path = f"{relative_folder_path}"

        # Add nested folder to filename if add_nested_folder is true
        if add_nested_seed_folder:
            folder_path = f"{folder_path}\\{file_seed}"

        # Generate file_path based on conditions
        full_path = f"{folder_path}\\{built_file_name}"

        return (built_file_name, folder_path, full_path)
    


class BKPathBuilder:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "output_type": ("STRING", {"root": f'', "multiline": False, "tooltip": "This is the root folder. It keeps the file types organized in the output folder. i.e. images, videos, prompts, etc."}),
                "project_name": ("STRING", {"folder": f'', "multiline": False, "tooltip": "This folder will describe the process being applied so that content generated with the same project are all in the same spot. i.e. 'text 2 video'"}),
                "subject_name": ("STRING", {"folder": f'', "multiline": False, "tooltip": "This is the name of the specific subject of the content you are creating. i.e. 'wendy' if you creating images of a girl named wendy."}),
                "file_seed": ("INT", {"default": 0,"tooltip": "Add the seed to the filename for reference."}),
                "add_nested_seed_folder": ("BOOLEAN", {"default": False, "tooltip": "If true will generate subfolders inside of the subect folder with the name of the specified seed. This is useful if you are generating multiple files for one seed."}),
                "use_absolute_paths": ("BOOLEAN", {"default": False, "tooltip": "If true will include the output folder path in the returned paths and the drive letter. Some nodes require this, but most of the time you will want to use relative. Returns paths as absolute paths instead of relative.)"}),
                "add_seed_to_name": ("BOOLEAN", {"default": True, "tooltip": "This is generally good for reference, especially with mutiple files. On by default but can be turned off.)"}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("file_name", "folder_path", "full_path",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "Easy Path Builder"  # Default label text
    OUTPUT_NODE = True

    def process(self, output_type, project_name, subject_name, file_seed, add_nested_seed_folder, use_absolute_paths, add_seed_to_name, extra_pnginfo=None):
        # Ensure paths are not ending with a backslash, to avoid duplication
        output_type = output_type.rstrip("\\")
        project_name = project_name.rstrip("\\")
        relative_folder_path = f"{output_type}\\{project_name}\\{subject_name}"

        # Add seed to filename if add_seed_to_name is true
        if add_seed_to_name:
            built_file_name = f"{subject_name}_{file_seed}"
        else:
            built_file_name = subject_name

        # Generate folder_path based on conditions
        if use_absolute_paths:
            folder_path = f"{self.output_dir}\\{relative_folder_path}"
        else:
            folder_path = f"{relative_folder_path}"

        # Add nested folder to filename if add_nested_folder is true
        if add_nested_seed_folder:
            folder_path = f"{folder_path}\\{file_seed}"

        # Generate file_path based on conditions
        full_path = f"{folder_path}\\{built_file_name}"

        return (built_file_name, folder_path, full_path)


class BKPrintToConsole:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "string": ("STRING", {"default": f'', "multiline": True}),
            }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "process"
    CATEGORY = "BKNodes"
    LABEL = "BK Print To Console"  # Default label text



    def process(self, string):
        print(string)
        return (string,)


class BKStringSplitter:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "string": ("STRING", {"default": f'', "multiline": True}),
            "character": ("STRING", {"default": f'', "multiline": False}),
            }}

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("first_string","second_string")

    FUNCTION = "process"
    CATEGORY = "BKNodes"
    LABEL = "BK Prompt Sync"  # Default label text



    def process(self, string, character):
        # Check if either string or character is None
        if string is None or character is None:
            raise ValueError("Both string and character must be provided")
        
        # Find the index of the character, default to -1 if not found
        index = string.find(character)
        
        # If the character is found, split using the index
        if index != -1:
            return (string[:index], string[index + 1:])
        
        # If the character is not found, return the entire string and an empty string
        return (string, "")

class BKRemoveLastFolder:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "string": ("STRING", {"default": f'', "multiline": True}),
            }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_string",)

    FUNCTION = "process"
    CATEGORY = "BKNodes"
    LABEL = "BK Remove Last Folder"  # Default label text

    def process(self, string):
        # Remove leading/trailing backslashes
        string = string.strip('\\')
        
        # Split into parts
        parts = string.split('\\')
        
        # Remove the last part if there’s more than one
        if len(parts) > 1:
            parts = parts[:-1]
        else:
            parts = []
        
        # Join back together without a trailing '\'
        result = '\\'.join(parts)
        
        return(result,)

###################################################################################################################
# BK Image Sync
###################################################################################################################
class BKImageSync:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image_a": ("IMAGE", ),
            "image_b": ("IMAGE", ),
            "string": ("STRING", ),

            }}
    
    @classmethod
    def IS_CHANGED(self, image_a, image_b, string):
        return float("nan")

    RETURN_TYPES = ("IMAGE","IMAGE","STRING")
    RETURN_NAMES = ("IMAGE_A","IMAGE_B","TEXT")

    FUNCTION = "process"
    CATEGORY = "BKNodes"
    LABEL = "BK Image Sync"  # Default label text



    def process(self, image_a, image_b, string):
        
        
        # Return the image, prompt, and other property
        return (image_a, image_b, string)

class BKPromptSync:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE", ),
            "prompt": ("STRING", {"default": f'', "multiline": True}),
            "remove": ("STRING", {"default": f'', "multiline": False}),
            "other": ("STRING", {"default": f'', "multiline": True}),
            }}
    
    @classmethod
    def IS_CHANGED(self, image, prompt, remove, other):
        return float("nan")

    RETURN_TYPES = ("IMAGE","STRING","STRING")
    RETURN_NAMES = ("IMAGE","prompt","other")

    FUNCTION = "process"
    CATEGORY = "BKNodes"
    LABEL = "BK Prompt Sync"  # Default label text



    def process(self, image, prompt, remove, other):
        # Split comma separated string "remove" into a string list
        remove_list = remove.split(",")
        
        # For each text item in the string list, 
        for item in remove_list:
            # Replace every occurrence of the text item in prompt with an empty string
            prompt = prompt.replace(item, "")
        
        # Remove all trailing and leading spaces and new line characters from prompt
        prompt = prompt.strip()
        
        # Return the image, prompt, and other property
        return (image, prompt, other)

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
class BKAddToJSON:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "json_string": ("STRING", {"default": f'', "multiline": False}),
                "name": ("STRING", {"default": f'', "multiline": False}),
                "value": ("STRING", {"default": f'file', "multiline": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("json_string",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Add To JSON"  # Default label text
    OUTPUT_NODE = True

    def process(self, name, value, json_string = None):

        if not json_string:
            # Create a valid JSON string directly from the key-value pair
            json_obj= dict()

        else:
            try:
                # Parse the existing JSON string
                json_obj = json.loads(json_string)

            except json.JSONDecodeError as e:

                # Initialize with the new key-value pair in case of error
                json_obj= dict()

        # Add or overwrite the value for the specified name
        json_obj[name] = value

        # Convert the updated dictionary, back to a valid JSON string using json.dumps
        json_str = json.dumps(json_obj)

        return (json_str,)




# Node class definition
class BKReadFromJSON:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "json_string": ("STRING", {"default": f'', "multiline": False}),
                "name": ("STRING", {"default": f'', "multiline": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("value",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Add To JSON"  # Default label text
    OUTPUT_NODE = True


        
    def process(self, name, json_string=""):
        if not json_string:
            # If the string is empty, return an empty value
            return ("",)
        
        try:
            # Parse the JSON string into a dictionary
            data = json.loads(json_string)
            
            # Return the value for the specified name or an empty string if not found
            return (data.get(name, ""),)
        except json.JSONDecodeError:
            # If JSON is invalid, raise an error with a message
            raise ValueError("The provided json_string is not a valid JSON.")

# Node class definition
class BKSaveTextFile:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text": ("STRING", {"default": f'', "multiline": False}),
                "path": ("STRING", {"default": f'', "multiline": False}),
                "subfolder": ("STRING", {"default": f'', "multiline": False}),
                "filename": ("STRING", {"default": f'file', "multiline": False}),
                "suffix": ("STRING", {"default": f'', "multiline": False}),
                "extension":  ("STRING", {"default": f'.txt', "multiline": False}),
                "mode":(["overwrite","append","append_w_newline"],),
            },
        }
    
    RETURN_TYPES = ("STRING","STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("folderpath", "text",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Save Text File"  # Default label tex
    OUTPUT_NODE = True


    
    def process(self, text, path, filename, suffix, extension, mode, subfolder=None):
        # Combine output_dir and path
        output_path = os.path.join(self.output_dir, path)

        # If path is relative, add output folder to beginning of path
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir.strip(), output_path.strip())

        if not text:
            print(f"No content found in text. Aborting writing file.")
            return (output_path, text)

        # Create directory if it doesn't exist
        if not os.path.exists(output_path.strip()):
            print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
            os.makedirs(output_path, exist_ok=True)

        # If subfolder is provided, add it to the output path
        if subfolder:
            output_path = os.path.join(output_path, subfolder)

        # Combine path, filename, suffix, and extension to get the full file path
        filepath = os.path.join(output_path.strip(), f"{filename.strip()}{suffix.strip()}{extension.strip()}")

        # Open file based on mode
        try:
            # If mode is 'overwrite', open the file in write mode (this will overwrite existing file)
            if mode == "overwrite":
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(text)
                print(f"Content written to {filepath} (overwritten)")

            # If mode is 'append', open the file in append mode (create if it doesn't exist)
            elif mode == "append":
                # Check if the file exists
                if os.path.exists(filepath):
                    with open(filepath, 'a', encoding='utf-8') as file:
                        file.write(text)
                    print(f"Content appended to {filepath}")
                else:
                    # If file doesn't exist, create it with the provided content
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(text)
                    print(f"File created and content written to {filepath}")
            elif mode == "append_w_newline":
                # Check if the file exists
                if os.path.exists(filepath):
                    with open(filepath, 'a', encoding='utf-8') as file:
                        file.write(f"{text}\n")
                    print(f"Content appended to {filepath}")
                else:
                    # If file doesn't exist, create it with the provided content
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(f"{text}\n")
                    print(f"File created and content written to {filepath}")

            else:
                print(f"Invalid mode: {mode}. Please use 'overwrite' or 'append'.")

            return (output_path, text)

        except Exception as e:
            print(f"Error processing file: {e}")
            return None




def combine_path(folder_path, file_name, suffix, extension):
    # Ensure the folder path ends without a trailing separator
    folder_path = Path(folder_path).resolve()
    
    # Clean up the extension to ensure it only has one leading period
    if extension.startswith('.'):
        extension = extension[1:]  # Remove the leading period if it exists
    # Combine the folder path with the file name and the cleaned extension
    full_path = folder_path / f"{file_name}{suffix}.{extension}"
    
    return str(full_path)

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


class BKSamplerOptionsSelector:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.output_path = ""
        self.file_path = ""
        self.invalid_filename_chars = r'[\/:*?"<>|]'
        self.invalid_path_chars = r'[:*?"<>|]'

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            },
        }
    
    #RETURN_TYPES = ("MODEL","CLIP","VAE")  # This specifies that the output will be text
    #RETURN_NAMES = ("MODEL","CLIP", "VAE")
    RETURN_TYPES = ("FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)  # This specifies that the output will be text
    RETURN_NAMES = ("CFG","SAMPLER","SCHEDULER" )
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Sampler Options Selector"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, cfg, sampler_name, scheduler):
        return float("nan")

    
    def process(self, cfg, sampler_name, scheduler):
        return (cfg, sampler_name, scheduler)


# Node class definition
class BKDynamicCheckpoints:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.output_path = ""
        self.file_path = ""
        self.invalid_filename_chars = r'[\/:*?"<>|]'
        self.invalid_path_chars = r'[:*?"<>|]'

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "path": ("STRING", {"default": f'', "multiline": False}),
                "select": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    #RETURN_TYPES = ("MODEL","CLIP","VAE")  # This specifies that the output will be text
    #RETURN_NAMES = ("MODEL","CLIP", "VAE")
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "INT", "STRING", "STRING", "INT")  # This specifies that the output will be text
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "NAME", "TOTAL_CHECKPOINTS","UNIQUE_PATHS","SELECTED_CHECKPOINT","CHECKPOINT_NUM")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Dynamic Checkpoints"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, select, path=None, extra_pnginfo = None):
        return float("nan")

    
    def process(self, select, path=None, extra_pnginfo = None):
        filtered_checkpoints = []
        all_checkpoints = folder_paths.get_filename_list("checkpoints")

        if all_checkpoints is None:
            raise ValueError(f"No checkpoints found in ComfyUI.")

        if path is None:
            filtered_checkpoints = all_checkpoints
        else:
            for checkpoint in all_checkpoints:
                if checkpoint.startswith(path):
                    filtered_checkpoints.append(checkpoint)
            if filtered_checkpoints is None:
                raise ValueError(f"No checkpoints in the 'checkpoints' folder start with path [{path}].")
            
        #sort the list alphabetically
        filtered_checkpoints.sort(key=str.lower)

        # Slect a checkpoint based on the seed, if the seed is too large, it will simply roll over and start at beginning of list
        if not filtered_checkpoints:
            selected_checkpoint = None
        else:
            idx = select % len(filtered_checkpoints)
            selected_checkpoint = filtered_checkpoints[idx]

        if selected_checkpoint is None:
            raise ValueError(f"Failed to find checkpoint in list using seed. Ensure the path is valid.")
            
        #Get total count of checkpoints found
        total_checkpoints = len(filtered_checkpoints)

        #Load checkpoint
        checkpoint_name = os.path.basename(selected_checkpoint)
        checkpoint = checkpointLoader(selected_checkpoint)
        checkpoint_model = checkpoint[0]
        checkpoint_clip = checkpoint[1]
        checkpoint_vae = checkpoint[2]

        unique_paths = {str(Path(p).parent) for p in all_checkpoints if Path(p).parent != Path('.')}

        return (checkpoint_model, checkpoint_clip, checkpoint_vae, checkpoint_name, total_checkpoints, unique_paths, selected_checkpoint, idx)


# Node class definition
class BKGetNextMissingCheckpoint:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.output_path = ""
        self.file_path = ""
        self.invalid_filename_chars = r'[\/:*?"<>|]'
        self.invalid_path_chars = r'[:*?"<>|]'

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "checkpoint_path": ("STRING", {"default": f'', "multiline": False}),
                "folder_path": ("STRING", {"default": f'', "multiline": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING", "INT", "INT", "STRING", "STRING", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "folder_path", "name","combined_path", "remaining_checkpoints","found_checkpoints","unique_paths","selected_checkpoints", "status")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Get Next Missing Checkpoint"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, folder_path, checkpoint_path=None):
        return float("nan")

    
    def process(self, folder_path, checkpoint_path=None):
            filtered_checkpoints = []
            all_checkpoints = folder_paths.get_filename_list("checkpoints")

            if all_checkpoints is None:
                raise ValueError(f"No checkpoints found in ComfyUI.")

            if checkpoint_path is None:
                filtered_checkpoints = all_checkpoints
            else:
                for checkpoint in all_checkpoints:
                    if checkpoint.startswith(checkpoint_path):
                        filtered_checkpoints.append(checkpoint)
                if filtered_checkpoints is None:
                    raise ValueError(f"No checkpoints in the 'checkpoints' folder start with path [{checkpoint_path}].")

            # Remove None values and ensure all elements are valid strings
            filtered_checkpoints = [item for item in filtered_checkpoints if isinstance(item, str) and item is not None]

            if len(filtered_checkpoints) == 0:
                raise ValueError(f"No checkpoints found in ComfyUI with that path.")

            missing_checkpoints = []
 
            for checkpoint in filtered_checkpoints:
                checkpoint_name_wo_ext = os.path.splitext(os.path.basename(checkpoint))[0]  # Get the checkpoint name from path
                combined_path = os.path.join(folder_path, checkpoint_name_wo_ext)  # Combine folder_path and checkpoint_name

                if not os.path.exists(combined_path):  # Check if the path doesn't exist
                    missing_checkpoints.append(checkpoint)  # Add to missing_checkpoints if not exists
         
            if missing_checkpoints:  # If there are missing checkpoints
                selected_checkpoint = missing_checkpoints[0]  # Select the first missing checkpoint
                selected_checkpoint_name_wo_ext = os.path.splitext(os.path.basename(selected_checkpoint))[0]
                combined_path = os.path.join(folder_path, selected_checkpoint_name_wo_ext)

            else:
                selected_checkpoint = None  # If all checkpoints are present, set selected_checkpoint to None

            if missing_checkpoints is None:
                raise ValueError(f"All checkpoint folders created. None are missing in the [{folder_path}] folder.")
        
            if selected_checkpoint is None:
                raise ValueError(f"Failed to select checkpoint from missing checkpoints list.")
            
            #Get total count of checkpoints found
            remaining_checkpoints = len(missing_checkpoints)
            found_checkpoints = len(filtered_checkpoints)

            #Load checkpoint
          
            checkpoint = checkpointLoader(selected_checkpoint)
            checkpoint_model = checkpoint[0]
            checkpoint_clip = checkpoint[1]
            checkpoint_vae = checkpoint[2]

            unique_paths = {str(Path(p).parent) for p in all_checkpoints if Path(p).parent != Path('.')}

            status = f"Processing [{selected_checkpoint_name_wo_ext}] with [{remaining_checkpoints}] remaining."

            return (checkpoint_model, checkpoint_clip, checkpoint_vae, folder_path, selected_checkpoint_name_wo_ext, combined_path,  remaining_checkpoints,found_checkpoints, unique_paths, selected_checkpoint, status)

class BKDynamicCheckpointsList:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.output_path = ""
        self.file_path = ""
        self.invalid_filename_chars = r'[\/:*?"<>|]'
        self.invalid_path_chars = r'[:*?"<>|]'

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "default_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Using the syntax <name>|<cfg>|<sampler> allows you to override this value on a line by line basis. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "default_sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The default algorithm used when sampling. Using the syntax <name>|<cfg>|<sampler> allows you to override this value on a line by line basis, this can affect the quality, speed, and style of the generated output."}),
                "select": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "names": ("STRING", {"default": f'', "multiline": True, "tooltip": "Names of checkpoints to load."}),
                
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    #RETURN_TYPES = ("MODEL","CLIP","VAE")  # This specifies that the output will be text
    #RETURN_NAMES = ("MODEL","CLIP", "VAE")
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "FLOAT", comfy.samplers.KSampler.SAMPLERS, "STRING", "INT", "STRING", "STRING", "INT")  # This specifies that the output will be text
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "CFG", "SAMPLER", "NAME", "TOTAL_CHECKPOINTS","UNIQUE_PATHS","SELECTED_CHECKPOINT","CHECKPOINT_NUM")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Dynamic Checkpoints"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, cfg, sampler_name, select, names=None, extra_pnginfo=None):
        return float("nan")

    
    def process(self, default_cfg, default_sampler, select, names=None, extra_pnginfo=None):
        filtered_checkpoints = []
        all_checkpoints = folder_paths.get_filename_list("checkpoints")

        if not all_checkpoints:
            raise ValueError("No checkpoints found in ComfyUI.")

        if names is None:
            filtered_checkpoints = all_checkpoints
        else:
            # Split the names string by lines and strip whitespace
            name_filters = [n.strip() for n in names.splitlines() if n.strip()]
            
            # Process each name filter to handle both formats
            for checkpoint in all_checkpoints:
                filename = os.path.basename(checkpoint)
                
                # Loop through the name filters and process both formats
                for name_filter in name_filters:
                    parts = name_filter.split('|')

                    if len(parts) == 1:  # <name> format
                        # In this case, no cfg or sampler is provided, so just match based on <name>
                        if any(part in filename for part in parts):
                            filtered_checkpoints.append(checkpoint)
                        continue
                    
                    elif len(parts) == 3:  # <name>|<cfg>|<sampler> format
                        name, cfg_str, sampler = parts
                        
                        # Validate that the cfg is a float
                        try:
                            cfg_value = float(cfg_str)
                        except ValueError:
                            raise ValueError(f"Invalid cfg format for {name}: {cfg_str} is not a valid float.")
                        
                        # If valid, check if the sampler is in the list of valid samplers
                        if sampler not in comfy.samplers.SAMPLER_NAMES:
                            raise ValueError(f"Sampler '{sampler}' for {name} is not found in the list of valid samplers.")
                        
                        # If valid, return the cfg value and sampler name
                        if any(name in filename for name in [name]):
                            default_cfg = cfg_value  # Set the cfg to the validated float value
                            default_sampler = sampler  # Set the sampler_name
                            filtered_checkpoints.append(checkpoint)
                        continue

        if not filtered_checkpoints:
            raise ValueError(
                "No checkpoints matched any of the provided names."
                if names else
                "No checkpoints found."
            )

        # Sort the list alphabetically (case-insensitive)
        filtered_checkpoints.sort(key=str.lower)

        # Select a checkpoint based on the seed (rolls over if too large)
        idx = select % len(filtered_checkpoints)
        selected_checkpoint = filtered_checkpoints[idx]

        if selected_checkpoint is None:
            raise ValueError("Failed to find checkpoint in list using seed. Ensure the path is valid.")

        # Load the selected checkpoint
        checkpoint_name = os.path.basename(selected_checkpoint)
        checkpoint_model, checkpoint_clip, checkpoint_vae = checkpointLoader(selected_checkpoint)

        # Collect unique parent paths of all checkpoints
        unique_paths = {
            str(Path(p).parent)
            for p in all_checkpoints
            if Path(p).parent != Path('.')
        }

        total_checkpoints = len(filtered_checkpoints)

        return (
            checkpoint_model,
            checkpoint_clip,
            checkpoint_vae,
            default_cfg,
            default_sampler,
            checkpoint_name,
            total_checkpoints,
            unique_paths,
            selected_checkpoint,
            idx,
        )

def checkpointLoader(ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))     
        return out[:3]

##################################################################################################################
# BK SAVE IMAGE
##################################################################################################################
#TODO: Needs overhaul
class BKSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.output_path = ""
        self.file_path = ""
        self.invalid_filename_chars = r'[\/:*?"<>|]'
        self.invalid_path_chars = r'[*?"<>|]'
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "images": ("IMAGE", ),
                
                "filename": ("STRING", {"default": f'image', "multiline": False}),
                "name_suffix": ("STRING", {"default": f'', "multiline": False}),
                "folder_path": ("STRING", {"default": f'', "multiline": False}),
                "subfolder": ("STRING", {"default": f'', "multiline": False}),
                "include_workflow": ("BOOLEAN", {"default": True, "tooltip": "If true will save a copy of the workflow into the PNG, Else will not."}),
                "strip_invalid_chars": ("BOOLEAN", {"default": True, "tooltip": "If true will automatically strip invalid characters from the filename, if false, will save the file with the filename specified."}),
                "add_seed_to_name": ("BOOLEAN", {"default": True, "tooltip": "If true will add seed to the name so the file is not overwritten, otherwise will keep overwriting file."}),
                "is_save_image": ("BOOLEAN", {"default": True, "tooltip": "Node will only save the image if set to true. Else it will not save the image."}),
            },
            "optional": {
                "seed_optional": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "positive_optional": ("STRING",{ "multiline": True, "forceInput": True}, ),
                "negative_optional": ("STRING",{"multiline": True, "forceInput": True}, ),
                "other_optional": ("STRING",{"multiline": True, "forceInput": True}, ),
                "mask_optional": ("MASK",),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("IMAGE","STRING","STRING","STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("IMAGE","folderpath", "filename", "full_path")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Save Image"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, images, folder_path, include_workflow, is_save_image, filename=None, subfolder=None, name_suffix=None, positive_optional=None, negative_optional=None, seed_optional=None, other_optional=None, extra_pnginfo=None, strip_invalid_chars=True, add_seed_to_name=True, mask_optional=None):
        return float("nan")

    
    def process(self, images, folder_path, include_workflow, is_save_image, filename=None, subfolder=None, name_suffix=None, positive_optional=None, negative_optional=None, seed_optional=None, other_optional=None, extra_pnginfo=None, strip_invalid_chars=True, add_seed_to_name=True, mask_optional=None):
        print_debug_header(self.is_debug, "BK SAVE IMAGE NODE")
        
        
        if filename is None:
            raise ValueError(f"Please enter a name to save the file as.")

        if strip_invalid_chars:
            filename = re.sub(self.invalid_filename_chars, '', filename)
            if not name_suffix is None:
                name_suffix = re.sub(self.invalid_filename_chars, '', name_suffix)
            folder_path = re.sub(self.invalid_path_chars, '', folder_path)
            subfolder = re.sub(self.invalid_path_chars, '', subfolder)

        # Check if the provided path is absolute or relative
        if not os.path.isabs(folder_path):
            folder_path = f"{self.output_dir.strip()}\\{folder_path}"

        if not subfolder is None:
            folder_path = os.path.join(folder_path, subfolder)

        # only add seed to name if option is toggled

        self.print_debug(f"add_seed_to_name: {add_seed_to_name}")
        if add_seed_to_name:
            self.print_debug(f"seed_optional: {seed_optional}")
            if seed_optional:
                filename = f"{filename}_{seed_optional}"
            else:
                filename = f"{filename}_NA"

        # Define the full file path with the correct file separator for the OS
        filepath = os.path.join(folder_path.strip(), f"{filename.strip()}")

        if is_save_image:
            # Ensure the output directory exists
            if folder_path.strip() != '':
                if not os.path.exists(folder_path.strip()):
                    print(f'The path `{folder_path.strip()}` specified doesn\'t exist! Creating directory.')
                    os.makedirs(folder_path, exist_ok=True)  

            savedpath = self.save_images(images, filepath, extra_pnginfo, include_workflow, name_suffix, positive_optional, negative_optional, seed_optional, other_optional, mask_optional)
            
            if len(images) == 0:
                print(f'No images found, nothing to save.')
            elif len(images) == 1:
                print(f'Image saved to: {savedpath}')
            else:
                print(f'Images saved to: {filepath}_###_{name_suffix}.png')
        
            print_debug_bar(self.is_debug)
            return images, folder_path.strip(), f"{filename.strip()}", f"{filepath}.png"
        else:
            print_debug_bar(self.is_debug)
            return images, folder_path.strip(), f"{filename.strip()}", f"{filepath}.png"


    def save_images(self, images, filepath, extra_pnginfo, include_workflow, suffix=None, positive_optional=None, negative_optional=None, seed_optional=None, other_optional=None, mask_optional=None) -> list[str]:
        
        # Having img_count outside the loop allows us to pickup where we left off for the next image
        img_count = 1

        if filepath is not None and suffix is not None:
            filepath = f"{filepath}_{suffix}"

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # If a mask is provided and is not empty, combine the mask with the image to create an alpha channel
            if mask_optional is not None and mask_optional.sum() > 0:  # Check if mask is not empty
                # Ensure the mask is a binary mask (values should be 0 or 1)
                mask = mask_optional.cpu().numpy().astype(np.uint8) * 255  # scale to 255 for transparency

                # Invert the mask: 1 (transparent) becomes 0, 0 (opaque) becomes 255
                inverted_mask = 255 - mask  # Invert mask values: 1 -> 0, 0 -> 255

                # Convert the image to RGBA (Red, Green, Blue, Alpha)
                img = img.convert("RGBA")
                img_array = np.array(img)

                # Replace the alpha channel with the inverted mask (fully transparent where mask is 1, opaque where mask is 0)
                img_array[..., 3] = inverted_mask  # Set the alpha channel based on the inverted mask

                img = Image.fromarray(img_array)

            # Create file path with extension
            filepath_w_ext = filepath + ".png"
            numbered_filepath_w_ext = filepath_w_ext

            # If the file already exists, add a number to the name
            if os.path.exists(filepath_w_ext):
                # Keep increasing the filename's number until we get one that does not exist
                while os.path.exists(numbered_filepath_w_ext):
                    # Generate new filename with incremented img_count
                    numbered_filepath = f"{filepath}_{img_count:06d}"
                    numbered_filepath_w_ext = f"{numbered_filepath}.png"
                    img_count += 1  # Increment the image count

            # Final filepath (with number if necessary)
            filepath_w_ext = numbered_filepath_w_ext

            metadata = PngInfo()
            if positive_optional is not None:
                metadata.add_text("positive", positive_optional)

            if negative_optional is not None:
                metadata.add_text("negative", negative_optional)

            if seed_optional is not None:
                metadata.add_text("seed", str(seed_optional))

            if other_optional is not None:
                metadata.add_text("other", str(other_optional))

            if include_workflow is True:
                if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            img.save(filepath_w_ext, pnginfo=metadata, optimize=True)
            img_count += 1
        return filepath_w_ext
    
    def print_debug(self, string):
        if self.is_debug:
            print(string)

    
class BKGetLastFolderName:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "path": ("STRING", {"default": f'', "multiline": False}),
            },
            "optional": {
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)  # This specifies that the output will be text
    RETURN_NAMES = ("path", "foldername",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Save Image"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, path):
        return float("nan")

    
    def process(self, path):
        if not path:
            return (path, "")

        # Normalize path (handles Windows / Unix / mixed slashes)
        norm_path = os.path.normpath(path)

        # If path ends with a file, get its parent directory
        if os.path.splitext(norm_path)[1]:
            foldername = os.path.basename(os.path.dirname(norm_path))
        else:
            foldername = os.path.basename(norm_path)

        return (path, foldername)

##################################################################################################################
# BK Get Matching Mask
##################################################################################################################

class BKGetMatchingMask:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.min_mask_size = 64
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "all_masks": ("MASK",),
                "ref_mask": ("MASK",),
                "notify_on_no_match": ("BOOLEAN", {"default": True, "tooltip": "If true, will throw an error if the ref mask is missing or empty. Will also display name of image."}),
            },
            "optional": {
                "user_mask": ("MASK",),
                "image_name": ("STRING",{ "multiline": True, "forceInput": True, "tooltip":"Image name displayed in error message if ref image is missing. (Optional)"}, ),
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("MASK", "MASK", "MASK", "BOOLEAN")  # This specifies that the output will be text
    RETURN_NAMES = ("matching_mask", "non_matching_mask", "combined_mask", "is_mask_found")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Get Matching Mask"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, all_masks, ref_mask, notify_on_no_match, image_name = "UNKNOWN", user_mask = None):
        return float("nan")

    def replace_nontype_with_default_comfyui_mask(self, mask):
        if mask is None:
            return self.create_empty_opaque_3d_image(self.min_mask_size, self.min_mask_size)
        else:
            return mask
        
    def process(self, all_masks, ref_mask, notify_on_no_match, image_name = "UNKNOWN", user_mask = None):
        print_debug_header(self.is_debug, "BK GET MATCHING MASK")
        
        # Prevent errors by fixing NoneType mask values to standard ComfyUI default empty mask
        user_mask  = self.replace_nontype_with_default_comfyui_mask(user_mask)
        all_masks  = self.replace_nontype_with_default_comfyui_mask(all_masks)
        ref_mask  = self.replace_nontype_with_default_comfyui_mask(ref_mask)

        # Ensure all masks are 3D and not 4D
        user_mask = self.convert_4d_to_3d(user_mask)
        ref_mask = self.convert_4d_to_3d(ref_mask)
        all_masks = self.convert_4d_to_3d(all_masks)

        # Flatten masks that are a set of masks instead of one mask for processing
        if self.image_batch_size(user_mask) > 1:
            print(f"BKGetMathchingMask: WARNING: user_mask was a set of masks, not a single mask. Flattening masks to one mask.")
            self.flatten_all_batched_masks_to_one_mask(user_mask)

        if self.image_batch_size(ref_mask) > 1:
            print(f"BKGetMathchingMask: WARNING: ref_mask was a set of masks, not a single mask. Flattening masks to one mask.")
            ref_mask = self.flatten_all_batched_masks_to_one_mask(ref_mask)

        binary_ref_mask = self.convert_to_binary_mask(ref_mask)
        

        list_of_non_matching_masks = []
        found_mask = None

        # NON-empty ref mask
        non_matching_count = 0
        for mask in all_masks:
            binary_mask = self.convert_to_binary_mask(mask)
            if torch.all(binary_ref_mask <= binary_mask):
                self.print_debug(f"Found a matching mask!")
                found_mask = binary_mask.unsqueeze(0)
            else:
                non_matching_count += 1
                list_of_non_matching_masks.append(binary_mask)
        
        
        
        # If any masks were found that did not match, combine them, else return an empty mask
        if len(list_of_non_matching_masks) > 0:
            non_matching_masks = self.recombine_a_list_of_masks(list_of_non_matching_masks)
            non_matching_masks = self.flatten_all_batched_masks_to_one_mask(non_matching_masks)
            combined_mask = self.combine_masks(user_mask, non_matching_masks)
        else:
            non_matching_masks = self.create_empty_opaque_3d_image(self.min_mask_size, self.min_mask_size)
            combined_mask = user_mask

        # If a matching mask was not found, return an empty mask
        is_found = True
        if found_mask is None:
            if notify_on_no_match:
                ValueError(f"BKGetMathchingMask: No matching mask found, and notify was set to true.")
            is_found = False
            found_mask = self.create_empty_opaque_3d_image(self.min_mask_size, self.min_mask_size)

        
        

        self.print_debug(f"found_mask.size()[{found_mask.size()}]")
        self.print_debug(f"non_matching_masks.size()[{non_matching_masks.size()}]")
        self.print_debug(f"combined_mask.size()[{combined_mask.size()}]")
        self.print_debug(f"non_matching_count[{non_matching_count}]")

        print_debug_bar(self.is_debug)
        return (found_mask, non_matching_masks, combined_mask, is_found)
    
    def combine_masks(self, user_mask, non_matching_mask):
        # Ensure the masks are tensors
        if not isinstance(user_mask, torch.Tensor) or not isinstance(non_matching_mask, torch.Tensor):
            raise TypeError("Both mask1 and mask2 must be torch tensors.")
        
        # Check if the masks have the same shape
        if user_mask.shape != non_matching_mask.shape:
            print(f"BKGetMatchingMask: WARNING: Could not merge the user_mask and the non_matching masks. They have different sizes. They must be the same size to merge them. Returning only the user_mask.")
            return user_mask

        # Combine the masks by taking the element-wise maximum
        combined_mask = torch.maximum(user_mask, non_matching_mask)

        return combined_mask
    
    def recombine_a_list_of_masks(self, masks):
        return torch.stack(masks)
    
    
    def print_debug(self, string):
        if self.is_debug:
            print(string)
    
    def image_batch_size(self, image_3d_4d):
        return image_3d_4d.size()[0]
    
    def create_empty_opaque_3d_image(self, height, width):
        # NOTE [Batch Size, Height, Width]
        # use the reference image to get batch size and channels
        #create a empty black / transparent image the size of the crop box
        return torch.zeros(( 1, 
                            height, 
                            width), 
                            dtype=torch.float32)
    
    def convert_4d_to_3d(self, tensor):
        if len(tensor.size()) == 4:
            return tensor.squeeze(3)
        else:
            return tensor
        
    def flatten_all_batched_masks_to_one_mask(self, masks):
        return masks.max(dim=0, keepdim=True)[0]
    
    def convert_to_binary_mask(self, mask):

        # Ensure the mask is a tensor
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Input mask must be a torch.Tensor.")

        # Create a new mask with values set to 1 if the pixel is greater than 0
        mask[mask > 0] = 1

        return mask


##################################################################################################################
# BK Body Ratios
##################################################################################################################


class BKBodyRatios:
    def __init__(self):
        # Initialize running averages
        self.avgheight = 0
        self.avgwidthratio = 0
        self.avgspaceratio = 0
        self.avgspacevswidthratio = 0
        self.count = 0  # To track the number of valid masks processed

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                
            },
            "optional": {
                "image_name": ("STRING",{ "multiline": True, "forceInput": True, "tooltip":"Image name displayed in error message if ref image is missing. (Optional)"}, ),
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK","STRING", "INT", "FLOAT", "FLOAT", "INT", "FLOAT", "FLOAT", "FLOAT", "STRING", "BOOLEAN", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("image", "mask", "orig_image_name", "height", "widthratio", "spaceratio", "avgheight", "avgwidthratio", "avgspaceratio","avgspacevswidthratio", "name_w_values", "is_save", "status")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Body Ratios"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, image, mask, image_name="NA"):
        return float("nan")



    def process(self, image, mask, image_name="NA", save=False):
        # Check if mask is None, empty, or contains more than one mask
        if mask is None or mask.size(0) == 0 or mask.size(0) > 1:
            return image, mask, image_name, None, None, None, self.avgheight, self.avgwidthratio, self.avgspaceratio, self.avgspacevswidthratio, "NA", False, "Invalid mask input"

        # Assuming mask is a tensor, we can find its bounding box
        mask = mask[0]  # Work with the first mask (assuming there's only one)

        # Find bounding box (this is an example, you can adapt it depending on how masks are represented)
        nonzero_coords = torch.nonzero(mask)
        if nonzero_coords.size(0) == 0:
            return image, mask, image_name, None, None, None, self.avgheight, self.avgwidthratio, self.avgspaceratio, self.avgspacevswidthratio, "NA", False, "Empty mask"

        top_left = nonzero_coords.min(dim=0)[0]
        bottom_right = nonzero_coords.max(dim=0)[0]

        height = bottom_right[0] - top_left[0]
        width = bottom_right[1] - top_left[1]

        # Calculate ratios
        width_ratio = width.item() / image.size(1) if image.size(1) > 0 else 0
        height = height.item()  # Convert to native Python number
        top_left_y = top_left[0].item()  # Convert to native Python number

        # New spacer ratio: height of bounding box relative to the distance from the top of the image
        spaceratio = (image.size(1) - height) / height if height > 0 else 0
        spacewidthratio = spaceratio / width_ratio if width_ratio > 0 else 0


        print(f" image.size(1)[{image.size(1)}] - height[{height}]  = image.size(1) - height[{image.size(1) - height}]")
        # Don't include images where the subject's head or hair are clipping through the top of the image
        if(image.size(1) - height) > 1:
            # Update running averages
            self.count += 1
            self.avgheight += (height - self.avgheight) / self.count
            self.avgwidthratio += (width_ratio - self.avgwidthratio) / self.count
            self.avgspaceratio += (spaceratio - self.avgspaceratio) / self.count
            self.avgspacevswidthratio += (spacewidthratio - self.avgspacevswidthratio) / self.count
            output_name = f"{image_name}_{height}_{width_ratio:.4f}_{spaceratio:.4f}_{self.avgspaceratio:.4f}"
            status = f"name,{image_name},height,{height},widthratio,{width_ratio},spaceratio,{spaceratio},avgheight,{self.avgheight:.4f},avgwidthratio,{self.avgwidthratio:.3f},avgspaceratio,{self.avgspaceratio:.3f},avgspacevswidthratio,{self.avgspacevswidthratio:.4f}"
            return image, mask, image_name, height, width_ratio, spaceratio, self.avgheight, self.avgwidthratio, self.avgspaceratio, self.avgspacevswidthratio, output_name, True, status
        else:
            return image, mask, image_name, None, None, None, self.avgheight, self.avgwidthratio, self.avgspaceratio, self.avgspacevswidthratio, "NA", False, "Head Proturding mask"


##################################################################################################################
# BK Crop And Pad
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKCropAndPad:
    def __init__(self):
        self.is_pad_left = False
        self.is_pad_top = False
        self.minimum_image_size = 64
        # This ratio comes from testing 30,000+ real single subject images TBD
        # self.head_clearance_ratio = 0.18 
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "image": ("IMAGE",),
                "person_mask": ("MASK",),
                "desired_size": ("INT",),
                "outpaint_padding": ("INT",),
            },
            "optional": {
                "image_name": ("STRING",{ "multiline": True, "forceInput": True, "tooltip":"Image name displayed in error message if ref image is missing. (Optional)"}, ),
                "user_mask": ("MASK",),
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK","MASK","MASK","MASK","BOOLEAN")  # This specifies that the output will be text
    RETURN_NAMES = ("cropped_image", "cropped_person_mask", "cropped_user_mask", "cropped_outpaint_mask", "combined_mask", "is_need_outpaint")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Crop And Pad"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, image, person_mask, desired_size, outpaint_padding, image_name = "NA", user_mask = None):
        print(f"BKCropAndPad IS_CHANGED called")
        return float("nan")
    
    def largest_dimension_of_image(self, image):
        return max(self.image_height(image), self.image_width(image))
    
    def smallest_dimension_of_image(self, image):
        return min(self.image_height(image), self.image_width(image))
    
    def flatten_all_batched_masks_to_one_mask(self, masks):
        return masks.max(dim=0, keepdim=True)[0]
    


# NOTE: image tensor coordinates origin is at TOP-LEFT corner
    def process(self, image, person_mask, desired_size, outpaint_padding, image_name="NA", user_mask=None):
        self.print_debug_header(self.is_debug,"BK CROP AND PAD")
        # Initialize output variables and constants

        if self.image_batch_size(person_mask) > 1:
            #TODO: Enable mask flatening of multi masks.
            print(f"BKCropAndPad: WARNING: Recieved multiple batched masks for person masks. Flattening them down to one mask.")
            person_mask = self.flatten_all_batched_masks_to_one_mask(person_mask)

        self.print_debug(f"desired_size before clamp[{desired_size}]")

        max_desired_size = self.largest_dimension_of_image(image)
        min_desired_size = self.minimum_image_size
        
        

        # Limit the desired size to the minimum image size of comfyUI (64) and the maximum size of the image
        desired_size = self.clamp_desired_size(desired_size, min_desired_size, max_desired_size)
        self.print_debug(f"desired_size afterclamp[{desired_size}]")
        
        self.print_debug(f"Desired Size: {desired_size}")

        self.print_debug(f"Orig Image [height x width][{self.image_height(image)} x {self.image_width(image)}]")

        # Check for empty person mask
        self.check_for_empty_mask(person_mask, image_name)

        # Get the bounding box of the person from the mask
        person_bounding_box = self.get_person_bounding_box(person_mask)
        image = self.draw_bounding_box_on_image(image, person_bounding_box, color=(1.0, 0.0, 0.0))
        
        crop_box = self.get_crop_bounding_box(person_bounding_box, image, desired_size)
        image = self.draw_bounding_box_on_image(image, crop_box, color=(0.0, 1.0, 0.0))

        self.print_box(person_bounding_box, label="PersonBox")
        self.print_box_values(crop_box, label="Crop Box")

        is_need_vertical_outpaint = self.is_crop_box_taller_than_image(image, crop_box)
        is_need_horizontal_outpaint = self.is_crop_box_wider_than_image(image, crop_box)

        if is_need_horizontal_outpaint:
            self.flip_horizontal_pad_side()

        if is_need_vertical_outpaint:
            self.flip_vertical_pad_side()

        is_need_outpaint = is_need_horizontal_outpaint or is_need_vertical_outpaint

        cropped_image = self.crop_4d_image_and_pad_white(image, crop_box)
        cropped_person_mask = self.crop_3d_mask_and_pad_opaque(person_mask, crop_box)

        # resize empty outpaint mask by padding amount in the direction in the axis of the outpaint
        self.print_debug(f"is_need_vertical_outpaint: {is_need_vertical_outpaint}")
        self.print_debug(f"is_need_horizontal_outpaint: {is_need_horizontal_outpaint}")
        self.print_debug(f"height: {self.image_height(image)}")
        self.print_debug(f"width: {self.image_width(image)}")
        height_for_padding = self.add_padding(self.image_height(image), outpaint_padding, is_need_vertical_outpaint)
        width_for_padding = self.add_padding(self.image_width(image), outpaint_padding, is_need_horizontal_outpaint)
        self.print_debug(f"height: {height_for_padding}")
        self.print_debug(f"width: {width_for_padding}")

        empty_outpaint_center_part_mask = self.create_empty_3d_transparent_mask(height_for_padding, width_for_padding)
        cropped_outpaint_mask = self.crop_3d_mask_and_pad_transparent(empty_outpaint_center_part_mask, crop_box)

        if self.is_mask_empty(user_mask):
            self.print_debug(f"user_mask empty")
            combined_mask = cropped_outpaint_mask
            cropped_user_mask = self.create_empty_comfyui_mask()
        else:
            self.print_debug(f"has user_mask")
            cropped_user_mask = self.crop_3d_mask_and_pad_opaque(user_mask, crop_box)
            combined_mask = self.combine_masks(cropped_user_mask, cropped_outpaint_mask)

        # Ensure all masks are 3D tensors before returning 
        cropped_person_mask = self.convert_4d_to_3d(cropped_person_mask)
        cropped_user_mask = self.convert_4d_to_3d(cropped_user_mask)
        cropped_outpaint_mask = self.convert_4d_to_3d(cropped_outpaint_mask)
        combined_mask = self.convert_4d_to_3d(combined_mask)

        

        print_debug_bar(self.is_debug)
        return cropped_image, cropped_person_mask, cropped_user_mask, cropped_outpaint_mask, combined_mask, is_need_outpaint
    
    def add_padding(self, original_size, padding, is_need_outpaint):
        if is_need_outpaint:
            return original_size - padding
        else:
            return original_size

    def draw_bounding_box_on_image(self, image, box, color=(1.0, 1.0, 0.0)):
        
        if not self.is_debug:
            return image
        
        # Draw Center Point in green
        image = self.draw_circle_rgb(image, self.box_center_x(box), self.box_center_y(box), color, 10)  # Green dot at center of person box
        
        #Draw Anchor Point in blue
        image = self.draw_circle_rgb(image, self.box_left(box), self.box_top(box), color, 20)  # Blue dot at top-left of person box

        return image

    def box_center_x(self, box):
        return box[1] + box[3] // 2
    
    def box_center_y(self, box):
        return box[0] + box[2] // 2
    
    def is_crop_box_taller_than_image(self, image, crop_box):
        self.print_debug(f"if image_height[{self.image_height(image)}] > box_height[{self.box_height(crop_box)}] vert outpaint needed") 
        return self.box_height(crop_box) > self.image_height(image)
    
    def is_crop_box_wider_than_image(self, image, crop_box):
        self.print_debug(f"if image_width[{self.image_width(image)}] > box_width[{self.box_width(crop_box)}] horiz outpaint needed")
        return self.box_width(crop_box) > self.image_width(image)
    
    def clamp_desired_size(self, desired_size, minimum, maximum):
        return max(min(desired_size, maximum), minimum)

    def convert_4d_to_3d(self, tensor):
        if len(tensor.size()) == 4:
            return tensor.squeeze(3)
        else:
            return tensor
        
    def print_box(self, box, label="Box"):
        self.print_box_headers()
        self.print_box_values(box, label=label)

    
    def print_image_resolution(self, image_3d_4d, label="Image"):
        self.print_debug(f"{label:60}: {self.image_height(image_3d_4d):>7} {self.image_width(image_3d_4d):>7}")
    
    def print_box_headers(self):
        self.print_debug(f"{'Label':60}: {'Height':>7} {'Width':>7} {'Top':>7} {'Left':>7}")

    def print_box_values(self, box, label="Box"):
        self.print_debug(f"{label:60}: {self.box_height(box):>7} {self.box_width(box):>7} {self.box_left(box):>7} {self.box_top(box):>7}")
    
    def create_empty_3d_opaque_mask(self, height, width):

        self.print_box_values((0, 0, height, width), label="Empty Opaque Mask Size")

        # Create a 3D mask tensor filled with zeros (transparent)
        mask = torch.ones((1, height, width), dtype=torch.float32)
        return mask

    def create_empty_3d_transparent_mask(self, height, width):
        self.print_box_values((0, 0, height, width), label="Empty OUTPAINT Mask Size")
        # Create a 3D mask tensor filled with zeros (transparent)
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        return mask
    
    def is_mask_empty(self, mask):
        if mask is None:
            return True
        elif not mask.any():
            return True
        return False

    def create_empty_comfyui_mask(self):
        # Create a 3D mask tensor filled with zeros (transparent)
        print ("Creating default empty comfyui mask 64x64")

        return self.create_empty_3d_transparent_mask(self.minimum_image_size, self.minimum_image_size)


    def combine_masks(self, user_mask, outpaint_mask):
        # Resize outpaint_mask to match user_mask size if needed
        if user_mask.size() != outpaint_mask.size():
            outpaint_mask = F.interpolate(outpaint_mask.unsqueeze(0).unsqueeze(0), size=user_mask.shape[1:], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Combine two masks by taking the highest transparency value at each pixel
        combined_mask = torch.max(user_mask, outpaint_mask)
        return combined_mask

    def check_for_empty_mask(self, person_mask, image_name):
        if not torch.any(person_mask > 0.5):
            raise ValueError(f"Person mask is empty or null [{image_name}]. Is there a person in the image?")
        
    def check_for_empty_image(self, image):
        if image is None or image.numel() == 0:
            raise ValueError(f"Image is empty or null.")

    def box_left(self, box):
        return box[1]
    def box_top(self, box):
        return box[0]

    def get_person_bounding_box(self, person_mask):
        # NOTE: Box format: (left, top, height, width)

        # NOTE: image tensor coordinates origin is at TOP-LEFT corner
        if not torch.any(person_mask):
            raise ValueError("The person mask is empty or has no non-zero elements.")
        


        # Get non-zero indices of the person_mask (where the person is located)
        # Person_mask is a set of masks, not a single mask, so we take the first one
        # each indice is in (y, x) format
        non_zero_indices = torch.nonzero(person_mask[0])

        # If no non-zero values exist, raise an error
        if non_zero_indices.numel() == 0:
            raise ValueError("No person found in the mask")

        # Find the minimum and maximum x and y values for the bounding box
        x_min = non_zero_indices[:, 1].min()  # Minimum x value (left of the bounding box)
        y_min = non_zero_indices[:, 0].min()  # Minimum y value (top of the bounding box)
        x_max = non_zero_indices[:, 1].max()  # Maximum x value (right of the bounding box)
        y_max = non_zero_indices[:, 0].max()  # Maximum y value (bottom of the bounding box)

        # NOTE: image tensor coordinates origin is at TOP-LEFT corner
        left = x_min.item()
        top = y_min.item()
        width = x_max.item() - x_min.item() + 1
        height = y_max.item() - y_min.item() + 1

        # NOTE: image tensor coordinates origin is at TOP-LEFT corner
        return (top, left, height, width)

    def box_horizontal_center(self, box):
        return self.box_left(box) + (self.box_width(box) // 2)

    def box_top_anchor_point(self, box):
        return self.box_top(box) + (self.box_height(box) // 2)

    def image_height(self, image_3d_4d):
        return image_3d_4d.size()[1]

    def image_width(self, image_3d_4d):
        return image_3d_4d.size()[2]

    def get_crop_bounding_box(self, person_bounding_box, image, desired_size):

        # NOTE: Box format: (left, top, height, width)

        # Get the minimum the crop zone has to be to fully fit the person inside or to meet the
        #    desired size
        crop_zone_min_size = max(self.box_width(person_bounding_box), self.box_height(person_bounding_box), desired_size, 0)
        
        # get the maximum the crop zone can be while still being able to fully fit in the image
        crop_zone_max_size = min(self.image_width(image), self.image_height(image))

        # it is possible for the "crop_zone_min_size" to be larger than the max, if the image is a vertical image
        #    it is something like a full size shot where the person is really tall, the minimum width will exceed
        #    the image width. So we take whichever one is bigger. In this case, we will have to outpaint.
        size = max(crop_zone_min_size, crop_zone_max_size)

        # Find the center of the crop zone
        #    If the cropzone fits inside the image, center on the person
        #    If the cropzone is larger than the image, pad to one side
        #    If the cropzone is equal to the image, then set to the center
        # NOTE: the flipper methods are pass by reference, and called inside of the _find_center_of_box method
        left = self.find_box_anchor_point(self.image_width(image), 
                                                      size, 
                                                      self.box_left(person_bounding_box),
                                                      self.box_width(person_bounding_box))

        top = self.find_box_anchor_point(self.image_height(image), 
                                                      size, 
                                                      self.box_top(person_bounding_box),
                                                      self.box_height(person_bounding_box))

        return (top, left, size, size)
    
    def draw_circle_rgb(self,
        images: np.ndarray,
        x: int,
        y: int,
        color: tuple,
        radius: int
    ) -> np.ndarray:
        """
        Draw a filled circle on a batch of RGB images.

        Parameters
        ----------
        images : np.ndarray
            Image tensor of shape [batch_size, height, width, 3]
        x : int
            X coordinate (column)
        y : int
            Y coordinate (row)
        color : tuple or list
            RGB color (R, G, B)
        radius : int
            Circle radius in pixels

        Returns
        -------
        np.ndarray
            Modified image tensor
        """



        assert images.ndim == 4 and images.shape[-1] == 3, \
            "Expected image shape [B, H, W, 3]"

        batch_size, height, width, _ = images.shape

        # Create coordinate grid
        yy, xx = np.ogrid[:height, :width]

        # Circle mask
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2

        # Apply color to all images in batch
        for c in range(3):
            images[:, mask, c] = color[c]

        return images

    def find_box_anchor_point(self, image_size, box_size, person_anchor_point, person_size):

        if(image_size == box_size):
            self.print_debug(f"Box size [{box_size}] is same as image size [{image_size}]. Box aligned to top or left.")
            return 0

        # if the box is going outside the bounds of the image's width then the person will always
        #    be inside, so we align it to one side or the other for outpadding, alternating as we go
        if(self.is_box_too_big_to_fit_in_image(image_size, box_size)):
            self.print_debug(f"Box size [{box_size}] is greater image size [{image_size}]. Box aligned to top or left. Outpainting needed.")
            return 0

        # Move the box as close to the person center as possible while keeping it within the image bounds 
        self.print_debug(f"Box size [{box_size}] is less than image size [{image_size}]. Centering box on person within bounds of image.")
        return self.get_closest_anchor_point_within_bounds(image_size, box_size, person_anchor_point, person_size)

    def is_box_too_big_to_fit_in_image(self, image_size, box_size):
        if box_size > image_size:
            return True
        return False
    
    def get_closest_anchor_point_within_bounds(self, image_size, box_size, person_anchor_point, person_size):
        self.print_debug(f"================================== GET CLOSEST ANCHOR POINT ==================================")
        # NOTE: image tensor coordinates origin is at TOP-LEFT corner
        person_center = person_anchor_point + (person_size // 2)
        self.print_debug(f"person_anchor_point[{person_anchor_point}] + (person_size[{person_size}] // 2) = person_center[{person_center}]")
        ideal_anchor_point = person_center - (box_size // 2)
        self.print_debug(f"ideal_anchor_point = person_center[{person_center}] - (box_size[{box_size}] // 2) = ideal_anchor_point[{ideal_anchor_point}]")

        lower_bound = 0
        self.print_debug(f"lower_bound = 0")
        upper_bound = image_size - box_size
        self.print_debug(f"upper_bound = image_size[{image_size}] - box_size[{box_size}] = upper_bound[{upper_bound}]")

        result = self.clamp(ideal_anchor_point, lower_bound, upper_bound)
        self.print_debug(f"clamp(ideal_anchor_point[{ideal_anchor_point}], lower_bound[{lower_bound}], upper_bound[{upper_bound}]) = result[{result}]")
        # NOTE: image tensor coordinates origin is at TOP-LEFT corner
        return result

    def clamp(self, value, lower, upper):
        return max(min(value, upper), lower)

    
    def bound_box_within_image(self, box, image_3d_4d):

        self.print_debug(f"self.box_left(box)[{self.box_left(box)}]")
        self.print_debug(f"MIN: [0]")
        self.print_debug(f"MAX: self.image_width(image_3d_4d) - self.box_width(box)[{self.image_width(image_3d_4d) - self.box_width(box)}]")
        left = self.clamp(self.box_left(box), 
                          0, 
                          self.image_width(image_3d_4d) - self.box_width(box))
        
        top = self.clamp(self.box_top(box), 
                         0,
                         self.image_height(image_3d_4d) - self.box_height(box))
        
        return (top, left, self.box_height(box), self.box_width(box))

    def crop_4d_image_using_box(self, image_4d, box):
        
        crop_zone = self.bound_box_within_image(box, image_4d)

        self.print_box_values(box, label="box")
        self.print_box_values(crop_zone, label="Crop Zone")
        self.print_image_resolution(image_4d, label="Image Before Crop")

        # Slice the image using the bounding box coordinates
        cropped_image = image_4d[:, 
                                 self.box_top(crop_zone)  : self.box_top(crop_zone) + self.box_height(crop_zone),
                                 self.box_left(crop_zone) : self.box_left(crop_zone) + self.box_width(crop_zone), 
                                 :]

        self.print_image_resolution(cropped_image, label="Image After Crop")

        return cropped_image

    def crop_3d_image_using_box(self, image_3d, box):

        crop_zone = self.bound_box_within_image(box, image_3d)

        self.print_box_values(box, label="box")
        self.print_box_values(crop_zone, label="Crop Zone")
        self.print_image_resolution(image_3d, label="Image Before Crop")

        # Slice the image using the bounding box coordinates
        cropped_image = image_3d[:, 
                            self.box_top(crop_zone)  : self.box_top(crop_zone) + self.box_height(crop_zone),
                            self.box_left(crop_zone) : self.box_left(crop_zone) + self.box_width(crop_zone),]
        
        self.print_image_resolution(cropped_image, label="Image After Crop")

        return cropped_image

    def create_empty_transparent_3d_image(self, height, width):
        # NOTE [Batch Size, Height, Width]
        # use the reference image to get batch size and channels
        #create a empty white / opaque image the size of the crop box
        return torch.ones(( 1, 
                            height, 
                            width), 
                            dtype=torch.float32)
    
    def image_4d_channel(self, image_4d):
        return image_4d.size()[3]
    
    def create_empty_white_4d_image(self, height, width):
        # NOTE [Batch Size, Height, Width, Channels]
        # use the reference image to get batch size and channels
        #create a empty white / opaque image the size of the crop box
        return torch.ones(( 1,
                            height, 
                            width, 
                            3), 
                            dtype=torch.float32)

    
    def box_width(self, box):
        # NOTE: box = (left, top, width, height)
        return box[3]
    
    def box_height(self, box):
        # NOTE: box = (left, top, width, height)
        return box[2]
    
    def image_batch_size(self, image_3d_4d):
        return image_3d_4d.size()[0]

    def create_empty_opaque_3d_image(self, crop_box, match_3d_image):

        # NOTE image_ref_4d must be a 4 dimensional shape [Batch Size, Width, Height, Channels]
        # use the reference image to get batch size and channels
        #create a empty black / transparent image the size of the crop box
        return torch.zeros((self.image_batch_size(match_3d_image), 
                                        self.box_height(crop_box), 
                                        self.box_width(crop_box)), 
                                        dtype=torch.float32)

    def print_debug(self, string):
        if self.is_debug:
            print(string)

    def crop_4d_image_and_pad_white(self, image, crop_box):
        self.print_debug(f"================================== CROP 4D IMAGE WHITE ==================================")
        # NOTE: image must be a 4 dimensional shape [Batch Size, Height, Width, Channels]
        # NOTE: ComfyUI images are 4D tensors [Batch Size, Height, Width, Channels]
        return self.paste_4d_image_into_another(
            self.create_empty_white_4d_image(self.box_height(crop_box), self.box_width(crop_box)),
            self.crop_4d_image_using_box(image, crop_box))

    def crop_3d_mask_and_pad_transparent(self, mask, crop_box):
        self.print_debug(f"================================== CROP 3D MASK TRANSPARENT ==================================")
        # NOTE: orig_mask must be a 3 dimensional shape [Batch Size, Height, Width]
        # NOTE: ComfyUI masks are 3D tensors [Batch Size, Height, Width]

        # if the mask is 4D, remove channel dimension to make it 3D
        if len(mask.size()) == 4:
            mask = mask.squeeze(3)

        # Crop and pad 3D mask
        cropped_and_padded_mask = self.paste_3d_mask_into_another(
            self.create_empty_transparent_3d_image(self.box_height(crop_box), self.box_width(crop_box)),
            self.crop_3d_image_using_box(mask, crop_box))

        return cropped_and_padded_mask 
    
    def remove_image_channel_dimension(self, image_4d):
        # Remove channel dimension to make it 3D
        return image_4d.squeeze(3)
    
    def crop_3d_mask_and_pad_opaque(self, image, crop_box):
        self.print_debug(f"================================== CROP 3D MASK OPAQUE ==================================")
        # NOTE: orig_mask must be a 3 dimensional shape [Batch Size, Height, Width]
        # NOTE: ComfyUI masks are 3D tensors [Batch Size, Height, Width]

        # if the mask is 4D, remove channel dimension to make it 3D
        if self.image_d_size(image) == 4:
            image = self.remove_image_channel_dimension(image)

        # Crop and pad as 4D image
        cropped_and_padded_mask = self.paste_3d_mask_into_another(
            self.create_empty_opaque_3d_image(crop_box, image), 
            self.crop_3d_image_using_box(image, crop_box))
        
        return cropped_and_padded_mask
    
    def image_d_size(self, image):
        return len(image.size())
    
    def flip_horizontal_pad_side(self):
        self.print_debug(f"is_pad_left before flip[{self.is_pad_left}]")
        self.is_pad_left = not self.is_pad_left
        self.print_debug(f"is_pad_left after flip[{self.is_pad_left}]")
    
    def flip_vertical_pad_side(self):
        self.print_debug(f"is_pad_top before flip[{self.is_pad_top}]")
        self.is_pad_top = not self.is_pad_top
        self.print_debug(f"is_pad_top after flip[{self.is_pad_top}]")
    
        
    def get_crop_coordinates(self, dest_image, src_image):
        #self.print_image_resolution(src_image, "src_image: get_crop_coordinates")
        #self.print_image_resolution(dest_image, "dest_image: get_crop_coordinates")
        x0 = self.get_x0_anchor_point_to_toggle_padding_side(dest_image, src_image)
        y0 = self.get_y0_anchor_point_to_toggle_padding_side(dest_image, src_image)
        xn = x0 + self.image_width(src_image)
        yn = y0 + self.image_height(src_image)
        return y0, x0, yn, xn
    
    def paste_4d_image_into_another(self, dest_image, src_image):

        self.print_debug(f"================================== PASTE 4D IMAGE ==================================")

        #self.print_image_resolution(src_image, "src_image: paste_4d_image_into_another")
        #self.print_image_resolution(dest_image, "dest_image: paste_4d_image_into_another")

        y0, x0, yn, xn = self.get_crop_coordinates(dest_image, src_image)

        self.print_image_resolution(src_image, label="Src Image To Paste")
        self.print_box_values((y0, x0, yn - y0, xn - x0), label="Paste Box")
        self.print_image_resolution(dest_image, label="Dst Img Bef Paste")

        # Slice the src image into the dest image. The dest_mask_4d indicies are (batch, x0 - xn, y0 - yn, channels)
        #    We actually specify a box using y0 - yn, x0 - xn coordinates in the dest_mask_4d
        #                         y0     :      yn  , x0    :      xn 
        #    This box has to be equal to the dimensions of the src_mask_4d, this is why we use the width and height from src_mask_4d
        #    and left_anchor_point and top_anchor_point determine the location inside of dest_mask_4d where we paste the src_mask_4d
        
        dest_image[:, y0:yn, x0:xn, :] = src_image

        return dest_image

    def paste_3d_mask_into_another(self, dest_image, src_image):
        self.print_debug(f"================================== PASTE 3D MASK ==================================")

        y0, x0, yn, xn = self.get_crop_coordinates(dest_image, src_image)

        self.print_image_resolution(src_image, label="Src Image To Paste")
        self.print_box_values((y0, x0, yn - y0, xn - x0), label="Paste Box")
        self.print_image_resolution(dest_image, label="Dst Img Aft Paste")

        # Slice the src image into the dest image. The dest_mask_3d indicies are (batch, x0:xn, y0:yn)
        #    We actually specify a box using  y0 - yn, x0 - xn coordinates in the dest_mask_3d
        #               y0     :      yn            ,         x0    :      xn
        #    This box has to be equal to the dimensions of the src_mask_3d, this is why we use the width and height from src_mask_3d
        #    and left_anchor_point and top_anchor_point determine the location inside of dest_mask_3d where we paste the src_mask_3d


        dest_image[:, y0:yn, x0:xn] = src_image
        return dest_image

    def get_x0_anchor_point_to_toggle_padding_side(self, dest_image, src_image):        
        #self.print_image_resolution(src_image, "src_image: get_x0_anchor_point_to_toggle_padding_side")
        #self.print_image_resolution(dest_image, "dest_image: get_x0_anchor_point_to_toggle_padding_side")
        if self.is_source_image_width_smaller_than_dest(dest_image, src_image):
            if not self.is_pad_left:
                self.print_debug(f"Padding LEFT side of image.")
                return 0
            else:
                self.print_debug(f"Padding RIGHT side of image.")
                return self.get_anchor_point_to_paste_image_on_right_side(dest_image, src_image)
        return 0
    
    def get_anchor_point_to_paste_image_on_right_side(self, dest_image, src_image):
        return self.image_width(dest_image) - self.image_width(src_image)

    def is_source_image_width_smaller_than_dest(self, dest_image, src_image):
        #self.print_image_resolution(src_image, "src_image: is_source_image_width_smaller_than_dest")
        #self.print_image_resolution(dest_image, "dest_image: is_source_image_width_smaller_than_dest")
        src_image_width = self.image_width(src_image)
        dest_image_width = self.image_width(dest_image)
        is_src_img_width_smaller_than_dest = src_image_width < dest_image_width

        self.print_debug(f"Is the width src < dest [{src_image_width} < {dest_image_width}]: {is_src_img_width_smaller_than_dest}")
        return  is_src_img_width_smaller_than_dest

    def get_y0_anchor_point_to_toggle_padding_side(self, dest_image, src_image):
        #self.print_image_resolution(src_image, "src_image: get_y0_anchor_point_to_toggle_padding_side")
        #self.print_image_resolution(dest_image, "dest_image: get_y0_anchor_point_to_toggle_padding_side")
        if self.is_source_image_height_smaller_than_dest(dest_image, src_image): 
            if self.is_pad_top:
                self.print_debug(f"Padding TOP side of image.")
                return 0
            else:
                self.print_debug(f"Padding BOTTOM side of image.")
                return self.get_anchor_point_to_paste_image_at_bottom(dest_image, src_image)
        return 0
    
    def get_anchor_point_to_paste_image_at_bottom(self, dest_image, src_image):
        return self.image_height(dest_image) - self.image_height(src_image)
    
    def is_source_image_height_smaller_than_dest(self, dest_image, src_image):
        #self.print_image_resolution(src_image, "src_image: is_source_image_height_smaller_than_dest")
        #self.print_image_resolution(dest_image, "dest_image: is_source_image_height_smaller_than_dest")
        src_image_height = self.image_height(src_image)
        dest_image_height = self.image_height(dest_image)
        is_src_img_height_smaller_than_dest = src_image_height < dest_image_height

        self.print_debug(f"Is the height src < dest [{src_image_height} < {dest_image_height}]: {is_src_img_height_smaller_than_dest}")
        return self.image_height(src_image) < self.image_height(dest_image)
   

##################################################################################################################
# BK Mask Square And Pad
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKMaskSquareAndPad:
    def __init__(self):
        self.minimum_image_size = 64
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "multi_masks": ("MASK",),
                "padding": ("INT", {"default": 0}),
            },
            "optional": {
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("MASK","INT","BOOLEAN")  # This specifies that the output will be text
    RETURN_NAMES = ("single_combined_mask","original_count","has_mask")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Mask Square And Pad"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, multi_masks, padding):
        return float("nan")


# NOTE: image tensor coordinates origin is at TOP-LEFT corner
    def process(self, multi_masks, padding):
        print_debug_bar(self.is_debug)
        #NOTE: Masks are [batch_size, height, width]

        if multi_masks is None:
            multi_masks = self.create_empty_opaque_3d_image(self.minimum_image_size, self.minimum_image_size)
            print("BK Mask Square And Pad: WARNING: Input mask was NONE. This will break other nodes. Correcting issue, and returning ComfyUI Default empty mask.")
            return (multi_masks, 0, False)
        
        # Ensure masks are 3 dimensional before processing. 4 dimensional masks are usually images.
        multi_masks = self.convert_4d_to_3d(multi_masks)
            

        count = self.image_batch_size(multi_masks)
        

        if self.is_mask_empty(multi_masks):
            return (multi_masks, count, False)
        
        all_modified_masks = []
        
        # We do this before we merge the masks, so it doesn't just create one large square mask over all masks.
        for mask in multi_masks:
            bool_mask = self.convert_mask_to_boolean(mask)
            squared_padded_masks = self.square_and_pad(bool_mask, padding)
            all_modified_masks.append(squared_padded_masks)
            

        all_modified_masks = self.recombine_a_list_of_masks(all_modified_masks)
        modified_mask = self.flatten_all_batched_masks_to_one_mask(all_modified_masks)



        has_mask = not self.is_mask_empty(modified_mask)


        print_debug_bar(self.is_debug)
        return (modified_mask, count, has_mask)
    
    def recombine_a_list_of_masks(self, masks):
        return torch.stack(masks)
    
    def is_mask_empty(self, mask):
        if mask is None:
            return True
        elif not mask.any():
            return True
        return False
    
    def print_debug(self, string):
        if self.is_debug:
            print(string)
    
    def image_batch_size(self, image_3d_4d):
        return image_3d_4d.size()[0]
    
    def create_empty_opaque_3d_image(self, height, width):
        # NOTE [Batch Size, Height, Width]
        # use the reference image to get batch size and channels
        #create a empty black / transparent image the size of the crop box
        return torch.zeros(( 1, 
                            height, 
                            width), 
                            dtype=torch.float32)
    
    def convert_4d_to_3d(self, tensor):
        if len(tensor.size()) == 4:
            return tensor.squeeze(3)
        else:
            return tensor
        
    def flatten_all_batched_masks_to_one_mask(self, masks):
        return masks.max(dim=0, keepdim=True)[0]
    
    def convert_mask_to_boolean(self, mask):

        # Ensure the mask is a tensor
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Input mask must be a torch.Tensor.")

        # Create a new mask with values set to 1 if the pixel is greater than 0
        mask[mask > 0] = 1

        return mask

    def square_and_pad(self, mask, padding):

        # Ensure the mask is a tensor
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Input mask must be a torch.Tensor.")

        # Find the indices of all non-zero pixels (i.e., the foreground pixels)
        non_zero_indices = torch.nonzero(mask)

        if non_zero_indices.size(0) == 0:
            # If there are no non-zero pixels, return the mask as is
            return mask

        # Get the min and max row and column indices to form the bounding box
        min_row = non_zero_indices[:, 0].min().item()
        max_row = non_zero_indices[:, 0].max().item()
        min_col = non_zero_indices[:, 1].min().item()
        max_col = non_zero_indices[:, 1].max().item()

        # Apply padding to the bounding box
        min_row = max(min_row - padding, 0)  # Ensure min_row doesn't go below 0
        max_row = min(max_row + padding, mask.shape[0] - 1)  # Ensure max_row doesn't exceed height
        min_col = max(min_col - padding, 0)  # Ensure min_col doesn't go below 0
        max_col = min(max_col + padding, mask.shape[1] - 1)  # Ensure max_col doesn't exceed width

        # Fill the bounding box area with 1
        mask[min_row:max_row+1, min_col:max_col+1] = 1

        return mask

##################################################################################################################
# BK Mask Test
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKMaskTest:
    def __init__(self):
        self.minimum_image_size = 64
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "mask": ("MASK",),
                "num_of_masks": (["Is Equal To", "Is Greater Than", "Is Less Than"], {"default":"Is Greater Than"}),
                "value": ("INT", {"default":0}),
            },
            "optional": {
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("MASK", "INT", "BOOLEAN", "BOOLEAN")  # This specifies that the output will be text
    RETURN_NAMES = ("mask", "count", "boolean_has_mask", "boolean_operation_result")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Mask Test"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, mask, num_of_masks, value):
        return float("nan")


# NOTE: image tensor coordinates origin is at TOP-LEFT corner
    def process(self, mask, num_of_masks, value):
        print_debug_header(self.is_debug, "BK MASK TEST")
        
        is_has_mask = not self.is_mask_empty(mask)
        count = self.image_batch_size(mask)

        if count == 1 and not is_has_mask:
            count = 0

        if num_of_masks == "Is Equal To":
            result = (count == value)
        elif num_of_masks == "Is Greater Than":
            result = (count > value)
        elif num_of_masks == "Is Less Than":
            result = (count < value)
        else:
            raise ValueError(f"Unsupported operation: {num_of_masks}")

        print_debug_bar(self.is_debug)
        return (mask, count, is_has_mask, result)
    
    def is_mask_empty(self, mask):
        if mask is None:
            return True
        elif not mask.any():
            return True
        return False
    
    def print_debug(self, string):
        if self.is_debug:
            print(string)
    
    def image_batch_size(self, image_3d_4d):
        return image_3d_4d.size()[0]


##################################################################################################################
# BK Bool Not
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKBoolNot:
    def __init__(self):
        self.minimum_image_size = 64
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "boolean": ("BOOLEAN",),
            },
            "optional": {
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("not_boolean",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Has Mask"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, boolean):
        return float("nan")


# NOTE: image tensor coordinates origin is at TOP-LEFT corner
    def process(self, boolean):
        print_debug_header(self.is_debug, "BK BOOLEAN NOT")
        
        not_boolean = not boolean

        print_debug_bar(self.is_debug)
        return (not_boolean,)

    def print_debug(self, string):
        if self.is_debug:
            print(string)

##################################################################################################################
# BK Bool Operation
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKBoolOperation:
    def __init__(self):
        self.minimum_image_size = 64
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "bool_a": ("BOOLEAN",),
                "operation": (["AND", "OR", "XOR"], {"default":"AND"}),
                "bool_b": ("BOOLEAN",),
            },
            "optional": {
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("result_boolean",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Boolean Operation"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, bool_a, operation, bool_b):
        return float("nan")


# NOTE: image tensor coordinates origin is at TOP-LEFT corner
    def process(self, bool_a, operation, bool_b):
        print_debug_header(self.is_debug, "BK BOOLEAN OPERATION")
        
        if operation == "AND":
            result_boolean = bool_a and bool_b
        elif operation == "OR":
            result_boolean = bool_a or bool_b
        elif operation == "XOR":
            result_boolean = bool_a != bool_b
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        print_debug_bar(self.is_debug)
        return (result_boolean,)

    def print_debug(self, string):
        if self.is_debug:
            print(string)

##################################################################################################################
# BK Image Size Test
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKImageSizeTest:
    def __init__(self):
        self.minimum_image_size = 64
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "image": ("IMAGE",),
                "is_image": (["Greater Than", "Less Than", "Equal To", "Greater Than Or Equal To", "Less Than Or Equal To"],),
                "height": ("INT", {"default":"1024", "min": 1}),
                "width": ("INT", {"default":"1024", "min": 1}),
            },
            "optional": {
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("IMAGE", "BOOLEAN",)  # This specifies that the output will be text
    RETURN_NAMES = ("image", "result_boolean",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Image Size Test"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, image, is_image, height, width):
        return float("nan")

    def image_height(self, image):
        return image.size()[1]

    def image_width(self, image):
        return image.size()[2]
    
    def is_image_greater_than(self, image, height, width):
        return self.image_height(image) > height and self.image_width(image) > width
    
    def is_image_less_than(self, image, height, width):
        return self.image_height(image) < height and self.image_width(image) < width
    
    def is_image_equal_to(self, image, height, width):
        return self.image_height(image) == height and self.image_width(image) == width
    
# NOTE: Image tensor coordinates origin is at TOP-LEFT corner
# NOTE: Image tensor is [batch_size, height, width, channels]
# NOTE: Mask tensor is [batch_size, height, width]
    def process(self, image, is_image, height, width):
        print_debug_header(self.is_debug, "BK BOOLEAN OPERATION")


        if is_image == "Greater Than":
            result_boolean = self.is_image_greater_than(image, height, width)
        elif is_image == "Less Than":
            result_boolean = self.is_image_less_than(image, height, width)
        elif is_image == "Equal To":
            result_boolean = self.is_image_equal_to(image, height, width)
        elif is_image == "Greater Than Or Equal To":
            result_boolean = self.is_image_greater_than(image, height, width) or self.is_image_equal_to(image, height, width)
        elif is_image == "Less Than Or Equal To":
            result_boolean = self.is_image_less_than(image, height, width) or self.is_image_equal_to(image, height, width)
        else:
            raise ValueError(f"Unsupported operation: {is_image}")

        print_debug_bar(self.is_debug)
        return (image, result_boolean,)

    def print_debug(self, string):
        if self.is_debug:
            print(string)
    
##################################################################################################################
# BK Add Mask Box
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKAddMaskBox:
    def __init__(self):
        self.minimum_image_size = 64
        self.is_debug = False
        self.ONLY_IF_MISSING_MODES = {
            "If No Mask In Region",
            "If No Mask In Region And Vertical Image",
            "If No Mask In Region And Horizontal Image",
            "If No Mask In Region And Square Image",
        }

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "top":("FLOAT",{"min": 0.0, "max": 100.0}),
                "left":("FLOAT",{"min": 0.0, "max": 100.0}),
                "width":("FLOAT",{"min": 0.0, "max": 100.0}),
                "height":("FLOAT",{"min": 0.0, "max": 100.0}),
                "apply":(["Always", "If Input Mask Empty","If No Mask In Region","If No Mask In Region And Vertical Image","If No Mask In Region And Horizontal Image","If No Mask In Region And Square Image"], {"default":"Always Always"}),
            },
            "optional": {
                "mask":("MASK",),
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("MASK","BOOLEAN", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("mask", "is_applied", "status")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Add Mask Box"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, top, left, height, width, apply, mask=None):
        return float("nan")


# NOTE: image tensor coordinates origin is at TOP-LEFT corner
    def process(self, top, left, height, width, apply, mask=None):

        print_debug_header(self.is_debug, "BK ADD MASK BOX")
        
        # Assuming mask is a 3D tensor with dimensions [batch_size, height, width]
        # If mask is None, create a new one that is the same size as the image
        if mask is None:
            raise ValueError("BKAddMaskBox: Mask must be provided. Must know size of image to create mask for.")
        
        # Get image size (height, width)
        _, img_height, img_width = mask.shape  # Image is assumed to be [batch_size, height, width]
        
        # Convert percentages to actual pixel values
        top_pixel = int(top / 100.0 * img_height)
        left_pixel = int(left / 100.0 * img_width)
        height_pixel = int(height / 100.0 * img_height)
        width_pixel = int(width / 100.0 * img_width)

        # Bound the mask area within the image dimensions
        bottom_pixel = min(top_pixel + height_pixel, img_height)
        right_pixel = min(left_pixel + width_pixel, img_width)

        is_applied = True
        status = "BKAddMaskBox: Box applied to mask"

        if apply == "If No Mask In Region And Vertical Image":
            if self.image_height(mask) < self.image_width(mask):
                return  self.skip(mask, f"BKAddMaskBox: Image height [{self.image_height(mask)}] is less than its width [{self.image_width(mask)}] in mode 'If No Mask In Region And Vertical Image'. It is not a vertical image. Skipping")
            
        if apply == "If No Mask In Region And Horizontal Image":
            if self.image_height(mask) > self.image_width(mask):
                return  self.skip(mask, f"BKAddMaskBox: Image height [{self.image_height(mask)}] is greater than its width [{self.image_width(mask)}] in mode 'If No Mask In Region And Horizontal Image'. It is not a horizontal image. Skipping")

        if apply == "If No Mask In Region And Square Image":
            if self.image_height(mask) != self.image_width(mask):
                return  self.skip(mask, f"BKAddMaskBox: Image height [{self.image_height(mask)}] is not equal its width [{self.image_width(mask)}] in mode 'If No Mask In Region And Square Image'. It is not a square image. Skipping")
        
        if apply == "If Input Mask Empty":
            if not self.is_mask_empty(mask):
                return  self.skip(mask, f"BKAddMaskBox: Input mask is not empty with mode 'If Input Mask Empty'. Skipping")
        
        # Check if we are in "Only If Missing" mode
        if apply in self.ONLY_IF_MISSING_MODES:
            # Check if the region already has a mask applied (non-zero region)
            if mask[0, top_pixel:bottom_pixel, left_pixel:right_pixel].sum() > 0:
                is_applied = False
                status = "BKAddMaskBox: Region already has a mask applied. Skipping."
                print(status)
                return (mask, is_applied, status)  # No mask applied if the region already has a mask in the area


        # Apply the mask in the specified region (set to 1 in the specified area)
        mask[0, top_pixel:bottom_pixel, left_pixel:right_pixel] = 1
        print(status)

        self.print_debug(f"mask.size()[{mask.size()}]")
        print_debug_bar(self.is_debug)
        
        return (mask, is_applied, status)
    
    def skip(self, mask, reason: str):
        print(reason)
        return mask, False, reason

    def print_debug(self, string):
        if self.is_debug:
            print(string)

    def create_empty_opaque_3d_image(self, height, width):
        # NOTE [Batch Size, Height, Width]
        # use the reference image to get batch size and channels
        #create a empty black / transparent image the size of the crop box
        return torch.zeros(( 1, 
                            height, 
                            width), 
                            dtype=torch.float32)

    def image_height(self, image_3d_4d):
        return image_3d_4d.size()[1]

    def image_width(self, image_3d_4d):
        return image_3d_4d.size()[2]
    
    def is_mask_empty(self, mask):
        if mask is None:
            return True
        elif not mask.any():
            return True
        return False



##################################################################################################################
# BK Create Mask For Image
##################################################################################################################

# NOTE: image tensor coordinates origin is at TOP-LEFT corner
class BKCreateMaskForImage:
    def __init__(self):
        self.minimum_image_size = 64
        self.is_debug = False

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "image":("IMAGE",),
                "mask_type": (["Empty - Background Visible", "Opaque - Background Hidden"], {"default":"Empty - Background Visible"}),
            },
            "optional": {
            },
            "hidden": {
            },
        }
    
    RETURN_TYPES = ("MASK",)  # This specifies that the output will be text
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Create Mask For Image"  # Default label text
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, image, mask_type):
        return float("nan")


# NOTE: image tensor coordinates origin is at TOP-LEFT corner
    def process(self, image, mask_type):

        if image is None:
            raise ValueError(f"Image is None. There is no image data to create a mask from.")

        mask = None
        if mask_type == "Empty - Background Visible":
            mask = self.create_empty_opaque_3d_image(self.image_height(image), self.image_width(image))
        else:
            mask = self.create_full_transparent_3d_image(self.image_height(image), self.image_width(image))

        return (mask,)

    def print_debug(self, string):
        if self.is_debug:
            print(string)

    def create_empty_opaque_3d_image(self, height, width):
        # NOTE [Batch Size, Height, Width]
        # use the reference image to get batch size and channels
        #create a empty black / transparent image the size of the crop box
        return torch.zeros(( 1, 
                            height, 
                            width), 
                            dtype=torch.float32)
    
    def create_full_transparent_3d_image(self, height, width):
        # NOTE [Batch Size, Height, Width]
        # use the reference image to get batch size and channels
        #create a empty black / transparent image the size of the crop box
        return torch.ones(( 1, 
                            height, 
                            width), 
                            dtype=torch.float32)

    def image_height(self, image_3d_4d):
        return image_3d_4d.size()[1]

    def image_width(self, image_3d_4d):
        return image_3d_4d.size()[2]

##################################################################################################################
# BK Load Image By Path
##################################################################################################################

    


# Node class definition
class BKLoadImageByPath:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.minimum_image_size = 64

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required":
                    {
                        "image_name": ("STRING", {"default": f'image', "multiline": False}),
                        "image_directory": ("STRING", {"default": f'', "multiline": False, "tooltip":"If a relative path is used, will use the output folder as the base, else will use the absolute path location."}),
                        "extension": ("STRING", {"default": f'.png', "multiline": False}),
                    },
        }

    
    RETURN_TYPES = ("IMAGE", "MASK","STRING", "STRING", "INT", "STRING","STRING", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("image", "mask","positive_prompt", "negative_prompt", "seed", "other","name", "folder_path")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Load Image By Path"  # Default label text
    OUTPUT_NODE = True
    
    def process(self, image_name, image_directory ,extension):
        negative = ""
        positive = ""
        seed = 0
        other = ""

        # Remove any periods prefixed on the extension
        if extension.startswith('.'):
            extension = extension[1:]  # Remove the leading period if it exists

        # Check if the provided path is absolute or relative
        if os.path.isabs(image_directory):
            # If absolute, use the directory and name directly
            image_path = f"{image_directory}\\{image_name.strip()}.{extension}"
        else:
            # If relative, construct the path with the output directory
            image_path = f"{self.output_dir.strip()}\\{image_directory}\\{image_name.strip()}.{extension}"

        # Open the image
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        # Read Prompts and Seed from image
        img_info = img.info

        # Read positive prompt metadata
        if "positive" in img_info:
            positive = str(img_info["positive"])
        
        # Read negative prompt metadata
        if "negative" in img_info:
            negative = str(img_info["negative"])

                # Read negative prompt metadata
        if "other" in img_info:
            other = str(img_info["other"])


        # Read seed meta data
        if "seed" in img_info:
            try:
                seed = int(img_info["seed"])
            except ValueError:
                print("WARNING: 'seed' value is not valid in image.")

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
                mask = torch.zeros((self.minimum_image_size, self.minimum_image_size), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return(output_image, output_mask, positive, negative, seed, other, image_name, image_directory)

# Node class definition
class BKLoadImage:
    def __init__(self):
        print("BKLoadImage - __INIT__")
        self.output_dir = folder_paths.output_directory
        self.minimum_image_size = 64

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

    
    RETURN_TYPES = ("IMAGE","MASK", "STRING", "STRING", "INT", "STRING","STRING", "STRING")  # This specifies that the output will be text
    RETURN_NAMES = ("image","mask", "positive_prompt", "negative_prompt", "seed", "other","name", "folder_path")
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "BK Load Image"  # Default label text
    OUTPUT_NODE = True

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


    
    def process(self, image):
        negative = ""
        positive = ""
        seed = 0
        other = ""

        image_path = folder_paths.get_annotated_filepath(image)
        image_directory = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        # Read Prompts and Seed from image
        img_info = img.info

        # Read positive prompt metadata
        if "positive" in img_info:
            positive = str(img_info["positive"])
        
        # Read negative prompt metadata
        if "negative" in img_info:
            negative = str(img_info["negative"])

        # Read other prompt metadata
        if "other" in img_info:
            other = str(img_info["other"])


        # Read seed meta data
        if "seed" in img_info:
            try:
                seed = int(img_info["seed"])
            except ValueError:
                print("WARNING: 'seed' value is not valid in image.")

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
                mask = torch.zeros((self.minimum_image_size, self.minimum_image_size), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return(output_image, output_mask, positive, negative, seed, other, image_name, image_directory, )
    
##################################################################################################################
# BK LoRA Testing Node
##################################################################################################################

#TODO: make it so that the input is a dropdown list with folderpaths. The folder paths should be the folder paths of the loras loaded, The node will then test between all .safetensors in the specified folder path. I think we can have a dropdown of folder paths, by passing the list to the input. I have seen this done somewhere before.

class BKLoRATestingNode:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "lora_relative_path": ("STRING", {
                "multiline": False,
            }),
            "seed": ("INT", {"default": 0,"tooltip": "Used to determine which LoRA to use next. Set it to 'increment' to cycle through each LoRA. The node will automatically loop when the number is beyond the range of the LoRAs. No need to try to clamp this number."}),
         }}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("MODEL", "CLIP", "lora_name", "lora_path", "seed", "lora_num")
    FUNCTION = "get_lora"
    CATEGORY = "BKNodes/LoRA Testing"  # A category for the node, adjust as needed
    LABEL = "LoRA Testing Node"  # Default label text


    def get_abs_folder(self, path, root_folder):
        if self.is_relative_path(path):
            abs_folder = self.get_absolute_folder_path(root_folder, path)
        else:
            abs_folder = path

        return abs_folder
    
    def filter_strings_by_prefix(string_list, prefix):
        # Use list comprehension to filter strings that start with the given prefix
        return [s for s in string_list if s.startswith(prefix)]

    def get_lora(self, model, clip, lora_relative_path,seed):
        result = (model, clip,"","")

        
        
        # Get list of all loras file paths
        all_loras = folder_paths.get_filename_list("loras")

        # Get all file paths that have the lora's name
        matching_lora_paths = find_matching_files(all_loras, lora_name)

        # Thow error if no LoRAs paths were found
        if matching_lora_paths is None:
            ValueError(f"No matching LoRA's found with name: [{lora_name}]")
        
        if len(matching_lora_paths) <= 0:
            ValueError(f"No matching LoRA's found with name: [{lora_name}]")

        # Use first matching LoRA found 
        lora_path = matching_lora_paths[0]

        # Warn user if mulitple LoRAs found
        if len(matching_lora_paths) > 1:
            print(f"WARNING: Multiple LoRAs Found:")
            for l in matching_lora_paths:
                print(f"- {l}")
            print(f"Using first LoRA found: [{lora_path}]")

        # Load LoRA using path
        lora_items = self.selected_loras.updated_lora_items_with_text(lora_path)

        # If the LoRA was loaded, apply the lora
        if len(lora_items) > 0:
            for item in lora_items:
                result = item.apply_lora(result[0], result[1])
            
        return(result[0], result[1], lora_name, lora_path, seed)    

def pad_num(number, pad):
    return str(number).zfill(pad)

def get_inc_num_betwen(start, end, step, num):
    # Calculate the increment
    increment = num * step
    
    # Roll over if the increment exceeds the range
    range_size = end - start
    if increment >= range_size:
        increment = increment % range_size  # This is the "rollover" effect
    
    # Add the increment to start and return the result
    result = start + increment
    return result

def find_matching_files(file_paths, lora_name):
    # List to hold the matching file paths
    matching_files = []
    
    # Loop through the list of file paths
    for file_path in file_paths:
        # Extract the filename without the extension
        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]

        # Compare the file name (without extension) with lora_name
        if file_name_without_extension == lora_name:
            matching_files.append(file_path)
    
    return matching_files
    
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

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "INT",)
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

    try:
        return int(s.strip())
    except (ValueError, TypeError):
        return -1
    
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

    RETURN_TYPES = ("MODEL", "CLIP","BOOLEAN", "STRING", "STRING", "STRING", "INT", "INT",)
    RETURN_NAMES = ("MODEL", "CLIP","has_next", "lora_name", "lora_prefix", "lora_path", "lora_number", "remaining")
    FUNCTION = "get_lora"
    CATEGORY = "BKNodes/LoRA Testing"  # A category for the node, adjust as needed
    LABEL = "LoRA Testing Node"  # Default label text

    def get_lora(self, clip, model, subfolder, name_prefix, name_suffix, extension, lora_list, while_loop_idx, lora_step, zero_padding):
        lora_result = (model, clip)
        
        total_count = count_lines(lora_list)
        if total_count > 1:
            has_next = has_next_line(while_loop_idx, lora_list)
        else:
            has_next = False
        current_lora_string = get_line_by_index(while_loop_idx, lora_list)
        idx = parse_int(current_lora_string)
        if idx >= 0:
            lora_number = get_nearest_step(lora_step, idx)
            padded_integer_string = f"{lora_number:0{zero_padding}d}"
            lora_path = subfolder + name_prefix + padded_integer_string + name_suffix + extension
            lora_name = name_prefix + padded_integer_string + name_suffix
        
            lora_items = self.selected_loras.updated_lora_items_with_text(lora_path)

            if len(lora_items) > 0:
                for item in lora_items:
                    lora_result = item.apply_lora(model, clip)
        
        else:
            lora_name = f"No Lora"
            lora_path = f"No Lora"
            name_prefix = f"No Lora"
            lora_number = -1
            
        return(lora_result[0], lora_result[1], has_next, lora_name, name_prefix, lora_path, lora_number, total_count) 
    
def to_utf8(input_string):
    """
    Converts a string to UTF-8 encoded bytes.
    
    Parameters:
        input_string (str): The string to encode.
    
    Returns:
        bytes: UTF-8 encoded version of the input string.
    """
    utf8_bytes = input_string.encode('utf-8')
    return utf8_bytes.hex()


class ToUTF8:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
             "text": ("STRING", {
                "multiline": False,
                "default": ""}),

         }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("utf_8",)
    FUNCTION = "convert"
    CATEGORY = "BKNodes"  # A category for the node, adjust as needed
    LABEL = "Convert To UTF8"  # Default label text

    def convert(self, text):
        result = to_utf8(text)
        return(result) 


class GetLargerValue:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "a": ("INT", {"default": 0}),
            "b": ("INT", {"default": 0}),
         }}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("larger_value",)
    FUNCTION = "get_lora"
    CATEGORY = "BKNodes/LoRA Testing"  # A category for the node, adjust as needed
    LABEL = "LoRA Testing Node"  # Default label text

    def get_lora(self, a, b):
        result = None
        absdiff = abs(a-b)
        sum = a + b
        result = (absdiff + sum) / 2
        return(result) 
    
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
    "Ollama Connectivity Data": JSONExtractor,
    "Single LoRA Test Node": SingleLoRATestNode,
    "Multi LoRA Test Node": MultiLoRATestNode,
    "Get Larger Value": GetLargerValue,
    "Convert To UTF8": ToUTF8,
    "BK Path Builder": BKPathBuilder,
    "BK Max Size": BKMaxSize,
    "BK Loop Path Builder": BKLoopPathBuilder,
    "BK Loop Status Text": BKLoopStatusText,
    "BK Add To Path": BKAddToPath,
    "BK Save Image": BKSaveImage,
    "BK Add To JSON": BKAddToJSON,
    "BK Get From JSON": BKReadFromJSON,
    "BK Save Text File": BKSaveTextFile,
    "BK Load Image": BKLoadImage,
    "BK Prompt Sync": BKPromptSync,
    "BK Load Image By Path": BKLoadImageByPath,
    "BK String Splitter": BKStringSplitter,
    "BK Print To Console": BKPrintToConsole,
    "BK Replace All Tags": BKReplaceAllTags,
    "BK Line Counter": BKLineCounter,
    "BK Dynamic Checkpoints": BKDynamicCheckpoints,
    "BK Remove Last Folder": BKRemoveLastFolder,
    "BK Replace Each Tag Random": BKReplaceEachTagRandom,
    "BK Read Text File": BKReadTextFile,
    "BK Multi Read Text File": BKMultiReadTextFile,
    "BK Write Text File": BKWriteTextFile,
    "BK Combo Tag": BKComboTag,
    "BK Dynamic Checkpoints List": BKDynamicCheckpointsList,
    "BK Sampler Options Selector": BKSamplerOptionsSelector,
    "BK TSV Loader": BKTSVFileLoader,
    "BK TSV String Parser": BKTSVStringParser,
    "BK Print To Console With Boarder": BKPrintToConsoleWithBorder,
    "BK Caption File Reader": BKCaptionFileParser,
    "BK Caption FileParser": BKCaptionFileScanner,
    "BK TSV Tag Replacer": BKTSVTagReplacer,
    "BK AI Text Parser": BKAITextParser,
    "BK TSV Header Formatter": BKTSVHeaderFormatter,
    "BK File Selector": BKFileSelector,
    "BK Get Next Img Without Caption": BKGetNextImgWOCaption,
    "BK Remove Any Sentences With Text": BKRemoveAnySentencesWithText,
    "BK File Select Next Missing": BKFileSelectNextMissing,
    "BK Remove Mask At Idx": BKRemoveMaskAtIdx,
    "BK Remove Mask At Idx SAM3": BKRemoveMaskAtIdxSAM3,
    "BK TSV Prompt Reader": BKTSVPromptReader,
    "BK LoRA Testing Node": BKLoRATestingNode,
    "BK Get Next Missing Checkpoint": BKGetNextMissingCheckpoint,
    "BK Is Vertical Image": BKIsVerticalImage,
    "BK Is A Greater Than B INT": BKIsAGreaterThanBINT,
    "BK Is A Greater Than Or Equal To B INT": BKIsAGreaterThanOrEqualToBINT,
    "BK Is A Less Than B INT": BKIsALessThanBINT,
    "BK Is A Less Than Or Equal To B INT": BKIsALessThanOrEqualToBINT,
    "BK File Select Next Unprocessed": BKFileSelectNextUnprocessed,
    "BK Move Or Copy File": BKMoveOrCopyFile,
    "BK Get Matching Mask": BKGetMatchingMask,
    "BK Get Last Folder Name": BKGetLastFolderName,
    "BK Next Unprocessed File In Folder": BKNextUnprocessedFileInFolder,
    "BK Next Unprocessed Image In Folder": BKNextUnprocessedImageInFolder,
    "BK AI Text Cleaner": BKAITextCleaner,
    "BK TSV Random Prompt": BKTSVRandomPrompt,
    "BK Crop And Pad": BKCropAndPad,
    "BK Body Ratios": BKBodyRatios,
    "BK Mask Square And Pad": BKMaskSquareAndPad,
    "BK Mask Test": BKMaskTest,
    "BK Bool Not": BKBoolNot,
    "BK Add Mask Box": BKAddMaskBox,
    "BK Create Mask For Image": BKCreateMaskForImage,
    "BK Bool Operation": BKBoolOperation,
    "BK Image Size Test": BKImageSizeTest,
    "BK Get Next Caption File": BKGetNextCaptionFile,
    "BK Image Sync": BKImageSync,
    "BK Lora Testing Node": BKLoRATestingNode,

}

# TODO: NODES: Create node that will save a json, with the hash of the model, the link to download it, and what resource it is from, then create a node that will use that information to download it? Or maybe instead of a json have it as readable text for anthony?
# TODO: Update the "WORKFLOWS" section to allow for sorting, creating of folders, and moving of files
# TODO: Create crop to mask node
# TODO: Create Remove background based on mask node
# TODO: Create node that will load vae and model and clip for wan and such that uses a config file?


'''
LORA TESTING NOTES
- Should load loras by relative folder path
- Should return name of LoRA so that an image can be saved with the LoRA name
- A folder path should be specified, so it will check if the LoRA has already been tested
- A file should be saved in the folder with the images that records which LoRAs have been tested already
- Not only should the file save the name of the lora, but somehow ID the prompt and identify if the prompt has been used with the LoRA already
-- Maybe hash the prompt text and save that alongside the LoRA name in the file
- A count should be able to be set for how many images for the prompts should be generated
- The node will only use a new prompt when it identifies that a prompt has been generated for all of the loras and for the qty as well
- Need to be able to specify a tag for teh name replacement for the lora


'''