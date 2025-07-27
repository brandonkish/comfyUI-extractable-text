import os
import json
import comfy.sd  # Import any necessary modules from ComfyUI

# Utility function for handling whitespace
def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

# Node class definition
class ExtractableTextNode:
    RETURN_TYPES = ()  # This specifies that the output will be text
    FUNCTION = "process"  # The function name for processing the inputs
    CATEGORY = "Extractable Nodes"  # A category for the node, adjust as needed
    LABEL = "Extractable Text Node"  # Default label text

    @classmethod
    def INPUT_TYPES(cls):
        # Define the types of inputs your node accepts (single "text" input)
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, autorefresh):
        # Force re-evaluation of the node
            return float("NaN")

    
    def process(self, text):
        # Ensure the input is treated as text
        cleaned_text = handle_whitespace(text)


# Register the node in ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Extractable Text Node": ExtractableTextNode,  # The name that will show in the UI
}
