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
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    OUTPUT_NODE = False  # Not an output node (does not save files or images)
    
    def process(self, text):
        # Ensure the input is treated as text
        cleaned_text = handle_whitespace(text)
        
        # Update the LABEL to display the input text directly on the node (for real-time view)
        self.LABEL = f"Text: {cleaned_text[:30]}..."  # Show the first 30 characters for brevity
        
        # Return the processed text (can be used later in the pipeline if needed)
        return {"output": cleaned_text}

# Register the node in ComfyUI's NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Extractable Text Node": MyCustomNode,  # The name that will show in the UI
}
