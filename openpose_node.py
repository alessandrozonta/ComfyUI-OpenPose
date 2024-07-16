import os
import torch
import cv2
import numpy as np
from .src import torch_openpose, util
from huggingface_hub import snapshot_download
from torchvision import transforms

# Define a transform to convert the image to a tensor
transform = transforms.ToTensor()

# find folder with models
def find_folder_upwards(start_dir, target_folder):
    """
    Search for a target folder starting from a specified directory and moving upwards through the directory tree.

    This function starts from the `start_dir` directory and searches for the `target_folder` in the current directory.
    If it does not find the `target_folder`, it moves up one level in the directory hierarchy and repeats the process.
    This continues until the `target_folder` is found or the root directory is reached.

    Parameters:
    start_dir (str): The directory to start the search from.
    target_folder (str): The name of the folder to search for.

    Returns:
    str: The path to the target folder if found. None if the folder is not found.

    Example:
    >>> find_folder_upwards('/home/user/projects/my_project', 'models')
    '/home/user/models'
    
    >>> find_folder_upwards('/home/user/projects/my_project', 'non_existent_folder')
    None

    Note:
    - This function assumes that `start_dir` is a valid directory path.
    - If `start_dir` is already the root directory, the function will immediately return None if `target_folder` is not found in it.
    """
    current_dir = start_dir

    while True:
        # Check if the target folder exists in the current directory
        if target_folder in os.listdir(current_dir):
            target_path = os.path.join(current_dir, target_folder)
            if os.path.isdir(target_path):
                return target_path

        # Move up one level in the directory tree
        parent_dir = os.path.dirname(current_dir)
        if current_dir == parent_dir:  # If current directory is root
            return None
        
        current_dir = parent_dir

def download_openpose_model_from_huggingface(model_name, output_directory):
    """Downloads an OpenPose model from the specified repository on Hugging Face Hub to the output directory.

    Args:
        model_name (str): The name of the model to download (e.g., "openpose_body_25").
        output_directory (str, optional): The directory to save the downloaded model files. Defaults to "../../models/openpose".

    Returns:
        str: The path to the downloaded model directory or `None` if an error occurs.
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist
    
    # Downloading the model file from the specified URL using snapshot_download
    repo_id = "alezonta/openpose"

    # Check if the file exists
    print("checking existence file")
    file_exists = os.path.isfile(f"{output_directory}/{model_name}")
    if not file_exists:
        print("downloading model")
        # The snapshot_download function is used to download the entire repository
        # or specific files from it. In this case, we specify the repo_id and download the specific file.
        snapshot_download(repo_id=repo_id, allow_patterns=model_name, local_dir=output_directory)
    else:
        print("model alredy downloaded")


  
class OpenPoseNode:
    def __init__(self):
        self.model_path = None  # Initialize to None
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "typology": (["COCO", "BODY_25"],),
                "transparency": ("FLOAT", {"default": 0.4, "min": 0, "max": 1, "step": 0.1}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("image with keypoints", "keypoints only", "keypoints")
    FUNCTION = "main"

    CATEGORY = "OpenPose"
    
    def main(self, input_image, typology, transparency):
        # Check for valid typology
        if typology not in ["COCO", "BODY_25"]:
            raise ValueError(f"Invalid typology: {typology}")
        
        model_name = "body_25.pth" if typology == "BODY_25" else "body_coco.pth"
        
        # find path
        model_path = find_folder_upwards(start_dir=os.getcwd(), target_folder="models")
        
        self.model_path = f"{model_path}/openpose/{model_name}"
        # load model 
        download_openpose_model_from_huggingface(model_name=model_name, output_directory=f"{model_path}/openpose/")
        
        # Check if the input is a batch by looking at its first dimension
        if len(input_image.shape) > 3 or input_image.shape[0] > 1:
            # If it's a batch, take the first image
            image = input_image[0]
        else:
            # If it's a single image, keep it as is
            image = input_image
        
        # remove the batch
        image = image.squeeze(0)
        
        # check if alfa channel is present, if not add it to the original image
        if image.shape[2] != 4:  # Check if image already has 4 channels (RGBA)
            # Create an alpha channel with full opacity
            alpha_channel = torch.ones(image.shape[0], image.shape[1], 1)

            # Concatenate channels to create RGBA image
            image = torch.cat((image, alpha_channel), dim=2)
            
        # Load the selected model
        self.model = torch_openpose.torch_openpose(typology.lower(), self.model_path)  # Replace with actual path


        # Normalize the float32 tensor to the range [0, 1]
        float_tensor_normalized = (image - image.min()) / (image.max() - image.min())
        # Scale the normalized tensor to the range [0, 255] and convert to torch.uint8
        image = (float_tensor_normalized * 255).to(torch.uint8)
        
        max_size = 1024
        # Convert the tensor to a numpy array
        numpy_image = image.cpu().numpy()

        # Convert the numpy array to a cv2 image
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        # Get the dimensions of the image
        height, width = cv2_image.shape[:2]           
        
        # Resize if necessary 
        if max(cv2_image.shape[:2]) > max_size:
            # Determine the scaling factor
            if height > width:
                scaling_factor = 1024.0 / height
            else:
                scaling_factor = 1024.0 / width
            # Resize the image
            new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
            resized_image = cv2.resize(cv2_image, new_dimensions, interpolation=cv2.INTER_AREA)
        else:
            resized_image = cv2_image

        # Get keypoints using the loaded model
        poses = self.model(resized_image)

        # Draw keypoints
        drawn_image = util.draw_bodypose(resized_image, poses, typology.lower(), transparency=transparency)
        drawn_image = drawn_image.astype(np.float32) / 255
        
        # only keypoints image
        black_image = np.zeros_like(resized_image)
        only_keypoints = util.draw_bodypose(black_image, poses, typology.lower())
        only_keypoints = only_keypoints.astype(np.float32) / 255
 
        # Resize back if necessary
        if max(image.shape[:2]) > max_size:
            drawn_image = cv2.resize(drawn_image, (cv2_image.shape[1], cv2_image.shape[0]), interpolation=cv2.INTER_AREA)
            only_keypoints = cv2.resize(only_keypoints, (cv2_image.shape[1], cv2_image.shape[0]), interpolation=cv2.INTER_AREA)

        # Apply the transform to the image
        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR) 
        only_keypoints = cv2.cvtColor(only_keypoints, cv2.COLOR_RGB2BGR) 
        drawn_image = np.transpose(drawn_image, (1, 2, 0))
        only_keypoints = np.transpose(only_keypoints, (1, 2, 0))
        image_tensor = transform(drawn_image).unsqueeze(0)
        only_keypoints = transform(only_keypoints).unsqueeze(0)

        # Collect poses in the specified format
        pose_data = {
            'people': [{'pose_keypoints_2d': poses}],
            'canvas_height': image_tensor.shape[1],
            'canvas_width': image_tensor.shape[2]
        }
        
        # Convert back to torch image and return
        return (image_tensor, only_keypoints, pose_data)


NODE_CLASS_MAPPINGS = {
    "OpenPose - Get poses": OpenPoseNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseNode": "OpenPose - Get poses"
}
