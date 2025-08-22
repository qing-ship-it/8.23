import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import time  # Add this import

class ImageUploaderNode:
    """
    ComfyUI Node: ImageUploaderNode
    Function: Upload input images to server and return the server-returned filename.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),  # API configuration parameters, must contain apiKey and base_url
                "image": ("IMAGE",),  # Input image tensor
            },
        }

    RETURN_TYPES = ("STRING",)  # Output type is string
    RETURN_NAMES = ("filename",)  # Output name is filename
    CATEGORY = "RunningHub"  # Node category
    FUNCTION = "process"  # Specify processing method

    def process(self, image: torch.Tensor, apiConfig: dict) -> tuple:
        """
        Processing method: Upload image to server and return filename.

        Parameters:
            image (torch.Tensor): Input image tensor, shape could be [C, H, W], [H, W, C] or others.
            apiConfig (dict): API configuration parameters, must contain 'apiKey' and 'base_url'.

        Returns:
            tuple: Contains the filename returned after upload.
        """
        # Check input image type
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be a torch.Tensor, but got {type(image)}.")
    
        # Convert image tensor to NumPy array
        image_np = image.detach().cpu().numpy()

        # Print image shape for debugging
        print(f"Original image shape: {image_np.shape}")

        # Process image shape to ensure it's [H, W, C]
        if image_np.ndim == 4:
            # Handle batch dimension, e.g., [B, C, H, W]
            print("Detected 4D tensor. Assuming shape [B, C, H, W]. Taking the first image in the batch.")
            image_np = image_np[0]
            print(f"Image shape after removing batch dimension: {image_np.shape}")

        if image_np.ndim == 3:
            if image_np.shape[0] in [1, 3, 4]:  # [C, H, W]
                image_np = np.transpose(image_np, (1, 2, 0))  # Convert to [H, W, C]
                print(f"Transposed image shape to [H, W, C]: {image_np.shape}")
            elif image_np.shape[2] in [1, 3, 4]:  # [H, W, C]
                # Already in [H, W, C] format, no need to transpose
                print(f"Image already in [H, W, C] format: {image_np.shape}")
            else:
                raise ValueError(f"Unsupported number of channels: {image_np.shape[2]}")
        elif image_np.ndim == 2:
            # Grayscale image [H, W]
            image_np = np.expand_dims(image_np, axis=-1)  # Convert to [H, W, 1]
            print(f"Expanded grayscale image to [H, W, 1]: {image_np.shape}")
        else:
            raise ValueError(f"Unsupported image shape: {image_np.shape}")

        # Determine image mode
        if image_np.shape[2] == 1:
            mode = "L"  # Grayscale image
            image_pil = Image.fromarray((image_np.squeeze(-1) * 255).astype(np.uint8), mode)
            print("Converted to PIL Image with mode 'L'")
        elif image_np.shape[2] == 3:
            mode = "RGB"  # RGB image
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode)
            print("Converted to PIL Image with mode 'RGB'")
        elif image_np.shape[2] == 4:
            mode = "RGBA"  # RGBA image
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode)
            print("Converted to PIL Image with mode 'RGBA'")
        else:
            raise ValueError(f"Unsupported number of channels: {image_np.shape[2]}")

        # Save PIL image to BytesIO buffer
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')  # Can choose 'JPEG' or other formats as needed
        # First get buffer size
        buffer_size = buffer.tell()
        # Then reset pointer to beginning
        buffer.seek(0)
        print("Saved PIL Image to BytesIO buffer.")

        # Print image size in MB
        buffer_size_mb = buffer_size / (1024 * 1024)
        print(f"Image size: {buffer_size_mb:.2f} MB")

        # Check if image size exceeds 10MB
        max_size_bytes = 10 * 1024 * 1024  # 10MB
        if buffer_size > max_size_bytes:
            raise Exception(f"Image size {buffer_size_mb:.2f}MB exceeds the 10MB limit.")

        # Prepare multipart/form-data
        files = {
            'file': ('image.png', buffer, 'image/png')  # Filename and content type
        }
        data = {
            'apiKey': apiConfig.get('apiKey'),
            'fileType': 'image',
        }

        # Get base_url, default to 'https://www.runninghub.cn'
        base_url = apiConfig.get('base_url', 'https://www.runninghub.cn')
        upload_url = f"{base_url}/task/openapi/upload"

        print(f"Uploading image to {upload_url} with apiKey: {data['apiKey']}")

        # Send POST request with retry mechanism
        max_retries = 5
        retry_delay = 1  # Initial delay 1 second
        
        for attempt in range(max_retries):
            try:
                response = requests.post(upload_url, data=data, files=files)
                print(f"Attempt {attempt + 1}: Received response with status code: {response.status_code}")
                
                if response.status_code == 200:
                    break  # Success, break out of retry loop
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception(f"Failed to connect to the server after {max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff, double delay after each failure
                continue

        # If all retries failed
        if response.status_code != 200:
            raise Exception(f"Upload failed with status code {response.status_code} after {max_retries} attempts.")

        # Parse JSON response
        try:
            response_json = response.json()
            print(f"Response JSON: {response_json}")
        except ValueError:
            raise Exception("Failed to parse JSON response from the server.")

        # Check API returned code
        if response_json.get('code') != 0:
            raise Exception(f"Upload failed: {response_json.get('msg')}")

        # Extract filename
        data_field = response_json.get('data', {})
        filename = data_field.get('fileName')

        if not filename:
            raise Exception("Upload succeeded but 'fileName' not found in the response.")

        print(f"Uploaded filename: {filename}")

        return (filename,) 