import requests
import json
import os
import time

# Try importing folder_paths safely
try:
    import folder_paths
    comfyui_env_available = True
except ImportError:
    folder_paths = None # Set to None if not available
    comfyui_env_available = False
    print("RH_AudioUploader: ComfyUI folder_paths not found. Cannot determine input file path.")


class RH_AudioUploader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),
                # This input receives the filename assigned by ComfyUI's upload endpoint via JS
                "audio": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "upload_and_get_filename"
    CATEGORY = "RunningHub"
    OUTPUT_NODE = False

    def upload_and_get_filename(self, apiConfig, audio):
        """
        Reads the audio file from ComfyUI's input directory based on the 'audio' filename,
        uploads it to the RunningHub API, and returns the resulting filename/ID.
        """
        # 1. Validate inputs
        if not comfyui_env_available or not folder_paths:
            raise ImportError("folder_paths module is required for RH_AudioUploader to find input files.")

        if not isinstance(apiConfig, dict) or not apiConfig.get("apiKey") or not apiConfig.get("base_url"):
            raise ValueError("Invalid or missing apiConfig structure provided to RH_AudioUploader.")
        
        if not audio or audio.strip() == "":
             raise ValueError("No audio filename provided. Please select and upload an audio file using the node's widget.")

        apiKey = apiConfig["apiKey"]
        baseUrl = apiConfig["base_url"]

        # 2. Get the full path to the uploaded file in ComfyUI's input directory
        try:
            audio_path = folder_paths.get_annotated_filepath(audio)
            if not audio_path or not os.path.exists(audio_path):
                 potential_path = os.path.join(folder_paths.get_input_directory(), audio)
                 if os.path.exists(potential_path):
                     audio_path = potential_path
                 else:
                      potential_path = os.path.join(folder_paths.get_input_directory(), 'uploads', audio)
                      if os.path.exists(potential_path):
                          audio_path = potential_path
                      else:
                         raise FileNotFoundError(f"Audio file not found in input directory: {audio}")

            print(f"RH_AudioUploader: Found audio file at: {audio_path}")

        except Exception as e:
            raise FileNotFoundError(f"Error finding audio file '{audio}': {e}")

        # 3. Prepare for RunningHub API upload
        upload_api_url = f"{baseUrl}/task/openapi/upload" # Using the same endpoint as image/video
        headers = {
            'User-Agent': 'ComfyUI-RH_AudioUploader/1.0' # Updated User-Agent
        }
        data = {
            'apiKey': apiKey,
            'fileType': 'audio' # Changed fileType to 'audio'
        }

        # 4. Read the file and upload with retry logic
        print(f"RH_AudioUploader: Uploading {audio_path} to {upload_api_url}...")
        max_retries = 5
        retry_delay = 1
        last_exception = None
        response = None

        for attempt in range(max_retries):
            try:
                with open(audio_path, 'rb') as f:
                    files = {
                        'file': (os.path.basename(audio_path), f)
                    }
                    response = requests.post(upload_api_url, headers=headers, data=data, files=files)
                    print(f"RH_AudioUploader: Upload attempt {attempt + 1}/{max_retries} - Status Code: {response.status_code}")
                    response.raise_for_status()
                break # Success
            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"RH_AudioUploader: Upload attempt {attempt + 1} failed: {e}")
                if e.response is not None:
                     print(f"RH_AudioUploader: Response content on error: {e.response.text}")
                if attempt < max_retries - 1:
                    print(f"RH_AudioUploader: Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                     print(f"RH_AudioUploader: Max retries ({max_retries}) reached.")
                     raise ConnectionError(f"Failed to upload audio to RunningHub API after {max_retries} attempts. Last error: {last_exception}") from last_exception

        if response is None:
             raise ConnectionError(f"Audio upload failed after {max_retries} attempts, no response received. Last error: {last_exception}")

        # 5. Parse successful response
        try:
            response_json = response.json()
            print(f"RH_AudioUploader: Upload API Response JSON: {response_json}")
        except json.JSONDecodeError as e:
             print(f"RH_AudioUploader: Failed to decode JSON response: {response.text}")
             raise ValueError(f"Failed to decode API response after successful upload: {e}") from e

        if response_json.get('code') != 0:
            raise ValueError(f"RunningHub API reported an error after upload: {response_json.get('msg', 'Unknown API error')}")

        rh_data = response_json.get("data", {})
        uploaded_filename = None
        if isinstance(rh_data, dict):
            uploaded_filename = rh_data.get("fileName")
        elif isinstance(rh_data, str):
             uploaded_filename = rh_data
        
        if not isinstance(uploaded_filename, str) or not uploaded_filename:
            raise ValueError("Upload succeeded but 'fileName' not found in RunningHub API response.data.")

        print(f"RH_AudioUploader: Upload successful. RunningHub filename/ID: {uploaded_filename}")
        return (uploaded_filename,) 