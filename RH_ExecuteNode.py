import requests
import time
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
import torch
import os
import websocket  # Requires websocket-client package
import threading
import comfy.utils # Import comfy utils for ProgressBar
import cv2 # <<< Added import for OpenCV
import safetensors.torch # <<< Added safetensors import
import torchaudio 
import torch.nn.functional as F # <<< Add F for padding

# Try importing ComfyUI video classes safely
try:
    # Use the correct official ComfyUI import path
    from comfy_api.input_impl import VideoFromFile
    video_support_available = True
    print("ComfyUI video support loaded successfully.")
except ImportError:
    try:
        # Alternative import paths for different ComfyUI versions
        from comfy_execution.graph_utils import VideoFromFile
        video_support_available = True
        print("ComfyUI video support loaded via alternative path.")
    except ImportError:
        try:
            from comfy.graph_utils import VideoFromFile
            video_support_available = True
            print("ComfyUI video support loaded via legacy path.")
        except ImportError:
            # Fallback if video support not available
            video_support_available = False
            print("Warning: ComfyUI video support not available. VIDEO output will return None.")

# Try importing folder_paths safely
try:
    import folder_paths
    comfyui_env_available = True # Use a more generic name
except ImportError:
    comfyui_env_available = False
    print("ComfyUI folder_paths not found. Some features like specific output paths might use fallbacks.")


class ExecuteNode:
    ESTIMATED_TOTAL_NODES = 10 # Default estimate

    def __init__(self):
        self.ws = None
        self.task_completed = False
        self.ws_error = None
        self.executed_nodes = set()
        self.prompt_tips = "{}"
        self.pbar = None
        self.node_lock = threading.Lock()
        self.total_nodes = None
        self.current_steps = 0 # Track current steps for logging

    def update_progress(self):
        """Increments the progress bar by one step and logs, stopping at total_nodes."""
        with self.node_lock:
            # Guard 1: Check completion status first
            if self.task_completed:
                # Optional: Log if needed, but return silently to avoid spam
                # print(f"Skipping progress update because task is already completed.")
                return

            # Guard 2: Check if progress bar exists AND if we are already at or beyond the total
            if not self.pbar or self.current_steps >= self.total_nodes:
                 # Optional: Log if trying to update when already >= total for debugging
                 # if self.pbar and self.current_steps >= self.total_nodes:
                 #     print(f"Debug: update_progress called when steps ({self.current_steps}) >= total ({self.total_nodes}). Skipping update.")
                 return

            # --- If guards passed, proceed with increment and update --- 
            self.current_steps += 1
            # Increment the ComfyUI progress bar by 1
            self.pbar.update(1)
            # Log the current state
            # Use min for logging safety, although current_steps should now never exceed total_nodes here
            display_steps = min(self.current_steps, self.total_nodes) 
            print(f"Progress Update: Step {display_steps}/{self.total_nodes} ({(display_steps/self.total_nodes)*100:.1f}%)")


    def complete_progress(self):
        """Sets the progress bar to 100% and marks task as completed."""
        # --- Use lock for thread safety ---\n        with self.node_lock:
            # Check if already completed to prevent redundant calls/logs
            if self.task_completed:
                return

            print(f"Finalizing progress: Setting task_completed = True")
            # --- Set completion flag FIRST ---\n            self.task_completed = True

            # --- Update progress bar to final state --- 
            if self.pbar:
                # Ensure the bar visually reaches 100% regardless of intermediate steps received
                print(f"Forcing progress bar to 100% ({self.total_nodes}/{self.total_nodes}). Current steps internally were {self.current_steps}.")
                # Use update_absolute to set the final value and total explicitly.
                # This handles cases where it finished early or exactly on time.
                self.pbar.update_absolute(self.total_nodes, self.total_nodes)
                # Also update internal counter for consistency, although it might be redundant now
                self.current_steps = self.total_nodes
            else:
                 print("Progress bar not available during finalization.")


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),
            },
            "optional": {
                "nodeInfoList": ("ARRAY", {"default": []}),
                "run_timeout": ("INT", {"default": 600, "min": 1, "max": 9999999}), # Corrected comma and added closing brace
                "concurrency_limit": ("INT", {"default": 1, "min": 1, "max": 100}), # Restored min/max
                "is_webapp_task": ("BOOLEAN", {"default": False}),
                "use_rtx4090_48g": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT", "STRING", "AUDIO", "VIDEO")
    RETURN_NAMES = ("images", "video_frames", "latent", "text", "audio", "video")

    CATEGORY = "RunningHub"
    FUNCTION = "process"
    OUTPUT_NODE = True # Indicate support for progress display

    # --- WebSocket Handlers ---
    def on_ws_message(self, ws, message):
        """Handle WebSocket messages and update internal state and progress bar"""
        try:
            # Check completion status AT THE START
            with self.node_lock:
                is_completed = self.task_completed
            if is_completed:
                 # print("WS Message received after task completion, ignoring.") # Optional: reduce log spam
                 return

            # --- Safely handle message decoding and JSON parsing ---
            print(f"--- Raw WS Message Received ---")
            
            # Handle different message types (string, bytes, etc.)
            processed_message = None
            if isinstance(message, bytes):
                # Try different encodings for bytes
                for encoding in ['utf-8', 'utf-16', 'latin-1']:
                    try:
                        processed_message = message.decode(encoding)
                        print(f"Successfully decoded bytes message using {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if processed_message is None:
                    print(f"Warning: Could not decode bytes message with any common encoding. Raw bytes length: {len(message)}")
                    print(f"First 50 bytes (hex): {message[:50].hex() if len(message) >= 50 else message.hex()}")
                    return # Skip this message
            elif isinstance(message, str):
                processed_message = message
            else:
                print(f"Warning: Received unknown message type: {type(message)}")
                return

            # Try to parse as JSON
            data = None
            try:
                data = json.loads(processed_message)
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse message as JSON: {e}")
                print(f"Raw message content (first 200 chars): {processed_message[:200]}")
                print(f"Message length: {len(processed_message)}")
                
                # Try to extract any JSON-like content if it's mixed with other data
                try:
                    # Look for JSON-like patterns in the message
                    import re
                    json_match = re.search(r'\{.*\}', processed_message, re.DOTALL)
                    if json_match:
                        potential_json = json_match.group(0)
                        data = json.loads(potential_json)
                        print("Successfully extracted JSON from mixed content")
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                    else:
                        print("No JSON pattern found in message, skipping...")
                        return
                except Exception as extract_e:
                    print(f"Failed to extract JSON from message: {extract_e}")
                    return
                    
            print(f"-----------------------------")
            # --- End safe message processing ---

            if data is None:
                print("No valid data extracted from WebSocket message")
                return
            message_type = data.get("type")
            node_data = data.get("data", {})
            node_id = node_data.get("node")

            # Handle node execution updates (both 'executing' and 'execution_success')
            # Based on user feedback, 'execution_success' might signal single node completion.
            if message_type == "executing" or message_type == "execution_success":
                if node_id is not None:
                    # Check if it's a new node before calling update
                    # Use lock to safely check and add to executed_nodes
                    with self.node_lock:
                         is_new_node = node_id not in self.executed_nodes
                         if is_new_node:
                             self.executed_nodes.add(node_id)
                    
                    if is_new_node:
                         self.update_progress() # This method is guarded internally
                         print(f"WS ({message_type}): Node {node_id} reported.")
                    else:
                         print(f"WS ({message_type}): Node {node_id} reported again (ignored for progress).")
                elif message_type == "executing" and node_id is None: # Null node signal
                    print("WS (executing): Received null node signal, potentially end of execution phase.")
                elif message_type == "execution_success" and node_id is None:
                    # If execution_success doesn't have a node_id, what does it mean?
                    # Log it for now, DO NOT call complete_progress.
                    print(f"WS (execution_success): Received signal without node_id. Data: {node_data}")
                    # self.complete_progress() # <<< REMOVED - This was incorrect based on user feedback

            # Handle other message types if necessary (e.g., specific overall error messages)
            # elif message_type == "execution_error": # Hypothetical example
            #     error_details = node_data.get("error", "Unknown WS error")
            #     print(f"WS: Received execution error: {error_details}")
            #     with self.node_lock:
            #         if not self.task_completed:
            #             if self.ws_error is None:
            #                 self.ws_error = Exception(f"WS Error: {error_details}")
            #             self.task_completed = True
            
            else:
                 print(f"WS: Received unhandled message type '{message_type}': {data}")

        except UnicodeDecodeError as e:
            print(f"Error: WebSocket message encoding issue: {e}")
            print("This is likely a non-critical WebSocket protocol issue. Continuing task...")
            # Don't set error state for encoding issues - these are usually non-critical
            # and the task can continue via HTTP polling
            
        except json.JSONDecodeError as e:
            print(f"Error: WebSocket message JSON parsing issue: {e}")
            print("This is likely a non-critical WebSocket protocol issue. Continuing task...")
            # Don't set error state for JSON parsing issues - these are usually non-critical
            
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")
            print(f"Exception type: {type(e).__name__}")
            
            # Only set error state for critical exceptions
            if isinstance(e, (ConnectionError, OSError, IOError)):
                print("Critical WebSocket error detected, marking as error state")
                with self.node_lock:
                    if not self.task_completed:
                        if self.ws_error is None:
                            self.ws_error = e
                        # Don't necessarily mark completed here, let polling confirm final state
                        # self.task_completed = True
            else:
                print("Non-critical WebSocket error, continuing task via HTTP polling...") 

    def on_ws_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
        self.ws_error = error
        # Mark task as complete via the centralized method
        self.complete_progress()

    def on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        # If closed unexpectedly, mark as complete to end loop
        # Use lock temporarily just to read task_completed safely
        with self.node_lock:
             should_complete = not self.task_completed
        if should_complete:
             print("Warning: WebSocket closed unexpectedly. Forcing task completion.")
             self.ws_error = self.ws_error or IOError(f"WebSocket closed unexpectedly ({close_status_code})")
             # Mark task as complete via the centralized method
             self.complete_progress()

    def on_ws_open(self, ws):
        """Handle WebSocket connection open"""
        print("WebSocket connection established")
        # Note: executed_nodes should be cleared at the start of 'process'

    def connect_websocket(self, wss_url):
        """Establish WebSocket connection"""
        print(f"Connecting to WebSocket: {wss_url}")
        websocket.enableTrace(False) # Keep this false unless debugging WS protocol
        self.ws = websocket.WebSocketApp(
            wss_url,
            on_message=self.on_ws_message,
            on_error=self.on_ws_error,
            on_close=self.on_ws_close,
            on_open=self.on_ws_open
        )
        ws_thread = threading.Thread(target=self.ws.run_forever, name="RH_ExecuteNode_WSThread")
        ws_thread.daemon = True
        ws_thread.start()
        print("WebSocket thread started.")

    def check_and_complete_task(self):
        """If task times out after null node, force completion."""
        # complete_progress now checks the flag internally and uses lock
        print("Task completion timeout after null node signal - attempting forced completion.")
        self.complete_progress()

    def get_workflow_node_count(self, api_key, base_url, workflow_id):
        """Get the total number of nodes from workflow JSON."""
        url = f"{base_url}/api/openapi/getJsonApiFormat"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
        }
        data = {
            "apiKey": api_key,
            "workflowId": workflow_id
        }

        max_retries = 5
        retry_delay = 1
        last_exception = None
        node_count = None

        for attempt in range(max_retries):
            response = None
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to get workflow node count...")
                response = requests.post(url, json=data, headers=headers, timeout=30)
                response.raise_for_status()

                result = response.json()

                if result.get("code") != 0:
                    api_msg = result.get('msg', 'Unknown API error')
                    print(f"API error on attempt {attempt + 1}: {api_msg}")
                    raise Exception(f"API error getting workflow node count: {api_msg}")

                workflow_json = result.get("data", {}).get("prompt")
                if not workflow_json:
                    raise Exception("No workflow data found in response")

                # Parse the workflow JSON
                workflow_data = json.loads(workflow_json)
                
                # Count the number of nodes
                node_count = len(workflow_data)
                print(f"Workflow contains {node_count} nodes")
                return node_count

            except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError, Exception) as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                last_exception = e
                if isinstance(e, json.JSONDecodeError) and response is not None:
                     print(f"Raw response text on JSON decode error: {response.text}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Max retries reached for getting workflow node count.")
                    raise e

        # This should theoretically never be reached due to the raise in the loop
        raise Exception("Failed to get workflow node count after multiple attempts.")

    def create_progress_bar(self, total_nodes):
        """Create a ComfyUI progress bar"""
        try:
            # Use the safe way to get progress bar
            from comfy.execution import ProgressBar
            self.pbar = ProgressBar(total_nodes)
            print(f"Progress bar created with total nodes: {total_nodes}")
            return True
        except ImportError as e:
            print(f"Warning: Could not import ComfyUI ProgressBar: {e}")
            try:
                # Fallback to manual progress tracking
                self.pbar = type('ProgressBarFallback', (), {
                    'update': lambda self, steps: print(f"Progress: {self.current_steps + steps}/{self.total_nodes}"),
                    'update_absolute': lambda self, current, total: print(f"Progress: {current}/{total}")
                })()
                print("Fallback progress bar created.")
                return True
            except Exception as fallback_e:
                print(f"Error creating fallback progress bar: {fallback_e}")
                self.pbar = None
                return False

    def process(self, apiConfig, nodeInfoList=[], run_timeout=600, concurrency_limit=1, is_webapp_task=False, use_rtx4090_48g=False):
        """Main processing function to execute the workflow via API"""
        # --- Reset state from previous runs --- 
        with self.node_lock:
            self.task_completed = False
            self.executed_nodes = set()
            self.ws_error = None
            self.current_steps = 0

        # --- Extract API config --- 
        api_key = apiConfig.get("apiKey")
        base_url = apiConfig.get("baseUrl", "https://api.runninghub.cn")
        workflow_id = apiConfig.get("workflowId")

        if not api_key or not workflow_id:
            raise ValueError("API Key and Workflow ID are required")

        print(f"Using API Key: {api_key[:5]}...{api_key[-5:]}")
        print(f"Workflow ID: {workflow_id}")
        print(f"Base URL: {base_url}")

        # --- Get workflow node count for progress bar --- 
        try:
            self.total_nodes = self.get_workflow_node_count(api_key, base_url, workflow_id)
        except Exception as e:
            print(f"Warning: Failed to get workflow node count: {e}")
            # Fallback to default estimated nodes
            self.total_nodes = self.ESTIMATED_TOTAL_NODES
            print(f"Using default estimated node count: {self.total_nodes}")

        # --- Create progress bar --- 
        self.create_progress_bar(self.total_nodes)

        # --- Prepare payload --- 
        payload = {
            "apiKey": api_key,
            "workflowId": workflow_id,
            "nodeInfoList": nodeInfoList,
            "isWebappTask": is_webapp_task,
            "useRTX409048g": use_rtx4090_48g,
            "concurrencyLimit": concurrency_limit
        }

        # --- Execute API call --- 
        print("Submitting workflow execution request...")
        response = None
        try:
            response = requests.post(
                f"{base_url}/api/openapi/executeWorkflow",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error submitting workflow execution: {e}")
            if response is not None:
                print(f"Response status code: {response.status_code}")
                print(f"Response content: {response.text}")
            raise

        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if result.get("code") != 0:
            api_msg = result.get('msg', 'Unknown API error')
            print(f"API error: {api_msg}")
            raise Exception(f"API error executing workflow: {api_msg}")

        task_id = result.get("data", {}).get("taskId")
        if not task_id:
            raise Exception("No task ID returned from API")

        print(f"Workflow execution started. Task ID: {task_id}")

        # --- Set up WebSocket for real-time updates --- 
        wss_url = f"wss://api.runninghub.cn/ws/task/{task_id}?apiKey={api_key}"
        self.connect_websocket(wss_url)

        # --- Poll for task completion --- 
        start_time = time.time()
        max_timeout = run_timeout  # seconds
        interval = 5  # seconds

        while True:
            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                print(f"Task execution timed out after {elapsed:.1f} seconds")
                # Mark as complete to stop WebSocket processing
                self.complete_progress()
                raise TimeoutError(f"Task execution timed out after {max_timeout} seconds")

            # Check if task completed via WebSocket
            with self.node_lock:
                if self.task_completed:
                    break
                if self.ws_error:
                    print(f"WebSocket error detected: {self.ws_error}")
                    # Continue polling despite WebSocket error
                    
            # Check task status via HTTP
            try:
                status_response = requests.get(
                    f"{base_url}/api/openapi/getTaskInfo",
                    params={"apiKey": api_key, "taskId": task_id},
                    timeout=30
                )
                status_response.raise_for_status()
                status_result = status_response.json()

                if status_result.get("code") != 0:
                    api_msg = status_result.get('msg', 'Unknown API error')
                    print(f"API error getting task status: {api_msg}")
                    # Continue polling despite API error
                    time.sleep(interval)
                    continue

                task_status = status_result.get("data", {}).get("status")
                print(f"Task status: {task_status}")

                if task_status in ["SUCCESS", "FAILED", "CANCELLED"]:
                    # Mark as complete to stop WebSocket processing
                    self.complete_progress()
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error checking task status: {e}")
                # Continue polling despite error
                
            time.sleep(interval)

        # --- Clean up WebSocket --- 
        if self.ws:
            self.ws.close()
            print("WebSocket connection closed")

        # --- Check for errors --- 
        with self.node_lock:
            if self.ws_error:
                print(f"WebSocket error occurred during execution: {self.ws_error}")
                # We still proceed to get results, as the task might have completed despite the WS error
                
        # --- Retrieve results --- 
        print("Retrieving task results...")
        try:
            result_response = requests.get(
                f"{base_url}/api/openapi/getTaskResult",
                params={"apiKey": api_key, "taskId": task_id},
                timeout=60
            )
            result_response.raise_for_status()
            result_data = result_response.json()

            if result_data.get("code") != 0:
                api_msg = result_data.get('msg', 'Unknown API error')
                print(f"API error getting task results: {api_msg}")
                raise Exception(f"API error retrieving task results: {api_msg}")

            results = result_data.get("data", {})
            print(f"Retrieved {len(results)} result(s)")

        except requests.exceptions.RequestException as e:
            print(f"Error retrieving task results: {e}")
            if result_response is not None:
                print(f"Response status code: {result_response.status_code}")
                print(f"Response content: {result_response.text}")
            raise

        # --- Process results --- 
        images = []
        video_frames = []
        latent = None
        text = []
        audio = []
        video = None

        for result in results:
            result_type = result.get("type")
            result_value = result.get("value")
            result_name = result.get("name", "Unnamed Result")

            # Handle different result types
            if result_type == "IMAGE" and result_value:
                print(f"Processing image result: {result_name}")
                try:
                    # Convert base64 to image
                    image_data = result_value.split(',')[1]  # Remove data URI prefix if present
                    image_bytes = BytesIO(base64.b64decode(image_data))
                    image = Image.open(image_bytes).convert("RGB")
                    # Convert to ComfyUI tensor format (HWC -> CHW, normalized to 0-1)
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
                    images.append(image_tensor)
                except Exception as e:
                    print(f"Error processing image result {result_name}: {e}")

            elif result_type == "VIDEO" and result_value:
                print(f"Processing video result: {result_name}")
                if video_support_available:
                    try:
                        # Get video URL
                        video_url = result_value
                        print(f"Video URL: {video_url}")

                        # Download video to a temporary file
                        temp_video_path = os.path.join(tempfile.gettempdir(), f"rh_video_{task_id}.mp4")
                        print(f"Downloading video to: {temp_video_path}")
                        response = requests.get(video_url, stream=True, timeout=60)
                        with open(temp_video_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Video downloaded to {temp_video_path}")

                        # Create VideoFromFile object
                        video = VideoFromFile()
                        video.load_video(temp_video_path)
                        print(f"Video loaded into ComfyUI video object")

                        # Extract frames for video_frames output
                        cap = cv2.VideoCapture(temp_video_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        print(f"Extracting {frame_count} frames from video")

                        for _ in range(frame_count):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            # Convert BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Convert to ComfyUI tensor format
                            frame_np = np.array(frame).astype(np.float32) / 255.0
                            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)
                            video_frames.append(frame_tensor)

                        cap.release()
                        print(f"Extracted {len(video_frames)} frames")

                    except Exception as e:
                        print(f"Error processing video result {result_name}: {e}")
                        video = None
                else:
                    print("Video support not available. Skipping video processing.")

            elif result_type == "TEXT" and result_value:
                print(f"Processing text result: {result_name}")
                text.append(f"{result_name}: {result_value}")

            elif result_type == "AUDIO" and result_value:
                print(f"Processing audio result: {result_name}")
                try:
                    # Get audio URL
                    audio_url = result_value
                    print(f"Audio URL: {audio_url}")

                    # Download audio
                    response = requests.get(audio_url, stream=True, timeout=60)
                    audio_bytes = BytesIO(response.content)

                    # Load audio with torchaudio
                    waveform, sample_rate = torchaudio.load(audio_bytes)
                    audio.append({
                        "waveform": waveform,
                        "sample_rate": sample_rate,
                        "name": result_name
                    })
                    print(f"Loaded audio: {result_name}, sample rate: {sample_rate}Hz")
                except Exception as e:
                    print(f"Error processing audio result {result_name}: {e}")

            elif result_type == "LATENT" and result_value:
                print(f"Processing latent result: {result_name}")
                try:
                    # Handle latent data based on format
                    if isinstance(result_value, dict) and "sample" in result_value:
                        # Assume it's already in a compatible format
                        latent = result_value
                    elif isinstance(result_value, str):
                        # Try to parse as base64 encoded safetensors
                        try:
                            # Decode base64
                            latent_data = base64.b64decode(result_value)
                            # Load safetensors
                            latent = safetensors.torch.load(latent_data)
                        except Exception as e:
                            print(f"Error parsing latent as safetensors: {e}")
                            # Try to parse as JSON
                            try:
                                latent = json.loads(result_value)
                            except json.JSONDecodeError as je:
                                print(f"Error parsing latent as JSON: {je}")
                                latent = None
                    else:
                        print(f"Unknown latent format: {type(result_value)}")
                        latent = None

                    if latent is not None:
                        print("Successfully processed latent result")
                    else:
                        print("Failed to process latent result")
                except Exception as e:
                    print(f"Error processing latent result {result_name}: {e}")

            else:
                print(f"Unhandled result type: {result_type} for {result_name}")

        # --- Final cleanup and return --- 
        # Combine image tensors if there are multiple
        combined_images = torch.cat(images) if images else None
        combined_video_frames = torch.cat(video_frames) if video_frames else None
        combined_text = "\n".join(text) if text else ""

        # Create a metadata dictionary to return additional info
        metadata = {
            "task_id": task_id,
            "execution_time": time.time() - start_time,
            "result_count": len(results)
        }
        print(f"Task {task_id} completed in {metadata['execution_time']:.1f} seconds")

        return (combined_images, combined_video_frames, latent, combined_text, audio, video)


# Create a class wrapper for the node (required for ComfyUI)
class RH_ExecuteNode(ExecuteNode):
    pass

# Add import for base64 which was missing
import base64
import tempfile # Add missing import for tempfile

# Add any additional imports that were missing
# Ensure that all required imports are included
try:
    # For text processing
    import re
except ImportError:
    print("Warning: re module not found. Some functionality may be limited.")

try:
    # For image processing
    import numpy as np
except ImportError:
    print("Warning: numpy module not found. Image processing will be limited.")

try:
    # For tensor operations
    import torch
except ImportError:
    print("Warning: torch module not found. Some functionality may be limited.")

# The following is required to make the node available in ComfyUI
NODE_CLASS_MAPPINGS = {"RH_ExecuteNode": RH_ExecuteNode}
NODE_DISPLAY_NAME_MAPPINGS = {"RH_ExecuteNode": "RH Execute Node"}

# Validate required packages
required_packages = [
    ("requests", "requests"),
    ("websocket-client", "websocket"),
    ("Pillow", "PIL"),
]

missing_packages = []
for package_name, import_name in required_packages:
    try:
        __import__(import_name)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    print(f"Warning: Missing required packages: {', '.join(missing_packages)}")
    print("Please install them using pip install {}".format(' '.join(missing_packages)))

# Add logging for successful loading
print("RH Execute Node loaded successfully!")
