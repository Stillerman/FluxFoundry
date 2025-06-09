## Gradio MCP server that launches modal finetune

import gradio as gr
import requests
import json
import time
import subprocess
import os
from typing import Optional, Dict, Any, Tuple

# Configuration - Update these URLs to match your deployed Modal app
# MODAL_BASE_URL = "https://stillerman--jason-lora-flux"  # Update with your actual Modal app URL
# START_TRAINING_URL = f"{MODAL_BASE_URL}-api-start-training.modal.run"
# JOB_STATUS_URL = f"{MODAL_BASE_URL}-api-job-status.modal.run"

def start_training(
    dataset_id: str,
    hf_token: str,
    output_repo: str,
    start_training_url: str,
    instance_name: Optional[str] = None,
    class_name: Optional[str] = None,
    max_train_steps: int = 500
) -> tuple[str, str]:
    """
    Start a LoRA training job for Flux image generation model.

    This function initiates a LoRA (Low-Rank Adaptation) training job on a dataset of images.
    It sends a request to a Modal API endpoint to start the training process.

    Parameters:
    - dataset_id (str, required): The HuggingFace dataset ID containing training 5-10 images, format: "username/dataset-name"
    - hf_token (str, required): HuggingFace access token with read permissions, format: "hf_xxxxxxxxxxxx"
    - output_repo (str, required): HuggingFace repository where trained LoRA will be uploaded, format: "username/repo-name"
    - start_training_url (str, required): Modal API endpoint for starting training, format: "https://modal-app-url-api-start-training.modal.run". If the app is already deployed, this can be found in the Modal [dashboard](https://modal.com/apps/) . Otherwise, the app can get deployed with the deploy_for_user function.
    - instance_name (str, optional): Name of the subject being trained (e.g., 'Fluffy', 'MyDog', 'John')
    - class_name (str, optional): Class category of the subject (e.g., 'person', 'dog', 'cat', 'building')
    - max_train_steps (int, optional): Number of training steps, range 100-2000, default 500

    Returns:
    - tuple[str, str]: (status_message, job_id)
      - status_message: Human-readable status with training details or error message
      - job_id: Unique identifier for the training job, empty string if failed

    Example usage:
    status, job_id = start_training(
        dataset_id="myuser/dog-photos",
        hf_token="hf_abcdef123456",
        output_repo="myuser/my-dog-lora",
        instance_name="Fluffy",
        class_name="dog",
        max_train_steps=500
    )
    """

    if not dataset_id or not hf_token or not output_repo or not start_training_url:
        return "❌ Error: Dataset ID, HuggingFace token, output repo, and start training URL are required", ""

    payload = {
        "dataset_id": dataset_id,
        "hf_token": hf_token,
        "output_repo": output_repo,
        "max_train_steps": max_train_steps
    }

    # Add optional parameters if provided
    if instance_name and instance_name.strip():
        payload["instance_name"] = instance_name.strip()
    if class_name and class_name.strip():
        payload["class_name"] = class_name.strip()

    try:
        response = requests.post(
            start_training_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "started":
                job_id = result.get("job_id", "")
                message = f"✅ Training started successfully!\n\n"
                message += f"**Job ID:** `{job_id}`\n"
                message += f"**Dataset:** {dataset_id}\n"
                message += f"**Output Repo:** {output_repo}\n"
                message += f"**Training Steps:** {max_train_steps}\n\n"
                message += "Copy the Job ID to check status below."
                return message, job_id
            else:
                return f"❌ Error: {result.get('message', 'Unknown error')}", ""
        else:
            return f"❌ HTTP Error {response.status_code}: {response.text}", ""

    except requests.exceptions.Timeout:
        return "❌ Error: Request timed out. The service might be starting up.", ""
    except requests.exceptions.RequestException as e:
        return f"❌ Error: Failed to connect to training service: {str(e)}", ""
    except json.JSONDecodeError:
        return "❌ Error: Invalid response from server", ""

def check_job_status(job_id: str, job_status_url: str) -> str:
    """
    Check the current status of a LoRA training job.

    This function queries the Modal API to get the current status of a training job
    using its unique job ID. It returns detailed information about the job progress.

    Parameters:
    - job_id (str, required): The unique job identifier returned from start_training function
    - job_status_url (str, required): Modal API endpoint for checking job status, format: "https://modal-app-url-api-job-status.modal.run". If the app is already deployed, this can be found in the Modal [dashboard](https://modal.com/apps/) . Otherwise, the app can get deployed with the deploy_for_user function.

    Returns:
    - str: Detailed status message containing:
      - Job status (completed, running, failed, error)
      - Training results if completed (dataset used, steps completed, training prompt)
      - Error messages if failed
      - Progress information if still running

    Possible status values:
    - "completed": Training finished successfully, LoRA model is ready
    - "running": Training is still in progress
    - "failed": Training failed due to an error
    - "error": System error occurred

    Example usage:
    status_info = check_job_status("job_12345abcdef", "https://modal-app-url-api-job-status.modal.run")
    """

    if not job_id or not job_id.strip():
        return "❌ Error: Job ID is required"

    try:
        response = requests.get(
            job_status_url,
            params={"job_id": job_id.strip()},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")

            if status == "completed":
                message = "🎉 **Training Completed!**\n\n"
                training_result = result.get("result", {})
                if isinstance(training_result, dict):
                    message += f"**Status:** {training_result.get('status', 'completed')}\n"
                    message += f"**Message:** {training_result.get('message', 'Training finished')}\n"
                    if training_result.get('dataset_used'):
                        message += f"**Dataset Used:** {training_result['dataset_used']}\n"
                    if training_result.get('training_steps'):
                        message += f"**Training Steps:** {training_result['training_steps']}\n"
                    if training_result.get('training_prompt'):
                        message += f"**Training Prompt:** {training_result['training_prompt']}\n"
                else:
                    message += f"**Result:** {training_result}"
                return message

            elif status == "running":
                return f"🔄 **Training in Progress**\n\nThe training job is still running. Check back in a few minutes."

            elif status == "failed":
                error_msg = result.get("message", "Training failed with unknown error")
                return f"❌ **Training Failed**\n\n**Error:** {error_msg}"

            elif status == "error":
                error_msg = result.get("message", "Unknown error occurred")
                return f"❌ **Error**\n\n**Message:** {error_msg}"

            else:
                return f"❓ **Unknown Status**\n\n**Status:** {status}\n**Response:** {json.dumps(result, indent=2)}"

        else:
            return f"❌ HTTP Error {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return "❌ Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"❌ Error: Failed to connect to status service: {str(e)}"
    except json.JSONDecodeError:
        return "❌ Error: Invalid response from server"

def deploy_for_user(token_id: str, token_secret: str) -> Tuple[str, str, str]:
    """
    Deploy a Modal endpoint for a user using their API key. Unless the user has already done this, its the first thing you must do in order to train a LoRA.
    
    This function deploys a Modal script using the user's Modal API token ID and secret,
    allowing them to set up their own training endpoint. You can create them at https://modal.com/settings/tokens
    
    Parameters:
    - token_id (str, required): The user's Modal token ID
    - token_secret (str, required): The user's Modal token secret
    
    Returns:
    - tuple[str, str, str]: (status_message, stdout, stderr)
      - status_message: Human-readable deployment status
      - stdout: Standard output from the modal deploy command
      - stderr: Standard error output from the modal deploy command
    
    Example usage:
    status, out, err = deploy_for_user("ak-1234567890abcdef", "as-secret123...")
    """
    
    if not token_id or not token_id.strip():
        return "❌ Error: Modal token ID is required", "", ""
    
    if not token_secret or not token_secret.strip():
        return "❌ Error: Modal token secret is required", "", ""
    
    script_path = "diffusers_lora_finetune.py"
    
    # Check if the script file exists
    if not os.path.exists(script_path):
        return f"❌ Error: Script file '{script_path}' not found", "", ""
    
    try:
        # Set up environment with user's Modal tokens
        env = os.environ.copy()
        env["MODAL_TOKEN_ID"] = token_id.strip()
        env["MODAL_TOKEN_SECRET"] = token_secret.strip()
        
        # Run modal deploy command
        result = subprocess.run(
            ["modal", "deploy", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            status_message = "✅ **Deployment Successful!**\n\n"
            status_message += "Your Modal endpoint has been deployed successfully.\n"
            status_message += "Check the output below for your endpoint URL."
            return status_message, result.stdout, result.stderr or "No errors"
        else:
            status_message = "❌ **Deployment Failed**\n\n"
            status_message += f"Exit code: {result.returncode}\n"
            status_message += "Check the error output below for details."
            return status_message, result.stdout or "No output", result.stderr or "No error details"
            
    except subprocess.TimeoutExpired:
        return "❌ Error: Deployment timed out after 5 minutes", "", "Timeout error"
    except FileNotFoundError:
        return "❌ Error: 'modal' command not found. Please install Modal CLI first.", "", "Modal CLI not installed"
    except Exception as e:
        return f"❌ Error: Deployment failed: {str(e)}", "", str(e)

# Create simplified single-page Gradio interface
with gr.Blocks(title="FluxFoundry LoRA Training", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎨 FluxFoundry LoRA Training
    
    Train custom LoRA models for Flux image generation and check training status.
    """)
    
    # Deployment Section
    gr.Markdown("## 🚀 Deploy Your Modal Endpoint")
    gr.Markdown("""
    First, deploy your own Modal endpoint using your Modal API key. This will create your personal training service.
    
    **Requirements:**
    - Modal account and API key
    - The `diffusers_lora_finetune.py` script in your current directory
    """)
    
    with gr.Row():
        with gr.Column():
            token_id = gr.Textbox(
                label="Modal Token ID",
                placeholder="ak-1234567890abcdef...",
                type="password",
                info="Your Modal token ID (found in Modal dashboard)"
            )
            token_secret = gr.Textbox(
                label="Modal Token Secret",
                placeholder="as-secret123...",
                type="password",
                info="Your Modal token secret"
            )
        
        with gr.Column():
            deploy_btn = gr.Button("🚀 Deploy Endpoint", variant="primary", size="lg")
    
    deploy_status = gr.Markdown(label="Deployment Status")
    
    with gr.Row():
        with gr.Column():
            deploy_stdout = gr.Textbox(
                label="Deployment Output",
                lines=10,
                max_lines=15,
                interactive=False,
                info="Standard output from modal deploy"
            )
        with gr.Column():
            deploy_stderr = gr.Textbox(
                label="Deployment Errors",
                lines=10,
                max_lines=15,
                interactive=False,
                info="Error output (if any)"
            )
    
    deploy_btn.click(
        fn=deploy_for_user,
        inputs=[token_id, token_secret],
        outputs=[deploy_status, deploy_stdout, deploy_stderr]
    )
    
    gr.Markdown("---")
    
    # Training Section
    gr.Markdown("## 🎯 Start Training")
    gr.Markdown("After deploying your endpoint above, use it to train LoRA models.")
    
    with gr.Row():
        with gr.Column():
            dataset_id = gr.Textbox(
                label="HuggingFace Dataset ID",
                placeholder="username/dataset-name",
                info="The HuggingFace dataset containing your training images"
            )
            hf_token = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_...",
                type="password",
                info="Your HuggingFace access token with read permissions"
            )
            output_repo = gr.Textbox(
                label="Output Repository",
                placeholder="username/my-lora-model",
                info="HuggingFace repository where the trained LoRA will be uploaded"
            )
            start_training_url = gr.Textbox(
                label="Start Training URL",
                placeholder="https://modal-app-url-api-start-training.modal.run",
                info="Modal API endpoint for starting training"
            )
                
        
        with gr.Column():
            instance_name = gr.Textbox(
                label="Instance Name (Optional)",
                placeholder="subject",
                info="Name of the subject being trained (e.g., 'Fluffy', 'MyDog')"
            )
            class_name = gr.Textbox(
                label="Class Name (Optional)",
                placeholder="person",
                info="Class of the subject (e.g., 'person', 'dog', 'cat')"
            )
            max_train_steps = gr.Slider(
                minimum=100,
                maximum=2000,
                value=500,
                step=50,
                label="Max Training Steps",
                info="Number of training steps (more steps = longer training)"
            )
    
    start_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
    
    with gr.Row():
        training_output = gr.Markdown(label="Training Status")
        job_id_output = gr.Textbox(
            label="Job ID",
            placeholder="Copy this ID to check status",
            interactive=False
        )
    
    start_btn.click(
        fn=start_training,
        inputs=[dataset_id, hf_token, output_repo, start_training_url, instance_name, class_name, max_train_steps],
        outputs=[training_output, job_id_output]
    )
    
    # Status Section
    gr.Markdown("## 📊 Check Status")
    
    job_id_input = gr.Textbox(
        label="Job ID",
        placeholder="Paste your job ID here",
        info="The Job ID returned when you started training"
    )
    job_status_url = gr.Textbox(
        label="Job Status URL",
        placeholder="https://modal-app-url-api-job-status.modal.run",
        info="Modal API endpoint for checking job status"
    )
    
    with gr.Row():
        status_btn = gr.Button("📊 Check Status", variant="secondary")
    
    status_output = gr.Markdown(label="Job Status")
    
    status_btn.click(
        fn=check_job_status,
        inputs=[job_id_input, job_status_url],
        outputs=[status_output]
    )


if __name__ == "__main__":
    print("🎨 Starting FluxFoundry Training Interface...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        mcp_server=True
    )
