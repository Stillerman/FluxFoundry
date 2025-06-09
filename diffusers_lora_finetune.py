# ---
# deploy: true
# ---


from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import modal

app = modal.App(name="dreambooth-lora-flux")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.31.0",
    "datasets==3.6.0",
    "pillow",
    "fastapi[standard]==0.115.4",
    "ftfy~=6.1.0",
    "gradio~=5.5.0",
    "huggingface-hub==0.32.4",
    "hf_transfer==0.1.8",
    "numpy<2",
    "peft==0.11.1",
    "pydantic==2.9.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "smart_open~=6.4.0",
    "starlette==0.41.2",
    "transformers~=4.41.2",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "wandb==0.17.6",
)


GIT_SHA = "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # specify the commit to fetch

image = (
    image.apt_install("git")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's home directory, /root. Then install `diffusers`
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
)

# ### Configuration with `dataclass`es

# Machine learning apps often have a lot of configuration information.
# We collect up all of our configuration into dataclasses to avoid scattering special/magic values throughout code.


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "Qwerty"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "Golden Retriever"
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-dev"


# ### Storing data created by our app with `modal.Volume`

# The tools we've used so far work well for fetching external information,
# which defines the environment our app runs in,
# but what about data that we create or modify during the app's execution?
# A persisted [`modal.Volume`](https://modal.com/docs/guide/volumes) can store and share data across Modal Apps and Functions.

# We'll use one to store both the original and fine-tuned weights we create during training
# and then load them back in for inference.



image = image.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"}  # turn on faster downloads from HF
)

def load_images_from_hf_dataset(dataset_id: str, hf_token: str) -> Path:
    """Load images from a HuggingFace dataset."""
    import PIL.Image
    from datasets import load_dataset

    img_path = Path("/img")
    img_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_id, token=hf_token, split="train")
    
    for ii, example in enumerate(dataset):
        # Assume the dataset has an 'image' column
        if 'image' in example:
            image = example['image']
            if isinstance(image, PIL.Image.Image):
                image.save(img_path / f"{ii}.png")
            else:
                # Handle other image formats
                pil_image = PIL.Image.open(image)
                pil_image.save(img_path / f"{ii}.png")
        else:
            print(f"Warning: No 'image' field found in dataset example {ii}")
    
    print(f"{len(dataset)} images loaded from HuggingFace dataset")
    return img_path


# ## Stateless API Training Function

@dataclass
class APITrainConfig:
    """Configuration for the API training function."""
    
    # Basic model info
    model_name: str = "black-forest-labs/FLUX.1-dev"
    
    # Training prompt components
    instance_name: str = "subject"
    class_name: str = "person"
    prefix: str = "a photo of"
    postfix: str = ""
    
    # Training hyperparameters
    resolution: int = 512
    train_batch_size: int = 3
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117


@app.function(
    image=image,
    gpu="A100-80GB",  # fine-tuning is VRAM-heavy and requires a high-VRAM GPU
    timeout=3600,  # 60 minutes
)
def train_lora_stateless(
    dataset_id: str,
    hf_token: str,
    output_repo: str,
    instance_name: Optional[str] = None,
    class_name: Optional[str] = None,
    max_train_steps: int = 500,
):
    """
    Stateless LoRA training function that reads from HF dataset and uploads to HF repo.
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "username/dataset-name")
        hf_token: HuggingFace API token
        output_repo: HuggingFace repository to upload the trained LoRA to
        instance_name: Name of the subject (optional, defaults to "subject")
        class_name: Class of the subject (optional, defaults to "person")
        max_train_steps: Number of training steps
    """
    import subprocess
    import tempfile
    from pathlib import Path
    
    import torch
    from accelerate.utils import write_basic_config
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download, upload_folder, login, create_repo
    
    # Login to HuggingFace
    login(token=hf_token)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        model_dir = temp_path / "model"
        output_dir = temp_path / "output"
        
        # Download base model
        print("📥 Downloading base model...")
        snapshot_download(
            "black-forest-labs/FLUX.1-dev",
            local_dir=str(model_dir),
            ignore_patterns=["*.pt", "*.bin"],  # using safetensors
            token=hf_token
        )
        
        # Load and validate model
        DiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
        print("✅ Base model loaded successfully")
        
        # Load training images from HF dataset
        print(f"📥 Loading images from dataset: {dataset_id}")
        img_path = load_images_from_hf_dataset(dataset_id, hf_token)
        
        # Set up training configuration
        config = APITrainConfig(
            instance_name=instance_name or "subject",
            class_name=class_name or "person", 
            max_train_steps=max_train_steps
        )
        
        # Set up hugging face accelerate library for fast training
        write_basic_config(mixed_precision="bf16")
        
        # Define the training prompt
        instance_phrase = f"{config.instance_name} the {config.class_name}"
        prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()
        
        print(f"🎯 Training prompt: {prompt}")
        print(f"🚀 Starting training for {max_train_steps} steps...")
        
        # Execute training subprocess
        def _exec_subprocess(cmd: list[str]):
            """Executes subprocess and prints log to terminal while subprocess is running."""
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            with process.stdout as pipe:
                for line in iter(pipe.readline, b""):
                    line_str = line.decode()
                    print(f"{line_str}", end="")

            if exitcode := process.wait() != 0:
                raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))
        
        # Run training
        _exec_subprocess([
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",
            f"--pretrained_model_name_or_path={model_dir}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={output_dir}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",
        ])
        
        print("✅ Training completed!")
        
        # Upload trained LoRA to HuggingFace repository
        
        print(f"📤 Uploading LoRA to repository: {output_repo}")

        # Create repository if it doesn't exist
        create_repo(
            repo_id=output_repo,
            repo_type="model",
            token=hf_token,
            exist_ok=True
        )

        # print contents of output_dir
        print(f"Contents of {output_dir}:")
        for file in output_dir.iterdir():
            print(file)

        upload_folder(
            folder_path=str(output_dir),
            repo_id=output_repo,
            repo_type="model",
            token=hf_token,
            commit_message=f"Add LoRA trained on {dataset_id}",
        )
        
        print(f"🎉 Successfully uploaded LoRA to {output_repo}")
        
        return {
            "status": "success",
            "message": f"LoRA training completed and uploaded to {output_repo}",
            "dataset_used": dataset_id,
            "training_steps": max_train_steps,
            "training_prompt": prompt
        }


# ## API Endpoints with Job ID System

@app.function(
    image=image,
    keep_warm=1,  # Keep one container warm for faster response
)
@modal.fastapi_endpoint(method="POST")
def api_start_training(item: dict):
    """
    Start LoRA training and return a job ID.
    
    Expected JSON payload:
    {
        "dataset_id": "username/dataset-name",
        "hf_token": "hf_...",
        "output_repo": "username/output-repo",
        "instance_name": "optional_subject_name",
        "class_name": "optional_class_name", 
        "max_train_steps": 500
    }
    """
    try:
        # Extract required parameters
        dataset_id = item["dataset_id"]
        hf_token = item["hf_token"] 
        output_repo = item["output_repo"]
        
        # Extract optional parameters
        instance_name = item.get("instance_name")
        class_name = item.get("class_name")
        max_train_steps = item.get("max_train_steps", 500)
        
        # Start training (non-blocking)
        call_handle = train_lora_stateless.spawn(
            dataset_id=dataset_id,
            hf_token=hf_token,
            output_repo=output_repo,
            instance_name=instance_name,
            class_name=class_name,
            max_train_steps=max_train_steps
        )
        
        job_id = call_handle.object_id
        
        return {
            "status": "started",
            "job_id": job_id,
            "message": "Training job started successfully",
            "dataset_id": dataset_id,
            "output_repo": output_repo,
            "max_train_steps": max_train_steps
        }
        
    except KeyError as e:
        return {
            "status": "error",
            "message": f"Missing required parameter: {e}"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to start training: {str(e)}"
        }


@app.function(
    image=image,
    keep_warm=1,
)
@modal.fastapi_endpoint(method="GET")
def api_job_status(job_id: str):
    """
    Check the status of a training job.
    Pass job_id as a query parameter: /job_status?job_id=xyz
    """
    try:
        from modal.functions import FunctionCall
        
        # Get the function call handle
        call_handle = FunctionCall.from_id(job_id)
        
        if call_handle is None:
            return {
                "status": "error",
                "message": "Job not found"
            }
        
        # Check if the job is finished
        try:
            result = call_handle.get(timeout=0)  # Non-blocking check
            return {
                "status": "completed",
                "result": result
            }
        except TimeoutError:
            return {
                "status": "running",
                "message": "Job is still running"
            }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Job failed: {str(e)}"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking job status: {str(e)}"
        }

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512


@app.function(
    image=image,
    gpu="A100",  # Inference requires GPU
    timeout=1800,  # 30 minutes
)
def generate_images_stateless(
    hf_token: str,
    lora_repo: str,
    prompts: list[str],
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
):
    """
    Stateless function to generate images using a LoRA from HuggingFace.
    
    Args:
        hf_token: HuggingFace API token
        lora_repo: HuggingFace repository containing the LoRA (e.g., "username/my-lora")
        prompts: List of text prompts to generate images for
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        width: Image width
        height: Image height
    
    Returns:
        Dictionary with status and list of generated images (as base64 strings)
    """
    import base64
    import io
    import tempfile
    from pathlib import Path
    
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download, login
    
    try:
        # Login to HuggingFace
        login(token=hf_token)
        
        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = temp_path / "model"
            lora_dir = temp_path / "lora"
            
            print("📥 Downloading base model...")
            # Download base model
            snapshot_download(
                "black-forest-labs/FLUX.1-dev",
                local_dir=str(model_dir),
                ignore_patterns=["*.pt", "*.bin"],  # using safetensors
                token=hf_token
            )
            
            print(f"📥 Downloading LoRA from {lora_repo}...")
            # Download LoRA
            snapshot_download(
                lora_repo,
                local_dir=str(lora_dir),
                token=hf_token
            )
            
            print("🔄 Loading pipeline...")
            # Load the diffusion pipeline
            pipe = DiffusionPipeline.from_pretrained(
                str(model_dir),
                torch_dtype=torch.bfloat16,
            ).to("cuda")
            
            # Load LoRA weights
            pipe.load_lora_weights(str(lora_dir))
            
            print(f"🎨 Generating {len(prompts)} images...")
            generated_images = []
            
            # Generate images for each prompt
            for i, prompt in enumerate(prompts):
                print(f"  Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
                
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                ).images[0]
                
                # Convert PIL Image to base64 string
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                generated_images.append({
                    "prompt": prompt,
                    "image": img_base64
                })
            
            print("✅ All images generated successfully!")
            
            return {
                "status": "success",
                "message": f"Generated {len(prompts)} images successfully",
                "lora_repo": lora_repo,
                "images": generated_images
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate images: {str(e)}"
        }


@app.function(
    image=image,
    keep_warm=1,
)
@modal.fastapi_endpoint(method="POST")
def api_generate_images(item: dict):
    """
    Generate images using a LoRA model.
    
    Expected JSON payload:
    {
        "hf_token": "hf_...",
        "lora_repo": "username/my-lora",
        "prompts": ["prompt1", "prompt2", ...],
        "num_inference_steps": 20,  // optional
        "guidance_scale": 7.5,      // optional
        "width": 512,               // optional
        "height": 512               // optional
    }
    """
    try:
        # Extract required parameters
        hf_token = item["hf_token"]
        lora_repo = item["lora_repo"] 
        prompts = item["prompts"]
        
        if not isinstance(prompts, list) or len(prompts) == 0:
            return {
                "status": "error",
                "message": "prompts must be a non-empty list"
            }
        
        # Extract optional parameters
        num_inference_steps = item.get("num_inference_steps", 20)
        guidance_scale = item.get("guidance_scale", 7.5)
        width = item.get("width", 512)
        height = item.get("height", 512)
        
        # Start generation (non-blocking)
        call_handle = generate_images_stateless.spawn(
            hf_token=hf_token,
            lora_repo=lora_repo,
            prompts=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )
        
        job_id = call_handle.object_id
        
        return {
            "status": "started",
            "job_id": job_id,
            "message": "Image generation job started successfully",
            "lora_repo": lora_repo,
            "num_prompts": len(prompts)
        }
        
    except KeyError as e:
        return {
            "status": "error",
            "message": f"Missing required parameter: {e}"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to start image generation: {str(e)}"
        }