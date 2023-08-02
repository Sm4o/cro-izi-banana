# This file runs during container build time to get model weights built into the container
import os

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
import torch


HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # this should match the model load used in app.py's init function
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
    pipeline = StableDiffusionPipeline.from_pretrained(
        "Linaqruf/anything-v3.0", controlnet=controlnet, torch_dtype=torch.float32,
    ).to("cuda")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights("Sm4o/wanostyle_2_offset", weight_name="wanostyle_2_offset.safetensors", use_auth_token=HF_AUTH_TOKEN)
    pipeline.load_lora_weights("Sm4o/cro9", weight_name="cro9.safetensors", use_auth_token=HF_AUTH_TOKEN)


if __name__ == "__main__":
    download_model()