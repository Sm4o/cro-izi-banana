import os
from io import BytesIO
import base64

import cv2
from PIL import Image
from io import BytesIO
from potassium import Potassium, Request, Response
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np

import torch

app = Potassium("cro-izi-banana")

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "Linaqruf/anything-v3.0", controlnet=controlnet, torch_dtype=torch.float32,
    ).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights("Sm4o/wanostyle_2_offset", weight_name="wanostyle_2_offset.safetensors", use_auth_token=HF_AUTH_TOKEN)
    pipeline.load_lora_weights("Sm4o/cro9", weight_name="cro9.safetensors", use_auth_token=HF_AUTH_TOKEN)

    context = {
        "model": pipeline,
        "controlnet": controlnet,
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    controlnet = context.get("controlnet")

    # Parse out arguments
    prompt = request.json.get("prompt")
    negative_prompt = request.json.get('negative_prompt', None)
    image_data = request.json.get('image_data', None)
    num_inference_steps = request.json.get("steps", 50)

    # Generate canny image
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB") 
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    canny_image = Image.fromarray(image)
    buffered = BytesIO()
    canny_image.save(buffered,format="JPEG")
    canny_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_model_cpu_offload()
    model.enable_xformers_memory_efficient_attention()

    image = model(
        prompt,
        canny_image,
        negative_prompt=negative_prompt,
        guidance_scale=7,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(request.json.get("seed")) if request.json.get("seed") else None,
        width=512,
        height=768,
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=80)
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # You could also consider writing this image to S3
    # and returning the S3 URL instead of the image data
    # for a slightly faster response time

    return Response(
        json = {
            "canny_base64": canny_base64,
            "image_base64": image_base64,
        },
        status=200
    )

if __name__ == "__main__":
    app.serve()