import os
from io import BytesIO
import base64

from potassium import Potassium, Request, Response
# from diffusers import DiffusionPipeline, DDPMScheduler
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch

app = Potassium("cro-izi-banana")

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    pipeline = StableDiffusionPipeline.from_pretrained(
        "Linaqruf/anything-v3.0", torch_dtype=torch.float32,
    ).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights("Sm4o/wanostyle_2_offset", weight_name="wanostyle_2_offset.safetensors", use_auth_token=HF_AUTH_TOKEN)
    pipeline.load_lora_weights("Sm4o/cro9", weight_name="cro9.safetensors", use_auth_token=HF_AUTH_TOKEN)

    context = {
        "model": pipeline,
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")

    prompt = request.json.get("prompt")
    negative_prompt = (
        "face,  ((eyes)), mouth (painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), "
        "watermark, text, error, ((blurry)), jpeg artifacts, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, "
        "low quality:1.4), bad anatomy, watermark, signature, text, logo"
    )

    image = model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7,
        num_inference_steps=request.json.get("steps", 50),
        generator=torch.Generator(device="cuda").manual_seed(request.json.get("seed")) if request.json.get("seed") else None,
        width=512,
        height=768,
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue())

    # You could also consider writing this image to S3
    # and returning the S3 URL instead of the image data
    # for a slightly faster response time

    return Response(
        json = {"output": str(img_str, "utf-8")},
        status=200
    )

if __name__ == "__main__":
    app.serve()