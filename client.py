
import base64


import banana_dev as client

# Create a reference to your model on Banana
model = client.Client(
    api_key="cd1ddb1f-e07d-4e86-b12b-6db2fb7c5eec",
    model_key="eaf40da7-7890-461e-9fbc-9ba35e788b86",
    url="https://cro-izi-banana-ljskrrjscd.run.banana.dev",
)

inputs = {
    "prompt": (
        "man wearing a ((white helmet covering face with white vizor))  <lora:cro9:0.6> , wanostyle art"
        " , monkey d luffy,  solo,, ((masterpiece)), (best quality), (extremely detailed), "
        "depth of field, sketch, dark intense shadows, sharp focus, soft lighting, hdr, colorful, "
        "good composition,  spectacular, closed shirt, anime screencap,, "
        "<lora:onePieceWanoSagaStyle_v2Offset:1>  , <lora:wanostyle_2_offset:1>  ((good contrast))"
    ),
    "seed": 1064668731
}

img_data, meta = model.call("/", inputs)

print(img_data)

with open("imageToSave3.png", "wb") as image_file:
    image_file.write(base64.b64decode(img_data["output"]))
