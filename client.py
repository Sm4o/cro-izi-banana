
import base64


import banana_dev as client

# Create a reference to your model on Banana
model = client.Client(
    api_key="cd1ddb1f-e07d-4e86-b12b-6db2fb7c5eec",
    model_key="eaf40da7-7890-461e-9fbc-9ba35e788b86",
    url="https://cro-izi-banana-ljskrrjscd.run.banana.dev",
)

inputs = {
    "prompt": "Ape drinking a mojito",
}

img_data, meta = model.call("/", inputs)

print(img_data)
breakpoint()

with open("imageToSave.png", "wb") as image_file:
    image_file.write(base64.b64decode(img_data["output"]))
