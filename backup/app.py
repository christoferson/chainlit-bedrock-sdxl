import chainlit as cl
import boto3
import base64
import io
import datetime
import random
from PIL import Image
import json
import os

bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-east-1")


shoe_2D_IMG_IMG_RED = dict(
    text = """
    product photo, yellow base color, red sole, running shoes, Asics, photorealistic, highly detailed and intricate, vibrant color, bokeh nature background, red shoelace
    """,
    negative=[
        "ugly", "tiling", "out of frame",
        "disfigured", "deformed", "bad anatomy", "cut off", "low contrast", 
        "underexposed", "overexposed", "bad art", "beginner", "amateur", "blurry", "draft", "grainy"
    ],
    style="photographic", #"3d-model",
    scale=12,
    image_strength=0.01
)


@cl.on_message
async def main(message: cl.Message):

    model_id = "stability.stable-diffusion-xl-v1"

    negative=[
        "ugly", "tiling", "out of frame",
        "disfigured", "deformed", "bad anatomy", "cut off", "low contrast", 
        "underexposed", "overexposed", "bad art", "beginner", "amateur", "blurry", "draft", "grainy"
    ]

    demo_sd_generate_text_to_image_xl_v1(model_id, message.content, negative, "photographic", 12)
    
    image = cl.Image(path="./output/img.png", name="image1", display="inline")

    
    await cl.Message(content=f"Received: {message.content}", elements=[image]).send()



# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html
def demo_sd_generate_text_to_image_xl_v1(model_id, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10):

    print(f"Call demo_sd_generate_text_to_image_xl_v1 | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "output/{}{}".format("img", file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 4294967295)
    steps = 50 #150 #30 #50
    #cfg_scale = cfg_scale
    start_schedule = 0.6
    change_prompt = prompt
    #negative_prompts = negative_prompts
    #style_preset = style_preset
    size = 1024

    # 
    config = {
        "filename": OUTPUT_IMG_PATH,
        "seed": seed,
        "change_prompt": change_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "start_schedule": start_schedule,
        "style_preset": style_preset,
        "size": size,
        "negative_prompts": negative_prompts
    }

    # 
    body = json.dumps(
        {
            "text_prompts": (
                [{"text": config["change_prompt"], "weight": 1.0}]
                + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
            ),
            "cfg_scale": config["cfg_scale"], # Determines how much the final image portrays the prompt. Use a lower number to increase randomness in the generation. 0-35,7
            #"clip_guidance_preset"
            #"height": "1024",
            #"width": "1024",
            "seed": config["seed"], # The seed determines the initial noise setting.0-4294967295,0
            #"start_schedule": config["start_schedule"],
            "steps": config["steps"], # Generation step determines how many times the image is sampled. 10-50,50
            "style_preset": config["style_preset"],
            "samples": 1,
        }
    )

    print(body)

    #print(body)

    # 
    print("Generating Image ...")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    # 
    response_body = json.loads(response.get("body").read())
    response_image = base64_to_image(response_body["artifacts"][0].get("base64"))

    # 
    response_image.save(OUTPUT_IMG_PATH)
    # 
    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(config, f, ensure_ascii = False)

    print("Complete")

### Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
