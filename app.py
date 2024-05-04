import chainlit as cl
import boto3
from botocore.config import Config
import base64
import io
import random
from PIL import Image
import json
import os
from chainlit.input_widget import Select, Slider, Tags
import logging
import traceback
from typing import Optional, List

AWS_REGION = os.environ["AWS_REGION"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]

config = Config(read_timeout=1000)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION, config=config)

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == (AUTH_ADMIN_USR, AUTH_ADMIN_PWD):
    return cl.User(identifier=AUTH_ADMIN_USR, metadata={"role": "admin", "provider": "credentials"})
  #elif (username, password) == (AUTH_USER_USR, AUTH_USER_PWD):
  #  return cl.User(identifier=AUTH_USER_USR, metadata={"role": "user", "provider": "credentials"})
  else:
    return None

async def setup_settings():

    negative=[
        "ugly", "tiling", "out of frame",
        "disfigured", "deformed", "bad anatomy", "cut off", "low contrast", 
        "underexposed", "overexposed", "bad art", "beginner", "amateur", "blurry", "draft", "grainy"
    ]

    settings = await cl.ChatSettings(
        [
            
            Slider(
                id = "ConfigScale",
                label = "Config Scale",
                initial = 10,
                min = 0,
                max = 35,
                step = 1,
            ),
            Slider(
                id = "Steps",
                label = "Steps",
                initial = 30,
                min = 10,
                max = 50,
                step = 1,
            ),
            Select(
                id="StylePreset",
                label="Style Preset",
                values=["anime", "photographic"],
                initial_index=1,
            ),
            Slider(
                id = "Seed",
                label = "Seed",
                initial = 0,
                min = 0,
                max = 4294967295,
                step = 1,
            ),
            Slider(
                id = "Samples",
                label = "Samples",
                initial = 1,
                min = 1,
                max = 4,
                step = 1,
            ),
            Tags(id="NegativePrompts", label="Negative Prompts", initial=negative),
        ]
    ).send()

    print("Save Settings: ", settings)

    return settings

@cl.on_chat_start
async def main():

    #session_id = str(uuid.uuid4())

    #cl.user_session.set("session_id", session_id)
    
    settings = await setup_settings()

    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):

    inference_parameters = dict (
        style_preset = settings["StylePreset"],
        config_scale = settings["ConfigScale"],
        steps = settings["Steps"],
        seed = settings["Seed"],
        samples = settings["Samples"],
        negative_prompts = settings["NegativePrompts"],
    )

    cl.user_session.set("inference_parameters", inference_parameters)

@cl.on_message
async def main(message: cl.Message):

    model_id = "stability.stable-diffusion-xl-v1"
    inference_parameters = cl.user_session.get("inference_parameters") 
    style_preset = inference_parameters.get("style_preset")
    seed = int(inference_parameters.get("seed"))
    cfg_scale = int(inference_parameters.get("config_scale"))
    steps = int(inference_parameters.get("steps"))
    samples = int(inference_parameters.get("samples"))
    negative_prompts = inference_parameters.get("negative_prompts")

    msg = cl.Message(content="Generating...")

    await msg.send()

    async with cl.Step(name="Model", type="llm", root=False) as step_llm:
        step_llm.input = msg.content

        try:

            msg.content = f"style_preset={style_preset}, cfg_scale={cfg_scale} steps={steps} seed={seed}"
            await msg.update()
            
            #image_path_list = await generate_text_to_image_v2(step_llm, model_id, message.content, negative, inference_parameters)
            
            msg.elements = []
            for i in range(samples):
                image_path = await generate_text_to_image_v3(step_llm, model_id, message.content, negative_prompts, inference_parameters, i+1)
                image = cl.Image(path=image_path, name="image1", display="inline") #size="large"

                msg.elements.append(image)
                await msg.update()

            #msg.content = f"style_preset={style_preset}, cfg_scale={cfg_scale} steps={steps} seed={seed}"
            await msg.update()

        except Exception as e:
            logging.error(traceback.format_exc())
            await msg.stream_token(f"{e}")
        finally:
            await msg.send()



# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html
async def generate_text_to_image(step_llm : cl.Step, model_id, prompt, negative_prompts, inference_parameters):

    style_preset = inference_parameters.get("style_preset")
    seed = int(inference_parameters.get("seed"))
    cfg_scale = int(inference_parameters.get("config_scale"))
    steps = int(inference_parameters.get("steps"))

    print(f"Call demo_sd_generate_text_to_image_xl_v1 | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "output/{}{}".format("img", file_extension))
    OUTPUT_IMG_PATH = os.path.join("./output/{}{}".format("img", file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = int(inference_parameters.get("seed"))
    if seed == 0:
        seed = random.randint(0, 4294967295)
        #inference_parameters["seed"] = seed
    #steps = #50 #150 #30 #50
    start_schedule = 0.6
    change_prompt = prompt
    size = 1024

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

    # 
    await step_llm.stream_token("Generating Image ...\n")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    # 
    response_body = json.loads(response.get("body").read())
    response_image = base64_to_image(response_body["artifacts"][0].get("base64"))

    await step_llm.stream_token("Saving Image ...\n")
    response_image.save(OUTPUT_IMG_PATH)
    # 
    #with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
    #    json.dump(config, f, ensure_ascii = False)
    await step_llm.send()
    print("Complete")



# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html
async def generate_text_to_image_v2(step_llm : cl.Step, model_id, prompt, negative_prompts, inference_parameters) -> List[str]:

    style_preset = inference_parameters.get("style_preset")
    seed = int(inference_parameters.get("seed"))
    cfg_scale = int(inference_parameters.get("config_scale"))
    steps = int(inference_parameters.get("steps"))
    samples = int(inference_parameters.get("samples"))

    print(f"Call generate_text_to_image_v2 | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    #ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    #OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "output/{}{}".format("img", file_extension))
    

    seed = int(inference_parameters.get("seed"))
    if seed == 0:
        seed = random.randint(0, 4294967295)
        #inference_parameters["seed"] = seed
    #steps = #50 #150 #30 #50
    start_schedule = 0.6
    change_prompt = prompt
    size = 1024

    config = {
        #"filename": OUTPUT_IMG_PATH,
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
            "samples": samples,
        }
    )

    # 
    await step_llm.stream_token("Generating Image ...\n")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    #
    image_path_list = []
    response_body = json.loads(response.get("body").read())
    idx = 1
    for artifact in response_body["artifacts"]:
        response_image : Image = base64_to_image(artifact.get("base64"))
        image_filename = f"{'img'}-{idx}{file_extension}"
        image_path = os.path.join("./output/{}".format(image_filename))
        print("OUTPUT_IMG_PATH: " + image_path)

        await step_llm.stream_token(f"Saving Image {image_filename}...\n")
        response_image.save(image_path)

        await step_llm.send()
        idx = idx + 1
        image_path_list.append(image_path)

    print("Complete")
    return image_path_list






# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html
async def generate_text_to_image_v3(step_llm : cl.Step, model_id, prompt, negative_prompts, inference_parameters, idx : int) -> str:

    style_preset = inference_parameters.get("style_preset")
    seed = int(inference_parameters.get("seed"))
    cfg_scale = int(inference_parameters.get("config_scale"))
    steps = int(inference_parameters.get("steps"))

    print(f"Call demo_sd_generate_text_to_image_xl_v1 | style_preset={style_preset} | cfg_scale={cfg_scale} | neg={negative_prompts}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    #ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    #OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "output/{}{}".format("img", file_extension))
    OUTPUT_IMG_PATH = os.path.join("./output/{}-{}{}".format("img", idx, file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = int(inference_parameters.get("seed"))
    if seed == 0:
        seed = random.randint(0, 4294967295)
        #inference_parameters["seed"] = seed
    #steps = #50 #150 #30 #50
    start_schedule = 0.6
    change_prompt = prompt
    size = 1024

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

    # 
    await step_llm.stream_token("Generating Image ...\n")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    # 
    response_body = json.loads(response.get("body").read())
    response_image_base64 = response_body["artifacts"][0].get("base64")
    response_image = base64_to_image(response_image_base64)

    await step_llm.stream_token("Saving Image ...\n")
    response_image.save(OUTPUT_IMG_PATH)
    # 
    #with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
    #    json.dump(config, f, ensure_ascii = False)
    await step_llm.send()
    print("Complete")
    return OUTPUT_IMG_PATH



### Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str) -> Image:
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
