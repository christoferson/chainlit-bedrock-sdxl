import chainlit as cl
import boto3
from botocore.config import Config
import base64
import io
import random
from PIL import Image
import json
import os
from chainlit.input_widget import Select, Slider, Switch
import logging
import traceback

AWS_REGION = os.environ["AWS_REGION"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]

config = Config(read_timeout=1000)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION, config=config)

#Todo multiple samples: samples 

async def setup_settings():

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
                label="StylePreset",
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
        ]
    ).send()

    print("Save Settings: ", settings)

    return settings

@cl.on_chat_start
async def main():

    #session_id = str(uuid.uuid4())

    #cl.user_session.set("session_id", session_id)
    
    settings = await setup_settings()

    #await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):

    knowledge_base_id = settings["KnowledgeBase"]
    #knowledge_base_id = knowledge_base_id.split(" ", 1)[0]
    
    #llm_model_arn = "arn:aws:bedrock:{}::foundation-model/{}".format(AWS_REGION, settings["Model"])
    #mode = settings["Mode"]
    #strict = settings["Strict"]
    #kb_retrieve_document_count = int(settings["RetrieveDocumentCount"])

    #bedrock_model_id = settings["Model"]

    inference_parameters = dict (
        #temperature = settings["Temperature"],
        #top_p = float(settings["TopP"]),
        #top_k = int(settings["TopK"]),
        #max_tokens_to_sample = int(settings["MaxTokenCount"]),
        #stop_sequences =  [],
    )


    cl.user_session.set("inference_parameters", inference_parameters)

@cl.on_message
async def main(message: cl.Message):

    model_id = "stability.stable-diffusion-xl-v1"

    negative=[
        "ugly", "tiling", "out of frame",
        "disfigured", "deformed", "bad anatomy", "cut off", "low contrast", 
        "underexposed", "overexposed", "bad art", "beginner", "amateur", "blurry", "draft", "grainy"
    ]

    style_preset = "photographic"
    cfg_scale = 12

    msg = cl.Message(content="Generating...")

    await msg.send()

    async with cl.Step(name="Model", type="llm", root=False) as step_llm:
        step_llm.input = msg.content

        try:
            
            await generate_text_to_image(step_llm, model_id, message.content, negative, style_preset, cfg_scale)
            
            image = cl.Image(path="./output/img.png", name="image1", display="inline")

            msg.content = f"style_preset={style_preset}, cfg_scale={cfg_scale}"
            msg.elements = [image]
            await msg.update()

        except Exception as e:
            logging.error(traceback.format_exc())
            await msg.stream_token(f"{e}")
        finally:
            await msg.send()



# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html
async def generate_text_to_image(step_llm : cl.Step, model_id, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10):

    print(f"Call demo_sd_generate_text_to_image_xl_v1 | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "output/{}{}".format("img", file_extension))
    OUTPUT_IMG_PATH = os.path.join("./output/{}{}".format("img", file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 4294967295)
    steps = 50 #150 #30 #50
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

    #print(body)

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
    

### Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
