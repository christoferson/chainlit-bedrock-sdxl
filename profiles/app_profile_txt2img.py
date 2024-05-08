import os
import boto3
import chainlit as cl
from chainlit.input_widget import Select, Slider, Tags
import traceback
import logging

AWS_REGION = os.environ["AWS_REGION"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]

bedrock = boto3.client("bedrock", region_name=AWS_REGION)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)


async def on_chat_start():
    

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

    await on_settings_update(settings)

#@cl.on_settings_update
async def on_settings_update(settings):

    inference_parameters = dict (
        style_preset = settings["StylePreset"],
        config_scale = settings["ConfigScale"],
        steps = settings["Steps"],
        seed = settings["Seed"],
        samples = settings["Samples"],
        negative_prompts = settings["NegativePrompts"],
    )

    cl.user_session.set("inference_parameters", inference_parameters)


    

#@cl.on_message
async def on_message(message: cl.Message):

    bedrock_model_id = cl.user_session.get("bedrock_model_id")
    inference_parameters = cl.user_session.get("inference_parameters")
    application_options = cl.user_session.get("application_options")

    prompt_template = bedrock_model_strategy.create_prompt(application_options, "", message.content)

    print("End")

