import os
import boto3
import chainlit as cl
from chainlit.input_widget import Select, Slider
import traceback
import logging

AWS_REGION = os.environ["AWS_REGION"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]

bedrock = boto3.client("bedrock", region_name=AWS_REGION)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

async def on_chat_start():
    
    model_ids = ["anthropic.claude-3-sonnet-20240229-v1:0"]
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Amazon Bedrock - Model",
                values=model_ids,
                initial_index=model_ids.index("anthropic.claude-3-sonnet-20240229-v1:0"),
                
            )
        ]
    ).send()
    await on_settings_update(settings)

#@cl.on_settings_update
async def on_settings_update(settings):

    bedrock_model_id = settings["Model"]

    application_options = dict (
        option_terse = False,
        option_strict = False
    )

    inference_parameters = dict (
        temperature = settings["Temperature"],
        top_p = float(settings["TopP"]),
        top_k = int(settings["TopK"]),
        max_tokens_to_sample = int(settings["MaxTokenCount"]),
        system_message = "You are a helpful assistant.",
        stop_sequences =  []
    )

    cl.user_session.set("bedrock_model_strategy", model_strategy)
    

#@cl.on_message
async def on_message(message: cl.Message):

    bedrock_model_id = cl.user_session.get("bedrock_model_id")
    inference_parameters = cl.user_session.get("inference_parameters")
    application_options = cl.user_session.get("application_options")

    prompt_template = bedrock_model_strategy.create_prompt(application_options, "", message.content)

    print("End")

