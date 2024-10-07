import sys
from datetime import datetime
from constants.paths import *

from models.Gemini import Gemini
from models.OpenAI import OpenAIModel

from results.Results import Results

from promptings.PromptingFactory import PromptingFactory
from Datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type=str, 
    default="HumanEval", 
    choices=[
        "HumanEval", 
        "MBPP", 
        "APPS",
        "xCodeEval", 
        "CC", 
    ]
)
parser.add_argument(
    "--strategy", 
    type=str, 
    default="MapCoder", 
    choices=[
        "Direct",
        "CoT",
        "SelfPlanning",
        "Analogical",
        "MapCoder",
    ]
)
parser.add_argument(
    "--model", 
    type=str, 
    default="ChatGPT", 
    choices=[
        "ChatGPT",
        "GPT4",
        "Gemini",
        "Mistral",
        "Qwen",
        "QwenGPTQ",
    ],
    help="Model to use for the strategy or coding for MapCoder"
)
parser.add_argument(
    "--retrieval", 
    type=str, 
    default="None", 
    choices=[
        "ChatGPT",
        "GPT4",
        "Gemini",
        "Mistral",
        "Qwen",
        "QwenGPTQ",
    ]
)
parser.add_argument(
    "--planning", 
    type=str, 
    default="None", 
    choices=[
        "ChatGPT",
        "GPT4",
        "Gemini",
        "Mistral",
        "Qwen",
        "QwenGPTQ",
    ]
)
parser.add_argument(
    "--debugging", 
    type=str, 
    default="None", 
    choices=[
        "ChatGPT",
        "GPT4",
        "Gemini",
        "Mistral",
        "Qwen",
        "QwenGPTQ",
    ]
)
parser.add_argument(
    "--temperature", 
    type=float, 
    default=0
)
parser.add_argument(
    "--pass_at_k", 
    type=int, 
    default=1
)
parser.add_argument(
    "--language", 
    type=str, 
    default="Python3", 
    choices=[
        "C",
        "C#",
        "C++",
        "Go",
        "PHP",
        "Python3",
        "Ruby",
        "Rust",
    ]
)

args = parser.parse_args()

DATASET = args.dataset
STRATEGY = args.strategy
MODEL_NAME = args.model
RETRIEVAL = args.retrieval
PLANNING = args.planning
DEBUGGING = args.debugging
TEMPERATURE = args.temperature
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language

if STRATEGY == "MapCoder":
    RUN_NAME = f"{RETRIEVAL}-{PLANNING}-{MODEL_NAME}-{DEBUGGING}-{STRATEGY}-{DATASET}-{LANGUAGE}-{TEMPERATURE}-{PASS_AT_K}"
else:
    RUN_NAME = f"{MODEL_NAME}-{STRATEGY}-{DATASET}-{LANGUAGE}-{TEMPERATURE}-{PASS_AT_K}"
RESULTS_PATH = f"./outputs/{RUN_NAME}.jsonl"

print(f"#########################\nRunning start {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_NAME)(temperature=TEMPERATURE),
    retrieval=RETRIEVAL,
    planning=PLANNING,
    debugging=DEBUGGING,
    data=DatasetFactory.get_dataset_class(DATASET)(),
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,
    results=Results(RESULTS_PATH),
)

strategy.run()

print(f"#########################\nRunning end {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

