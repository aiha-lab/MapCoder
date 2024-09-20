#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --model GPT4 --dataset HumanEval --strategy MapCoder