#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/main.py --model Mistral --dataset HumanEval --strategy MapCoder