import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from .Base import BaseModel

class HuggingFaceBaseModel(BaseModel):
    def __init__(
        self,
        model_name='mistralai/Mistral-7B-Instruct-v0.3',
        id="Mistral",
        device=None,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_tokens=16384,
    ):
        self.model_name = model_name
        self.id = id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Load model and tokenizer
        # import pdb; pdb.set_trace()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto", 
            torch_dtype="auto"
        )
        # self.model.to(self.device)
        # self.model.parallelize()

    # def load_model(model_name):
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.model_name,
    #         device_map="auto",
    #         torch_dtype="auto"
    #         )

    def prompt(self, processed_input: list[dict]):
        # Generate prompt text
        prompt_text = [{"role": processed_input[0]["role"], "content": processed_input[0]["content"]}] 
        
        # Tokenize prompt text
        # input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device)

        # format and tokenize the tool use prompt 
        inputs = self.tokenizer.apply_chat_template(
            prompt_text,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Set max length
        # max_length = prompt_tokens + self.max_tokens
        
        # Generate text
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            # temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=True,
            max_time=180,
            # pad_token_id=self.tokenizer.eos_token_id,
            # eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode generated text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Calculate number of tokens
        total_tokens = output_ids.size(1)
        prompt_tokens = inputs['input_ids'].size(1)
        completion_tokens = total_tokens - prompt_tokens
        
        # Extract response text from the assistant
        response_text = generated_text[len(prompt_text[0]['content']):].strip()

        return response_text, prompt_tokens, completion_tokens


class Mistral(HuggingFaceBaseModel):
    def __init__(
        self,
        model_name='mistralai/Mistral-7B-Instruct-v0.3',
        id="Mistral",
        device=None,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_tokens=16384,
    ):
        super().__init__(
            model_name=model_name,
            id=id,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
       )

class Qwen(HuggingFaceBaseModel):
    def __init__(
        self,
        model_name='Qwen/Qwen2.5-32B-Instruct',
        id='Qwen',
        device=None,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_tokens=16384,
    ):
        super().__init__(
            model_name=model_name,
            id=id,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
       )

class QwenGPTQ(HuggingFaceBaseModel):
    def __init__(
        self,
        model_name='Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4',
        id='QwenGPTQ',
        device=None,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_tokens=16384,
    ):
        super().__init__(
            model_name=model_name,
            id=id,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
       )