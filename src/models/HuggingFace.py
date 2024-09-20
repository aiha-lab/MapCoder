import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from .Base import BaseModel

class HuggingFaceBaseModel(BaseModel):
    def __init__(
        self,
        model_name='mistralai/Mistral-7B-Instruct-v0.3',
        device=None,
        temperature=0.7,
        top_p=0.9,
        max_tokens=32768,
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def prompt(self, processed_input: list[dict]):
        # Generate prompt text
        prompt_text = [{"role": processed_input[0]["role"], "content": processed_input[0]["content"]}] 
        # for message in processed_input:
        #     role = message.get('role', '')
        #     content = message.get('content', '')
        #     import pdb; pdb.set_trace()
        #     if role == 'system':
        #         prompt_text += f"System: {content}\n"
        #     elif role == 'user':
        #         prompt_text += f"User: {content}\n"
        #     elif role == 'assistant':
        #         prompt_text += f"Assistant: {content}\n"

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
        # max_length = input_ids.size(1) + self.max_tokens
        
        import pdb; pdb.set_trace()

        # Generate text
        output_ids = self.model.generate(
            **inputs,
            max_length=self.max_tokens,
            # do_sample=True,
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
        response_text = generated_text[len(prompt_text):].strip()

        return response_text, prompt_tokens, completion_tokens


class Mistral(HuggingFaceBaseModel):
    def __init__(
        self,
        model_name='mistralai/Mistral-7B-Instruct-v0.3',
        device=None,
        temperature=0.7,
        top_p=0.9,
        max_tokens=32768,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
       )