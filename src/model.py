from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

class TinyLlamaModel:
    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Carregando modelo {self.model_name} em CPU float32...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        self.model.to("cpu")
        print(f"Modelo {self.model_name} carregado com sucesso em CPU!")

    def generate_response(self, query: str, context: List[str], max_length: int = 512) -> str:
        if self.model is None or self.tokenizer is None:
            self.load_model()

        formatted_context = "\n".join(context)
        prompt = f"""<|system|>
Você é um assistente que responde perguntas com base no conteúdo do(s) documento(s) fornecido(s).
Use apenas as informações fornecidas no contexto para responder à pergunta.
Se a informação não estiver no contexto, diga que ela não está disponível no(s) documento(s).

Contexto:
{formatted_context}
<|user|>
{query}
<|assistant|>
"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response

