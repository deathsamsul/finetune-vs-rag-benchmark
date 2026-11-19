import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

from langchain_huggingface import HuggingFacePipeline
from ragas.llms import LangchainLLMWrapper


def build_local_judge(model_name: str):

    print(f"Loading local judge: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)


#Quantization reduces the memory required to store model weights by representing them with fewer bits.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        return_full_text=False,
    )

    local_llm = HuggingFacePipeline(
        pipeline=generator
    )

    return LangchainLLMWrapper(local_llm)
