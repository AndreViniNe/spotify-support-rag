from accelerate import Accelerator
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = Accelerator().device

def load_llm(model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct") -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return HuggingFacePipeline(pipeline=pipe)
