from fastapi import FastAPI
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from accelerate import Accelerator

app = FastAPI()
device = Accelerator().device

# transformers 5.x dropped the dedicated "summarization" pipeline; summarization
# is now done by prompting a causal LM through "text-generation".
summarizer = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct", device=device)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.get("/summarize/{text}")
def summarize(text: str):
    messages = [
        {"role": "user", "content": f"Summarize the following text in 1-2 sentences:\n\n{text}"}
    ]
    result = summarizer(messages, max_new_tokens=100, do_sample=False)
    reply = result[0]["generated_text"][-1]["content"]
    return {"summary": reply.strip()}