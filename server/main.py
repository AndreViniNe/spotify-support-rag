from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langchain_core.prompts import PromptTemplate

from services.generation import load_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = load_llm()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.get("/invoke_chain")
def invoke_chain(text: str, request: Request):
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | request.app.state.llm

    question = "How would you explain LangChain to a kid?"

    return chain.invoke({"question": question})
