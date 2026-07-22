from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embeddings(model_name: str = MODEL_NAME) -> HuggingFaceEmbeddings:
    # Load the HuggingFaceEmbeddings model with the specified model name and encoding parameters.
    return HuggingFaceEmbeddings(
        model_name=model_name, 
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64},
        )
