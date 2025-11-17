from langchain_ollama import ChatOllama
from typing import List, Optional
from pydantic import BaseModel, Field


hostname = "llm.cosy.bio"
api_url = f"https://{hostname}"

class CosyChatOllama(ChatOllama):
    def __init__(self, model: str = "medllama2:latest", temperature: float = 0.0,
                 format: str = "json", num_predict=256, num_ctx=2048) -> object:
        super().__init__(base_url=api_url, model=model, temperature=temperature,
                         format=format, num_predict=num_predict, num_ctx=num_ctx)


class SchemaNode(BaseModel):
    variable: Optional[str] = Field(None, description="The variable that has been mapped.")
    ontology_id: Optional[str] = Field(None, description="The ontology identifier.")
    ontology_name: Optional[str] = Field(None, description="The name of the ontology node.")
    confidence: Optional[float] = Field(None, description="The confidence (semantic similarity) score.")

