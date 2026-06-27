from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, conlist, confloat, constr
import os
import asyncio
from ollama import Client, ResponseError
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()



# Authentication details
protocol = "https"
hostname = "dev.chat.cosy.bio"
host = f"{protocol}://{hostname}"
api_url = f"{host}/ollama"
api_key =  "sk-02e2176c3dbc4f629fc285238a739d8d"
headers = {"Authorization": "Bearer " + api_key}

import os
import asyncio
from ollama import Client, ResponseError
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaWrapper:
    def __init__(self, headers=None, timeout=100):
        host = os.getenv("OLLAMA_HOST")
        api_key = os.getenv("OLLAMA_API_KEY")
        if not host or not api_key:
            raise ValueError("Missing OLLAMA_HOST or OLLAMA_API_KEY in environment")

        # Use AsyncClient for better compatibility if available,
        # but we'll stick to your logic with a fix for the execution.
        self.client = Client(host=host,
                             headers={"Authorization": f"Bearer {api_key}"})
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(self, model, messages):
        """Standard def that handles the event loop safely"""

        # We define the logic without nested asyncio.run if possible,
        # but for a quick fix that works in both .py and Notebooks:
        try:
            # Using the client directly. If your client is synchronous,
            # you don't actually need asyncio.run at all!
            return self.client.chat(model=model, messages=messages)

        except Exception as e:
            if "401" in str(e):
                raise Exception("Authentication failed (check API key)")
            print(f"Ollama error: {e}")
            raise

'''
class CosyChatOllama(ChatOllama):
    def __init__(self, model: str = "medllama2:latest", temperature: float = 0.0,
                 format: str = "json", num_predict=256, num_ctx=2048) -> None:
        super().__init__(base_url=api_url, model=model, temperature=temperature,
                         format=format, num_predict=num_predict, num_ctx=num_ctx, client_kwargs={'headers': headers})

def get_llm(ollama_model: str = "gpt-oss:20b") -> ChatOllama:
    return ChatOllama(
            base_url=api_url,
            model=ollama_model,
            temperature=0.0,
            seed=28,
            num_ctx=25000,
            num_predict=-1,
            top_k=100,
            top_p=0.95,
            format="json",
            client_kwargs={'headers': headers})

'''

class Candidate(BaseModel):
    rank: Optional[int] = Field(None, description="Rank of this candidate (1 = best match).")
    AI_code: Optional[str] = Field(None, description="The ontology identifier.")
    AI_name: Optional[str] = Field(None, description="The name of the ontology node.")
    confidence: Optional[float] = Field(None, description="The confidence (semantic similarity) score.")
    CUI: Optional[str] = Field(None, description="The UMLS concept identifier related to the ontology concept.")

class CUICandidate(BaseModel):
    rank: Optional[int] = Field(None, description="Rank of this candidate (1 = best match).")
    CUI: Optional[str] = Field(None, description="The UMLS concept identifier.")
    CUI_label: Optional[str] = Field(None, description="The label of the UMLS concept.")
    representative_names: Optional[List[str]] = Field(None, description="The list of names that share the same meaning.")
    confidence: Optional[float] = Field(None, description="The confidence (semantic similarity) score.")
    reasoning: Optional[str] = Field(None, description="The explaination of the LLM choice")


class RankedCandidates(BaseModel):
    variable: Optional[str] = Field(None, description="The variable that has been mapped.")
    candidates: Optional[List[Candidate]] = Field(None, description="Ranked list of up to 5 ontology candidates.")

class RankedCUICandidates(BaseModel):
    variable: Optional[str] = Field(None, description="The variable that has been mapped.")
    candidates: Optional[List[CUICandidate]] = Field(None, description="Ranked list of up to 3 ontology candidates.")

class RagAudit(BaseModel):
    status: Literal["VALID", "INVALID", "Needs_Review"] = Field(..., description="Verdict on the RAG model's primary pick.")
    explanation: str = Field(..., description="Why the RAG pick was accepted or rejected.")

class SupervisorOutput(BaseModel):
    rag_review: RagAudit
    supervised_ranking: RankedCUICandidates



