"""
This file contains the data models for the documents and chunks.

"""


from pydantic import BaseModel, Field
from typing import List, Literal, Optional




class Document(BaseModel):
    doc_id: str = Field(..., description="Internal UUID or slug")
    arxiv_id: str = Field(..., description="e.g. arxiv:1406.1078")
    title: str
    document_summary: str = Field(
        default="",
        description="Short paragraph condensed from per-section sentences; no repetition of terms, one consistent summary.",)
    document_summary_embedding: Optional[List[float]] = Field(
        None, description="Embedding of document_summary (OpenAI text-embedding-3-small), set in parse_document."
    )

    

class RAGChunk(BaseModel):
    # foreign key
    doc_id: str = Field(..., description="References documents.doc_id")
    
    content: str = Field(..., description="Text content of the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding (filled by embed_chunks)")
    section: Optional[str] = None
    subsection: Optional[str] = None

    # dataset: List[str] = Field(default_factory=list)
    # metrics: List[str] = Field(default_factory=list)
