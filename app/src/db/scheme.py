from pydantic import BaseModel


class Paper(BaseModel):
    id: int
    title: str
    authors: str
    abstract: str
    doi: str
    arxiv_id: str | None