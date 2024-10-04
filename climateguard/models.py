
from pydantic import BaseModel


class Claim(BaseModel):
    claim: str
    context: str
    analysis: str
    disinformation_score: str
    disinformation_category: str

class Claims(BaseModel):
    claims: list[Claim]


class Article(BaseModel):
    title: str
    content: str
    url: str
    date: str
    topic: str
    source: str
    