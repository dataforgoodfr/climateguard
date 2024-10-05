from pydantic import BaseModel, Field
from pydantic.type_adapter import TypeAdapter


class Claim(BaseModel):
    quote: str = Field(
        description="The exact quote from the article corresponding to the claim"
    )
    claim: str = Field(description="The claim that potentially needs verification")
    context: str = Field(
        description="A reformulation of the context in which this claim was made (maximum 1 paragraph)"
    )
    analysis: str = Field(
        description="Analysis from your expert's perspective on the potential misinformation of this claim based on the context."
    )


class Claims(BaseModel):
    article_needs_fact_checking: bool = Field(
        description="Whether the article is misleading and should be fact-checked"
    )
    claims: list[Claim] = Field(
        description="A list of all misleading environmental claims that should be fact-checked"
    )


class Article(BaseModel):
    title: str
    content: str
    url: str
    date: str
    topic: str
    source: str

    @staticmethod
    def from_json(filepath: str) -> "Article":
        with open(filepath) as f:
            return TypeAdapter(list[Article]).validate_json(f.read())
