import asyncio
import json
from typing import Literal

from openai import AsyncOpenAI
from openai.lib._parsing._completions import type_to_response_format_param

from climateguard.legacy.models import Article, Claims, Transcript

client = AsyncOpenAI()


async def adetect_claims(
    input: Article | Transcript, language: Literal["French", "English", "Latvian"]
) -> tuple[Claims, int]:
    # Prompt
    language_instruction_prompt = (
        f"\nThe article is written in {language}. Your analysis should be in English."
        if language != "English"
        else ""
    )
    system_prompt = f"""You are an expert in misinformation on environmental topics, a specialist in climate science, and have extensive knowledge about the IPCC reports.
Your role is to thoroughly analyze the provided news article to identify any misleading environmental claims or opinions that may require fact-checking.

Read the article thoroughly to fully understand its content and context. Focus exclusively on environmental issues such as climate change, ecological transition, energy, biodiversity, pollution, pesticides, and natural resources like water and minerals, and exclude any social or economic aspects.
{language_instruction_prompt}

**Examples of Claims to Flag**:

- "Climate change might be happening, but it's not urgent; we still have plenty of time to adapt."
- "Electric vehicles still rely on fossil fuels for electricity, so they're not really helping the environment."
- "Planting more trees will easily offset all of our carbon emissions."
- "We can solve climate change just by using more technology without needing to change our lifestyles."
- "Plastic recycling is efficient enough to manage pollution, so there's no need to reduce plastic use."
- "Renewable energy alone can’t meet global energy demands reliably, so we still need coal and oil."
- "Increased carbon dioxide is good for plants and agriculture, so climate change could actually benefit us."
- "The Earth naturally balances itself, so human impact isn't significant in the long run."
- "It's more important to focus on economic growth than to worry about environmental regulations."
- "Natural disasters are part of the Earth's natural cycles, not related to climate change.\""""
    user_prompt = f"# {input.title}\n\n{input.content}"

    # LLM call
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=type_to_response_format_param(Claims),
        temperature=0,
        # We force early stopping, otherwise the model still outputs claims that we probably don't care about (sometimes)
        stop='{"article_needs_fact_checking":false',
        seed=42,
    )

    # Parse the output completion
    # The content is empty if we stopped because of the stop token '{"article_needs_fact_checking":false'
    if response.choices[0].message.content == "":
        claims = Claims(article_needs_fact_checking=False, claims=[])
    else:
        claims = Claims.model_validate(json.loads(response.choices[0].message.content))

    n_tokens = response.usage.total_tokens

    return claims, n_tokens


def detect_claims(
    input: Article | Transcript, language: Literal["French", "English", "Latvian"]
) -> tuple[Claims, int]:
    # This is a synchronous version for convenience, but it won't work if you call this function from a Jupyter notebook
    return asyncio.run(adetect_claims(input, language))
