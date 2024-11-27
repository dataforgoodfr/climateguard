import pandas as pd

from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm
import tiktoken

import os
from dotenv import load_dotenv

load_dotenv()
os.getenv('OPENAI_API_KEY')

client = AsyncOpenAI()
model = "gpt-4o-mini"

async def correct_text(
        text: str, tpm_limit: int = 2e6
    ):
    system_prompt = """
        L'utilisateur va fournir un extrait de 2 minutes d'une émission de télévision ou de radio.
        Le transcript ne contiendra pas de ponctuation et peut être de qualité médiocre (vocabulaire incorrect, mauvais découpage du texte). 
        La transcription est phonétique. Lorsque la phrase n'a pas de sens, reformule en prenant compte de la phonétique pour que le texte final soit en français correct.  
        
        Exemple :
        Transcription : 'elles doivent donc être renouvelés et sané col débuts en europe'
        Correction : 'Elles doivent donc être renouvelées, et ce n'est que le début en Europe.'
    """
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = len(encoding.encode(system_prompt))
    text_tokens = len(encoding.encode(text))
    total_tokens = prompt_tokens + text_tokens
    wait_time_s = 60 * total_tokens / tpm_limit
    await asyncio.sleep(wait_time_s)
    
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

async def run_correction(texts: list[str]):
    semaphore = asyncio.Semaphore(10)
    
    async def bounded_detect_claim(text):
        async with semaphore:
            return await correct_text(text)
    
    correction = await tqdm.gather(*[bounded_detect_claim(text) for text in texts])
    return correction