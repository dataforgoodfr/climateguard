import sys

sys.path.append("..")

from statistics import median

import tiktoken
from datasets import load_dataset
from ecologits import EcoLogits
from openai import OpenAI
from tqdm import tqdm

from app.prompts.prompts import PROMPTS

# Initialize EcoLogits
EcoLogits.init(providers=["openai"])

with open("../secrets/openai_key") as f:
    key = f.read()
client = OpenAI(api_key=key)

encoding = tiktoken.encoding_for_model('gpt-4o-mini')
dataset = load_dataset("DataForGood/climateguard")
chat_prompt = PROMPTS["0.0.1"]

energy_min = []
energy_max = []
gwp_min = []
gwp_max = []
tokens = []

for record in tqdm(dataset["train"]):
    response = client.chat.completions.create( 
        model="gpt-4o-mini",
        messages=[
                {"role": "user", "content": chat_prompt.prompt + record["plaintext"]},
            ]
    )
    n_tokens = len(encoding.encode(chat_prompt.prompt + record["plaintext"]))

    tokens.append(n_tokens)
    energy_min.append(response.impacts.energy.value.min)
    energy_max.append(response.impacts.energy.value.max)
    gwp_min.append(response.impacts.gwp.value.min)
    gwp_max.append(response.impacts.gwp.value.max)
    print(response.impacts.energy)
    print(response.impacts.gwp)
    break

print("median tokens: ", median(tokens))
print("median energy_min: ", median(energy_min))
print("median energy_max: ", median(energy_max))
print("median gwp_min: ", median(gwp_min))
print("median gwp_max: ", median(gwp_max))
