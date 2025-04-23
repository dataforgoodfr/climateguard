# import vllm
# # from vllm import LLM, SamplingParams
# import litserve as ls
# from litserve.specs.openai import ChatMessage

# class VLLMLlamaAPI(ls.LitAPI):
#     def setup(self, device):
#         self.llm = vllm.LLM(model="Qwen/Qwen2.5-0.5B-Instruct") #, quantization="fp8")

#     def predict(self, prompt):
#         print(prompt)
#         sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
#         outputs =  self.llm.generate(prompt, sampling_params)
#         yield from ChatMessage(role="assistant", content=outputs[0].outputs[0].text) 

#     # def encode_response(self, output):
#     #     yield 

# if __name__ == "__main__":
#     api = VLLMLlamaAPI()
#     server = ls.LitServer(api, spec=ls.OpenAISpec())
#     server.run(port=8000)

import litgpt
import litserve as ls

class LitOpenAI(ls.LitAPI):
    def setup(self, device):
        self.llm = litgpt.LLM.load("Qwen/Qwen2.5-0.5B-Instruct")

    def predict(self, prompt):
        yield from self.llm.generate(prompt, max_new_tokens=200, stream=True)

if __name__ == "__main__":
    api = LitOpenAI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
