import os
from vllm import SamplingParams
from vllm import LLM

class InferlessPythonModel:
    def initialize(self):
        self.template = """SYSTEM: You are a helpful assistant.
        USER: {}
        ASSISTANT: """
        self.llm = LLM(
          model="TheBloke/Mixtral-8x7B-v0.1-GPTQ",
          quantization="gptq",
          dtype="float16")
    
    def infer(self, inputs):
        prompts = [self.template.format(inputs["questions"])]
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=256,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"generated_result": result_output[0]}

    def finalize(self, args):
        pass
