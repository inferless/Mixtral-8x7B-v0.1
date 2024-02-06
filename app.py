from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

class InferlessPythonModel:
    def initialize(self):
        
        model_name_or_path = "TheBloke/Mixtral-8x7B-v0.1-GPTQ"
        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename="model",
                use_safetensors=True,
                trust_remote_code=False,
                device="cuda:0",
                use_triton=False,
                disable_exllama=True,
                disable_exllamav2=True,
                quantize_config=None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=False)

    def infer(self, inputs):
        prompts = inputs["prompt"]
        input_ids = self.tokenizer(prompts, return_tensors='pt').input_ids.cuda()
        output = self.model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
        text = self.tokenizer.decode(output[0])

        return {'generated_result': output.shape[1]}

    def finalize(self):
        pass