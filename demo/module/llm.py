from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    #BitsAndBytesConfig,
    pipeline,
    GenerationConfig,
)
import torch
import traceback
import warnings

warnings.filterwarnings("ignore")

MODEL_NAME = "Qwen/Qwen3-4B"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4",
# )

gen_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)


class QwenChat:
    def __init__(self, model_name=MODEL_NAME, cache_dir="./"):
        print("🔧 LLM 모델 로딩 중...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            #quantization_config=quantization_config,
            cache_dir=cache_dir,
        )
        self.model.to("cpu")

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
        )

        print("✅ LLM 준비 완료")

    def get_prompt(self, query: str):
        full_prompt = f"""
You are an English conversation teacher. Please lead the conversation naturally.

<question>
{query}
<answer>
"""

        messages = [
            {"role": "system", "content": "You're an English conversation teacher"},
            {"role": "user", "content": full_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        return full_prompt, prompt

    def answer_extraction(self, response: str) -> str:
        try:
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant", 1)[1]

            if "</think>" in response:
                think_end = response.find("</think>") + len("</think>")
                response = response[think_end:]

            return response.strip()

        except Exception:
            print("Error in answer_extraction")
            print(traceback.format_exc())
            return response

    def ask(self, query: str) -> str:
        try:
            _, prompt = self.get_prompt(query)
            raw = self.pipe(prompt, generation_config=gen_config)
            generated_text = raw[0]["generated_text"]
            answer = self.answer_extraction(generated_text)
            return answer

        except Exception:
            print("Error in ask()")
            print(traceback.format_exc())
            return "오류가 발생했습니다."
