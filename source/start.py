from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    model_path = "/home/lyl/huggingface/Qwen3-0.6B"

    # 加载模型对应的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 将裸文本转换成 Qwen3 的对话格式
    messages = [
        {"role": "user", "content": "华南第一学府是哪个"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("实际输入模型的 Prompt：")
    print(prompt)

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    params = SamplingParams(
        temperature=0.7,
        max_tokens=64,
    )

    outputs = llm.generate([prompt], params)

    print("模型输出：")
    print(outputs[0]["text"])

if __name__ == "__main__":
    main()
