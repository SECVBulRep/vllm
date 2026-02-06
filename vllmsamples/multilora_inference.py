from huggingface_hub import snapshot_download
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

def initialize_engine(tokenizer_path: str) -> LLMEngine:
    engine_args = EngineArgs(
        model="Qwen/Qwen2.5-3B-Instruct",
        tokenizer=tokenizer_path,   # берем tokenizer из LoRA
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_cpu_loras=2,
        max_num_seqs=256,
    )
    return LLMEngine.from_engine_args(engine_args)

def main():
    lora_path = snapshot_download(
        repo_id="lleticiasilvaa/Qwen2.5-3B-Instruct-Spider-Text2SQL-SchemaLinking"
    )

    engine = initialize_engine(tokenizer_path=lora_path)

    sql_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=[" [markdown]", "\nor\n"]
    )

    test_prompts = [
        ("A robot may not injure a human being", SamplingParams(temperature=0.0, max_tokens=128), None),
        ("To be or not to be,", SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2, max_tokens=128), None),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n"
            "context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n"
            "question: Name the ICAO for lilongwe international airport [/user] [assistant]",
            sql_sampling,
            LoRARequest("sql-lora", 1, lora_path),
        ),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n"
            "context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n"
            "question: Name the ICAO for lilongwe international airport [/user] [assistant]",
            sql_sampling,
            LoRARequest("sql-lora2", 2, lora_path),
        ),
    ]

    request_id = 0
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sp, lr = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sp, lora_request=lr)
            request_id += 1

        for out in engine.step():
            if out.finished:
                print(out.outputs[0].text)
                print("-" * 50)

if __name__ == "__main__":
    main()
