# from modelscope import snapshot_download
from llama_cpp import Llama
import time
import subprocess
# import os
# from datetime import datetime

# from typing_extensions import runtime

def bench_model(
    model_dir,
    n_gpu,
    threads,
    n_batch,
    n_ubatch
):
    llm = Llama(
        model_path=model_dir,
        n_gpu_layers= n_gpu,
        n_threads=threads,
        n_batch=n_batch,             # prompt 预填充（prefill）批大小
        n_ubatch=n_ubatch,            # 微批大小
        flash_attn=False,         # 关闭 flash attention
        override_tensor="ffn_cpu_odd",  # 奇数层 FFN 强制放 CPU
        verbose=True,
        n_ctx=512
        )

    start = time.time()
    output = llm(
        prompt=prompt,
        max_tokens=100,
        echo=False
    )
    end = time.time()

    duration = end - start
    speed = 100 / duration

    print(f"Speed = {speed:.3f}, Time = {duration:.3f}")

def GetBest(model_dir, llama_bin_path, trails = 70):
    cmd = [
        "llama-optimus",
        "--llama-bin", llama_bin_path,
        "--model", model_dir,
        "--metric", "tg",
        "--trials", str(trails),
        "--repeat", "2"
    ]
    #cmd 是一个列表，列表里的每个元素就是在终端里敲命令时的一个单词。

    result = subprocess.run(
        args=cmd,
        capture_output=True, #捕捉输出与报错
        text=True #以文本形式返回
    )

    if result.returncode != 0:
        raise RuntimeError(f"运行失败: {result.stderr}")
    else:
        return result.returncode, result.stdout #返回状态码与标准输出

def main():
    model_dir = "E:\\local_AI\\Qwen3VL-8B\\Qwen3VL-4.0B-8b-BF16.gguf"
    llama_bin_path = "E:\\local_AI\\llama-b8204-bin-win-cuda-12.4-x64\\bin"
    prompt = "介绍一下人工智能"

    out = GetBest(model_dir, llama_bin_path)
    print(out)

    n_gpu = 34
    n_threads = 23
    n_batch = 1516
    n_ubatch = 6826
    

    
    bench = bench_model(model_dir=model_dir, n_gpu=n_gpu, threads=n_threads, n_batch=n_batch, n_ubatch=n_ubatch)
    



if __name__ == "__main__":
    main()
