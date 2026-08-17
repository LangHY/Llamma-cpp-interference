from modelscope import snapshot_download
from llama_cpp import Llama
import time
import subprocess
import os
from datetime import datetime

from typing_extensions import runtime

def bench_model(n_gpu, prompt, model_dir):
    llm = Llama(
        model_path=model_dir,
        n_gpu_layers= n_gpu,
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

    # 保存结果到 interference 目录
    result_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"result_{timestamp}.txt")

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {model_dir}\n")
        f.write(f"返回码: {out[0]}\n")
        f.write(f"测试结果:\n{out[1]}\n")

    print(f"结果已保存至: {result_file}")

    # 关机
    print("10秒后关机...")
    time.sleep(10)
    subprocess.run(["shutdown", "/s", "/t", "0"])



if __name__ == "__main__":
    main()
