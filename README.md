#bilisimvadisi2026
Qwen Modeli (Görsel ve Metin Analizi)

bash```
sudo docker run --runtime nvidia --gpus '"device=0,1"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --enable-chunked-prefill
```
