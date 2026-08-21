


**Qwen Modeli (Görsel ve Metin Analizi)**

```bash
sudo docker run --runtime nvidia --gpus '"device=0,1"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /tmp:/tmp \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --mm-cache-preprocessor-workers 0
  --allowed-local-media-path /tmp
```

**Llama Modeli (Akıl Yürütme ve Karar)**
```bash
sudo docker run --runtime nvidia --gpus '"device=2,3,4,5"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8001:8000 \
  -e HUGGING_FACE_HUB_TOKEN="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" \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```






#bilisimvadisi2026
