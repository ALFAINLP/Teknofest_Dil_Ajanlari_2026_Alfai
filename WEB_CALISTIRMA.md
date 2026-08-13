# ALFAI web arayüzünü çalıştırma

Model sunucuları çalışır durumda olmalıdır:

- Qwen3-VL: `http://127.0.0.1:8000`
- Llama 3.1: `http://127.0.0.1:8001`

## İlk kurulum

Proje kökünde:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-web.txt
npm install
```

## Her çalıştırmada

Backend terminali:

```bash
cd ~/Desktop/Alfai
source .venv/bin/activate
python -m uvicorn web_api:app --host 127.0.0.1 --port 8080
```

Frontend terminali:

```bash
cd ~/Desktop/Alfai
npm run dev -- --host 127.0.0.1
```

Tarayıcı: `http://127.0.0.1:5173`

