# LM Studio Vision – PMS Graph Analyzer

This project lets you classify **PMS graphs** as **Normal** or **Abnormal** using a **vision-capable LM Studio model** (e.g., LLaVA, Llama-3.2 Vision, Phi-3.5 Vision).

You can run it in two ways:

* **Proxy mode**: keep your old client shape (`images` / `allImages`) and forward requests to LM Studio in OpenAI multimodal format.
* **Direct mode**: send images straight to LM Studio in the OpenAI-style multimodal format.

---

## Files

* **`lmstudio_vision_proxy.py`**
  A FastAPI proxy that accepts old-style payloads (`images` / `allImages`) and translates them into OpenAI multimodal format for LM Studio.

* **`analyze_pms_withai_lmstudio.py`**
  A direct Python client that loads graphs from a folder, classifies them via LM Studio, and writes results to Excel.

---

## Setup

Install required packages:

```bash
pip install fastapi uvicorn requests pandas openpyxl
```

Make sure **LM Studio** is running with a **vision-capable model loaded** and the **local server enabled** (default: `http://localhost:1234`).

---

## Usage

### Option 1 – Proxy mode (keep existing client payloads)

1. Start LM Studio with a vision model.

2. Run the proxy:

   ```bash
   uvicorn lmstudio_vision_proxy:app --host 0.0.0.0 --port 8000
   ```

3. Point your existing client to:

   ```
   http://localhost:8000/v1/chat/completions
   ```

   Your old payload shape (`images` / `allImages`) will work unchanged.

---

### Option 2 – Direct mode (simpler client)

Run the analyzer script to scan a directory of graphs and classify them:

```bash
python analyze_pms_withai_lmstudio.py
```

Environment variables you can override:

* `DIR` → Input directory of graphs (default: PMS folder path in OneDrive)
* `OUT_XLSX` → Output Excel file path (`PMS_Results.xlsx` by default)
* `LMSTUDIO_API_BASE` → LM Studio server base URL (default: `http://localhost:1234`)
* `LMSTUDIO_MODEL` → Model name/id (`local` works for the loaded model)
* `LMSTUDIO_API_KEY` → API key (if enabled in LM Studio)

The output Excel will contain:

| Graph Name | Result   |
| ---------- | -------- |
| Graph1     | Normal   |
| Graph2     | Abnormal |

---

## Notes

* Use a **vision-enabled LM Studio model**. Non-vision models will return errors or ignore images.
* For strict classification, the script sets **temperature = 0** (deterministic, one-word answers).
* First inference may take longer (model warm-up); subsequent runs are faster.
* Proxy ensures backwards compatibility; direct client is leaner and preferred if you can adjust your client.
