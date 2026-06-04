# translate-voice

Batch pipeline to transcribe Japanese game-audio `.ogg` files and translate the results
into other languages. Progress is tracked in `VoiceIndex.json`.

## Repository layout

```
client/               Python client CLI (uv project)
transcribe_server/    ASR server — kotoba-whisper-v2.1 (uv project)
translate_server/     Translation server — translategemma GGUF (uv project)
```

Each sub-directory is an independent `uv` project; run `uv sync` inside each one
before use.

---

## VoiceIndex.json

Stores transcriptions and translations keyed by audio file basename (without extension).

```json
{
	"voice_001": {
		"jp": "日本語テキスト",
		"en": "English text"
	}
}
```

Rules: sorted keys, `ensure_ascii=False`, tab-indented (`\t`), UTF-8.  
`"jp"` is always the Japanese source produced by the transcribe step.  
Other keys are BCP-47 language codes produced by the translate step.

---

## Client — `client/voice_client.py`

**Install:** `cd client && uv sync`

**Run:**
```bash
# Transcribe all voice_*.ogg files not yet in VoiceIndex
uv run python voice_client.py transcribe \
    --voice-dir /path/to/voices \
    --voice-index VoiceIndex.json

# Translate all "jp" entries that don't yet have the target language
uv run python voice_client.py translate \
    --voice-index VoiceIndex.json \
    --target-lang en \
    --glossary-file glossary.json \   # optional: {"勇者": "Hero"}
    --context "fantasy RPG game"      # optional
```

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIBE_SERVER_URL` | `http://localhost:8001` | Transcribe server base URL |
| `TRANSLATE_SERVER_URL` | `http://localhost:8002` | Translate server base URL |

**Behaviour:** VoiceIndex is written after each successful item so partial runs are
safe to resume. Errors are printed and skipped — the loop does not abort.

The file also contains the full BCP-47 → language-name `LANGUAGES` dict (moved here
from the translategemma Jinja2 chat_template).

---

## Transcribe server — `transcribe_server/server.py`

**Model:** `kotoba-tech/kotoba-whisper-v2.1` (Japanese ASR, distil-whisper based)  
**Extra features:** `punctuator=True` (uses `stable-ts` + `punctuators` for
post-processing punctuation)

**Install:**
```bash
cd transcribe_server && uv sync
```

**Run:**
```bash
uv run python server.py
```

**HTTP API:**
```
POST /transcribe
Content-Type: multipart/form-data
  audio: <binary .ogg or other audio>

200 OK
{ "result": { "text": "日本語テキスト" } }

GET /health → { "status": "ok" }
```

**GPU auto-detection** (`GPU_DEVICE` env var, default `auto`):
- `cuda` → `float16` + `attn_implementation="sdpa"`
- `mps` → `float16`
- `cpu` → `float32`

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `GPU_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `TRANSCRIBE_HOST` | `0.0.0.0` | Bind address |
| `TRANSCRIBE_PORT` | `8001` | Port |
| `CHUNK_LENGTH_S` | `30` | Audio chunk length for long files |

---

## Translate server — `translate_server/server.py`

**Model:** translategemma GGUF via `llama-cpp-python`
- `GPU_COUNT=1` (default) → `mradermacher/translategemma-4b-it-GGUF` Q8
- `GPU_COUNT≥2` → `mradermacher/translategemma-27b-it-GGUF` Q8 (fits 2× T4 = 32 GB)

Model is downloaded from HuggingFace Hub on first start via `hf_hub_download`.

**Install — CPU:**
```bash
cd translate_server && uv sync
```

**Install — CUDA (replace cu124 with your toolkit version):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
uv sync --no-install-package llama-cpp-python
```

**Install — Apple Metal:**
```bash
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python --no-cache-dir
uv sync --no-install-package llama-cpp-python
```

**Run:**
```bash
GPU_COUNT=1 uv run python server.py
GPU_COUNT=2 uv run python server.py   # 27b model, tensor_split=[0.5, 0.5]
```

**HTTP API:**
```
POST /translate
Content-Type: application/json
{
  "source_lang": "ja",
  "target_lang": "en",
  "text": "翻訳するテキスト",
  "custom_glossary": { "勇者": "Hero" },   // optional
  "context": "fantasy RPG"                // optional
}

200 OK
{ "result": { "text": "Text to translate" } }

GET /health → { "status": "ok" }
```

**Custom chat_template:** The default translategemma Jinja2 template is overridden in
`translate_server/server.py` (see `_CHAT_TEMPLATE`). Key changes from the original:
- `languages` dict removed from the template and replaced by the `LANGUAGES` Python
  constant looked up before rendering
- `context` block injected after the translator role preamble
- `glossary` block injected as a term list before the source text
- Rendered with `jinja2.Environment(trim_blocks=True)`

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `GPU_COUNT` | `1` | Selects 4b (1) or 27b (≥2) model |
| `N_GPU_LAYERS` | `-1` | Layers to offload; `-1` = all |
| `N_THREADS` | `8` | CPU inference threads |
| `TENSOR_SPLIT` | auto | Comma-separated GPU split ratios, e.g. `0.5,0.5` |
| `TRANSLATE_MODEL_REPO` | (by GPU_COUNT) | Override HuggingFace repo ID |
| `TRANSLATE_MODEL_FILE` | (by GPU_COUNT) | Override GGUF filename |
| `TRANSLATE_HOST` | `0.0.0.0` | Bind address |
| `TRANSLATE_PORT` | `8002` | Port |
