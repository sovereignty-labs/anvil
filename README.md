# nollama

**The model runner Ollama should have been.**

One Go binary. Plain GGUFs. Transparent [llama-server](https://github.com/ggml-org/llama.cpp) under the hood. Federation across your fleet.

nollama manages llama-server processes so you don't have to — without inserting itself into your inference path. Smart defaults from GGUF metadata. Full access to every llama.cpp flag. Zero overhead on the hot path.

No blob store. No proprietary format. No cloud. MIT licensed.

---

## Quickstart

```bash
# Install nollama
curl -fsSL https://raw.githubusercontent.com/sovereignty-labs/nollama/main/install.sh | sh

# Install llama-server (auto-detects your GPU and builds with CUDA/ROCm/Metal)
nollama runtime install

# Pull a model from HuggingFace
nollama pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S

# Run it
nollama load Qwen3.6-35B-A3B-UD-IQ3_S.gguf
```

That's it. Your model is serving on `http://localhost:11434/v1` - OpenAI-compatible, works with any client.

## Which model should I run?

Pick by your GPU's VRAM. These are tested starting points - nollama works with any GGUF.

| VRAM | Model | Pull Command | Speed (approx) |
|------|-------|-------------|----------------|
| **8 GB** | Qwen3.6-35B-A3B UD-IQ2_M | `nollama pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ2_M` | ~50 tok/s |
| **12 GB** | Qwen3.6-35B-A3B UD-IQ3_S | `nollama pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S` | ~80 tok/s |
| **16 GB** | Qwen3.6-35B-A3B UD-IQ4_NL | `nollama pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL` | ~60 tok/s |
| **24 GB** | Qwen3.6-35B-A3B UD-Q4_K_M | `nollama pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` | ~45 tok/s |
| **24 GB** | Qwen3.6-27B UD-Q4_K_XL | `nollama pull unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL` | ~25 tok/s |
| **48 GB+** | Qwen3.6-35B-A3B Q8_0 | `nollama pull unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0` | ~30 tok/s |

**MoE (35B-A3B) vs Dense (27B):** The MoE model only activates 3B parameters per token despite being 35B total - much faster at the same file size. The 27B dense model scores higher on coding benchmarks (SWE-bench 77.2% vs 73.4%) but is slower because all parameters fire every token. For chat and general use, the MoE is the better experience.

**Unsloth Dynamic (UD) quants** upscale important layers and are calibrated on real-world data. They outperform standard quants at the same bit rate. Prefer UD variants when available.

## What makes nollama different

**Plain GGUF files.** Your models live in a directory you can `ls`. No blob store, no hashed filenames, no proprietary format. Copy them, move them, use them with any tool.

```
/mnt/models/
├── Qwen3.6-35B-A3B-UD-IQ3_S.gguf
├── gemma-4-26B-A4B-Q3_K_XL.gguf
└── nemotron-nano-8b-Q8_0.gguf
```

**Transparent flags.** See exactly what nollama will pass to llama-server before it runs:

```bash
$ nollama load model.gguf --dry-run

  Model:    gemma-4-26B-A4B-Q3_K_XL.gguf (12.1 GB)
  Target:   GPU 0 (RTX 3090, 24GB - 18.3 GB estimated)

  Would run:
    llama-server \
      --model /mnt/models/gemma-4-26B-A4B-Q3_K_XL.gguf \
      --n-gpu-layers 99 \
      --ctx-size 131072 \
      --flash-attn on \
      --jinja \
      --host 0.0.0.0 \
      --port 11434
```

**Smart defaults from GGUF metadata.** nollama reads the model's architecture, quantization type, context length, and chat template from the GGUF header. It picks the right GPU, caps context to fit in VRAM, and passes `--jinja` when a template is embedded. Override anything with `--` passthrough:

```bash
nollama load model.gguf -- --ctx-size 65536 --parallel 4 --cache-type-k q8_0
```

**Multi-model routing.** Load multiple models, and requests route by the `model` field in the OpenAI API request:

```bash
nollama load coding-model.gguf --gpu 0
nollama load chat-model.gguf --gpu 1
# Requests with "model": "coding-model" go to GPU 0
# Requests with "model": "chat-model" go to GPU 1
```

**Zero inference overhead.** nollama is a reverse proxy on the hot path. It routes requests to the right llama-server process and gets out of the way. No token parsing, no weight touching, no middleware tax.

## Federation

Manage models across your fleet from any terminal:

```bash
# Register remote nodes
nollama remote add gpu-host http://gpu-host.example.internal:11434
nollama remote add inference-host http://inference-host.example.internal:11434

# See everything
nollama status
# NODE   GPU              MODEL                    VRAM         ENDPOINT
# local  RTX 3090 24GB    Qwen3.6-35B-A3B-IQ3_S    20.8/24.0GB  http://localhost:11434
# gpu-host   RTX 3060 12GB    (idle)                    —            http://gpu-host:11434
# inference-host   RTX 3090 24GB    gemma-4-Q3_K_XL          18.2/24.0GB  http://inference-host:11434

# Load a model on a remote node
nollama load model.gguf --node gpu-host

# Copy a model to another node (over your LAN)
nollama cp model.gguf --to inference-host
```

## MCP Server

nollama exposes fleet management as MCP tools. Any MCP-capable client - Claude Desktop, Open WebUI, or custom agents - can manage inference programmatically:

```bash
nollama serve --mcp
```

Tools: `nollama_status`, `nollama_load`, `nollama_unload`, `nollama_models`, `nollama_pull`, `nollama_inspect`, `nollama_runtimes`, `nollama_rm`.

## Runtime Management

nollama manages the llama-server binary. No cmake, no build chain for most users:

```bash
# Install the latest llama-server (auto-detects GPU, builds with CUDA/ROCm if needed)
nollama runtime install

# Use a fork (TurboQuant, ik_llama, etc.)
nollama runtime build --repo https://github.com/ikawrakow/ik_llama.cpp --name ik-llama

# Switch runtimes per model
nollama load model.gguf --runtime ik-llama

# Point at a pre-existing binary
nollama runtime add custom /path/to/llama-server
```

## Hardware Profiles

Named flag sets for common configurations:

```bash
# Asymmetric TurboQuant - Q8 keys, turbo3 values
nollama load model.gguf --profile turboquant-asymmetric

# Maximum concurrent slots for agent swarms
nollama load model.gguf --profile agent-fleet

# Profiles stack
nollama load model.gguf --profile turboquant-asymmetric --profile agent-fleet
```

## Configuration

Config is optional. nollama works with zero config. The config file pre-defines what to load on startup:

```yaml
# /etc/nollama/config.yaml
model_dir: /mnt/models
bind: 0.0.0.0:11434

autoload:
  - model: Qwen3.6-35B-A3B-UD-IQ3_S.gguf
    gpu: 0
    flags:
      ctx-size: 131072
      parallel: 4
      flash-attn: "on"

aliases:
  advisor: Qwen3.6-35B-A3B-UD-IQ3_S

mcp:
  enabled: true
  transport: sse
  bind: 0.0.0.0:11436
```

```bash
nollama serve --config /etc/nollama/config.yaml
```

## CLI Reference

```bash
nollama pull <org/repo:quant>        # Download a GGUF from HuggingFace
nollama load <model.gguf>            # Load a model (auto-detects GPU, applies smart defaults)
nollama load <model> --dry-run       # Show what flags would be passed without launching
nollama load <model> --gpu 0         # Explicit GPU selection
nollama load <model> --swap          # Evict LRU model if VRAM is full
nollama unload <model>               # Stop a running model
nollama status                       # Fleet-wide status (all nodes, GPUs, models, VRAM)
nollama models                       # List downloaded GGUFs
nollama inspect <model.gguf>         # Read GGUF metadata
nollama search <query>               # Search HuggingFace for models
nollama run <model.gguf>             # Interactive chat with tok/s display
nollama cp <model> --to <node>       # Copy model to another node
nollama rm <model>                   # Delete a model
nollama remote add <name> <url>      # Register a remote node
nollama remote list                  # List remotes
nollama remote ping                  # Health check all remotes
nollama runtime install              # Install llama-server (auto-detects GPU)
nollama runtime build                # Build from source (for forks)
nollama runtime list                 # List installed runtimes
nollama runtime use <name>           # Switch active runtime
nollama serve                        # Run as a daemon
nollama version                      # Show version
```

## Why not Ollama?

Ollama promised to make llama.cpp accessible. nollama delivers on that promise without the baggage:

| | Ollama | nollama |
|---|---|---|
| Model storage | Hashed blob store | Plain GGUF files |
| Configuration | Proprietary Modelfile | GGUF metadata + CLI flags |
| Chat templates | Hardcoded list | `--jinja` (reads GGUF-embedded templates) |
| Quantization | 5 types | All (inherited from llama.cpp) |
| Inference overhead | 30-50% slower | 0% (pure proxy) |
| Multi-node fleet | No | Federation + MCP |
| Flag visibility | Hidden | `--dry-run` shows everything |
| Fork support | No | Per-model runtime selection |
| Cloud inference | Pivoting to cloud | Never |
| llama.cpp credit | Begrudging footnote | Front and center |

## Attribution

nollama exists because of [llama.cpp](https://github.com/ggml-org/llama.cpp), created by Georgi Gerganov and maintained by hundreds of contributors. nollama does not modify, fork, or vendor llama.cpp. It manages llama-server processes and gets out of the way. The real work happens in llama.cpp.

## License

MIT. No VC. No cloud. No blob store. Just models on your hardware.

Built by [Kit Porath](https://github.com/sovereignty-labs) at Sovereignty Labs.
