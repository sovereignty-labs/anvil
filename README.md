# anvil

**Run models on your own iron.**

Ollama opened local LLMs to a generation of people, then drifted into a blob store, a closed desktop app, and a cloud pivot — leaving everyone who just wanted a clean local tool without one. Anvil is for them: a transparent wrapper over llama.cpp that manages your models and your whole fleet, then gets out of the way. No blob store, no proprietary format, no cloud. Owned by you, not a funding round.

---

## Quickstart

```bash
# Install anvil
curl -fsSL https://raw.githubusercontent.com/sovereignty-labs/anvil/main/install.sh | sh

# Install llama-server as the user who will run the service
anvil runtime install

# Pull a model from HuggingFace
anvil pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S

# Run it directly if you are not using systemd
anvil load Qwen3.6-35B-A3B-UD-IQ3_S.gguf
```

That's it. Your model is serving on `http://localhost:11434/v1` - OpenAI-compatible, works with any client.

## Run Under systemd

The service user is the source of truth for runtime lookup. If you installed runtimes as your own user, run the service as that same user so anvil resolves `~/.local/share/anvil/runtimes`.

```bash
# Copy the example unit into place
sudo cp systemd/anvil.service /etc/systemd/system/anvil.service

# Edit these lines in /etc/systemd/system/anvil.service:
# User=CHANGE_ME
# Group=CHANGE_ME
```

Set `User=` and `Group=` to the user that ran `anvil runtime install`, then start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now anvil
```

If `anvil status` shows an `ERROR` row saying a runtime wasn't found in `/root/...`, the service is running as root but your runtimes are in your home. Set `User=` in the unit.

## Which model should I run?

Pick by your GPU's VRAM. These are tested starting points - anvil works with any GGUF.

| VRAM | Model | Pull Command | Speed (approx) |
|------|-------|-------------|----------------|
| **8 GB** | Qwen3.6-35B-A3B UD-IQ2_M | `anvil pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ2_M` | ~50 tok/s |
| **12 GB** | Qwen3.6-35B-A3B UD-IQ3_S | `anvil pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S` | ~80 tok/s |
| **16 GB** | Qwen3.6-35B-A3B UD-IQ4_NL | `anvil pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL` | ~60 tok/s |
| **24 GB** | Qwen3.6-35B-A3B UD-Q4_K_M | `anvil pull unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` | ~45 tok/s |
| **24 GB** | Qwen3.6-27B UD-Q4_K_XL | `anvil pull unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL` | ~25 tok/s |
| **48 GB+** | Qwen3.6-35B-A3B Q8_0 | `anvil pull unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0` | ~30 tok/s |

**MoE (35B-A3B) vs Dense (27B):** The MoE model only activates 3B parameters per token despite being 35B total - much faster at the same file size. The 27B dense model scores higher on coding benchmarks (SWE-bench 77.2% vs 73.4%) but is slower because all parameters fire every token. For chat and general use, the MoE is the better experience.

**Unsloth Dynamic (UD) quants** upscale important layers and are calibrated on real-world data. They outperform standard quants at the same bit rate. Prefer UD variants when available.

## What makes anvil different

**Plain GGUF files.** Your models live in a directory you can `ls`. No blob store, no hashed filenames, no proprietary format. Copy them, move them, use them with any tool.

```
/mnt/models/
├── Qwen3.6-35B-A3B-UD-IQ3_S.gguf
├── gemma-4-26B-A4B-Q3_K_XL.gguf
└── nemotron-nano-8b-Q8_0.gguf
```

**Transparent flags.** See exactly what anvil will pass to llama-server before it runs:

```bash
$ anvil load model.gguf --dry-run

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

**Smart defaults from GGUF metadata.** anvil reads the model's architecture, quantization type, context length, and chat template from the GGUF header. It picks the right GPU, caps context to fit in VRAM, and passes `--jinja` when a template is embedded. Override anything with `--` passthrough:

```bash
anvil load model.gguf -- --ctx-size 65536 --parallel 4 --cache-type-k q8_0
```

**Multi-model routing.** Load multiple models, and requests route by the `model` field in the OpenAI API request:

```bash
anvil load coding-model.gguf --gpu 0
anvil load chat-model.gguf --gpu 1
# Requests with "model": "coding-model" go to GPU 0
# Requests with "model": "chat-model" go to GPU 1
```

**Zero inference overhead.** anvil is a reverse proxy on the hot path. It routes requests to the right llama-server process and gets out of the way. No token parsing, no weight touching, no middleware tax.

## Federation

Manage models across your fleet from any terminal:

```bash
# Register remote nodes
anvil remote add gpu-host http://gpu-host.example.internal:11434
anvil remote add inference-host http://inference-host.example.internal:11434

# See everything
anvil status
# NODE   GPU              MODEL                    VRAM         ENDPOINT
# local  RTX 3090 24GB    Qwen3.6-35B-A3B-IQ3_S    20.8/24.0GB  http://localhost:11434
# gpu-host   RTX 3060 12GB    (idle)                    —            http://gpu-host:11434
# inference-host   RTX 3090 24GB    gemma-4-Q3_K_XL          18.2/24.0GB  http://inference-host:11434

# Load a model on a remote node
anvil load model.gguf --node gpu-host

# Copy a model to another node (over your LAN)
anvil cp model.gguf --to inference-host
```

## MCP Server

anvil exposes fleet management as MCP tools. Any MCP-capable client - Claude Desktop, Open WebUI, or custom agents - can manage inference programmatically:

```bash
anvil serve --mcp
```

Tools: `anvil_status`, `anvil_load`, `anvil_unload`, `anvil_models`, `anvil_pull`, `anvil_inspect`, `anvil_runtimes`, `anvil_rm`.

## Runtime Management

anvil manages the llama-server binary. No cmake, no build chain for most users:

```bash
# Install the latest llama-server (auto-detects GPU, builds with CUDA/ROCm if needed)
anvil runtime install

# Use a fork (TurboQuant, ik_llama, etc.)
anvil runtime build --repo https://github.com/ikawrakow/ik_llama.cpp --name ik-llama

# Switch runtimes per model
anvil load model.gguf --runtime ik-llama

# Point at a pre-existing binary
anvil runtime add custom /path/to/llama-server
```

## Hardware Profiles

Named flag sets for common configurations:

```bash
# Asymmetric TurboQuant - Q8 keys, turbo3 values
anvil load model.gguf --profile turboquant-asymmetric

# Maximum concurrent slots for agent swarms
anvil load model.gguf --profile agent-fleet

# Profiles stack
anvil load model.gguf --profile turboquant-asymmetric --profile agent-fleet
```

## Configuration

Config is optional. anvil works with zero config. The config file pre-defines what to load on startup:

```yaml
# /etc/anvil/config.yaml
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
anvil serve --config /etc/anvil/config.yaml
```

## CLI Reference

```bash
anvil pull <org/repo:quant>        # Download a GGUF from HuggingFace
anvil load <model.gguf>            # Load a model (auto-detects GPU, applies smart defaults)
anvil load <model> --dry-run       # Show what flags would be passed without launching
anvil load <model> --gpu 0         # Explicit GPU selection
anvil load <model> --swap          # Evict LRU model if VRAM is full
anvil unload <model>               # Stop a running model
anvil status                       # Fleet-wide status (all nodes, GPUs, models, VRAM)
anvil models                       # List downloaded GGUFs
anvil inspect <model.gguf>         # Read GGUF metadata
anvil search <query>               # Search HuggingFace for models
anvil run <model.gguf>             # Interactive chat with tok/s display
anvil cp <model> --to <node>       # Copy model to another node
anvil rm <model>                   # Delete a model
anvil remote add <name> <url>      # Register a remote node
anvil remote list                  # List remotes
anvil remote ping                  # Health check all remotes
anvil runtime install              # Install llama-server (auto-detects GPU)
anvil runtime build                # Build from source (for forks)
anvil runtime list                 # List installed runtimes
anvil runtime use <name>           # Switch active runtime
anvil serve                        # Run as a daemon
anvil version                      # Show version
```

## Why not Ollama?

Ollama promised to make llama.cpp accessible. anvil delivers on that promise without the baggage:

| | Ollama | anvil |
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

anvil exists because of [llama.cpp](https://github.com/ggml-org/llama.cpp), created by Georgi Gerganov and maintained by hundreds of contributors. anvil does not modify, fork, or vendor llama.cpp. It manages llama-server processes and gets out of the way. The real work happens in llama.cpp.

## License

MIT. No VC. No cloud. No blob store. Just models on your hardware.

Built by [Kit Porath](https://github.com/sovereignty-labs) at Sovereignty Labs.
