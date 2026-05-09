# nollama

offical repo for the nollama project

# nollama

**The model runner Ollama should have been.**

nollama is a transparent wrapper for [llama-server](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov. It manages llama-server processes with smart defaults, keeps your models as plain GGUF files, and gives you a single pane of glass across every GPU in your fleet. Zero inference overhead. Zero opinions about your models. Zero cloud.

```bash
nollama runtime install                              # download llama-server
nollama pull unsloth/gemma-4-26B-A4B-it-GGUF:Q3_K_XL # grab a model from HuggingFace
nollama load gemma-4-26B-A4B-it-Q3_K_XL.gguf         # running. done.
```

---

## Why

Ollama made local LLMs accessible. Then it kept going — blob stores, proprietary Modelfiles, a hardcoded template list that ignores the Jinja templates already embedded in your GGUFs, a forked backend that benchmarks 30-50% slower than upstream, a cloud pivot, and a closed-source desktop app. All on VC money.

The local inference community doesn't need a platform. It needs a tool.

nollama is that tool. It wraps llama-server the way it should have been wrapped from the start: transparently, with every flag visible, every file where you left it, and nothing between you and full llama.cpp performance.

---

## What nollama does

**Manages llama-server processes.** Load and unload models with one command. nollama reads the GGUF header, detects your hardware, computes smart defaults, and spawns llama-server with the right flags. Use `--dry-run` to see exactly what it would pass — nothing is hidden.

**Keeps models as plain files.** Your GGUFs live in a directory. You can `ls` them, `cp` them, `rsync` them to another machine, and open them with any GGUF-compatible tool. No blob store. No hashed filenames. No lock-in.

**Pulls from HuggingFace directly.** `nollama pull org/repo:quant` downloads the GGUF and saves it with its original filename. No proprietary registry. No intermediary format.

**Reads the GGUF, doesn't replace it.** Chat templates, context length, architecture metadata — it's all embedded in the file. nollama passes `--jinja` and lets llama-server handle it. No hardcoded template list. No Modelfile. No Go-template-to-Jinja translation layer.

**Supports every quantization llama.cpp supports.** Q2 through Q8, all IQ formats, BF16, F16, F32. If llama-server can load it, nollama can serve it.

**Routes requests across loaded models.** One OpenAI-compatible endpoint per node. Multiple models loaded simultaneously. Requests route by model name. Pure HTTP reverse proxy on the hot path — nollama adds zero inference overhead.

**Manages the llama-server binary for you.** `nollama runtime install` downloads pre-built releases. `nollama runtime build` compiles from source — including forks like [TheTom's TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) or [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). Multiple runtimes coexist. Per-model runtime selection with `--runtime`.

**Federates across your fleet.** Register remote nodes. Run `nollama status` from any terminal and see every GPU, every model, every endpoint across every machine. Swap models on remote nodes without SSH. No auto-routing, no heuristics — you're the orchestrator.

**Exposes fleet management via MCP.** AI agents can check fleet status, load models, pull GGUFs, and manage inference resources programmatically. Control plane only — inference goes through the standard OpenAI-compatible endpoint.

---

## Quick Start

### Install nollama

```bash
# From source (requires Go 1.22+)
go install github.com/hirdforge/nollama/cmd/nollama@latest

# Or download a release binary
curl -sSL https://github.com/hirdforge/nollama/releases/latest/download/nollama-linux-amd64 -o nollama
chmod +x nollama && sudo mv nollama /usr/local/bin/
```

### Install llama-server

```bash
# Auto-detects your platform and GPU
nollama runtime install
```

### Pull and run a model

```bash
nollama pull unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S
nollama serve &
nollama load Qwen3.6-35B-A3B-Q4_K_S.gguf
```

Your model is now serving at `http://localhost:11434/v1/chat/completions`. Point Open WebUI, Continue, or any OpenAI-compatible client at it.

---

## Usage

### Inspect a model

See what nollama knows about a GGUF before loading it:

```bash
$ nollama inspect gemma-4-26B-A4B-Q3_K_XL.gguf

  Model:     gemma-4-26B-A4B-Q3_K_XL
  Arch:      Gemma4 (MoE, 26B total, 4B active)
  Quant:     Q3_K_XL
  Size:      12.1 GB
  Context:   131,072 (embedded)
  Template:  Jinja ✓ (embedded in GGUF)

  Available hardware:
    GPU 0: RTX Pro 4000 Blackwell — 24,576 MiB (23,100 free)
    GPU 1: RTX 5060 Ti           — 16,384 MiB (16,000 free)
    CPU:   96 GB RAM, 24 threads

  Recommendation: GPU 0 (18.3 GB estimated, fits with 4.7 GB headroom)
```

### Load with control

```bash
# Auto-detect best GPU
nollama load model.gguf

# Explicit GPU
nollama load model.gguf --gpu 0

# CPU inference
nollama load model.gguf --cpu

# See exactly what flags would be passed (nothing hidden)
nollama load model.gguf --dry-run

# Override any llama-server flag
nollama load model.gguf -- --ctx-size 131072 --parallel 4 --cache-type-k q8_0

# Use a specific runtime (fork)
nollama load model.gguf --runtime turboquant
```

### Fleet status

```bash
$ nollama status

NODE      GPU              MODEL                         VRAM         ENDPOINT             UPTIME
agent-host     RTX Pro 4000 24G gemma-4-26B-A4B-Q3_K_XL      18.2/24.0GB  http://agent-host:11434   3d 14h
agent-host     5060 Ti 16GB     gemma-4-26B-A4B-Q5_K_S       14.1/16.0GB  http://agent-host:11435   3d 14h
gpu-host      RTX 3090 24GB    Qwen3.6-35B-A3B-Q4_K_S       23.2/24.0GB  http://gpu-host:11434    1d 2h

MODELS: 8 files, 142GB (/mnt/models/)
NODES:  2 online
```

### Multiple runtimes

Run different models on different llama.cpp forks simultaneously:

```bash
# Install mainline
nollama runtime install

# Build TheTom's TurboQuant fork
nollama runtime build --repo https://github.com/TheTom/llama-cpp-turboquant \
                      --branch feature/turboquant-kv-cache \
                      --name turboquant

# Standard models use mainline, TurboQuant models use the fork
nollama load standard-model.gguf
nollama load turbo-model.gguf --runtime turboquant
```

### Federation

```bash
# Register remote nodes
nollama remote add gpu-host http://gpu-host.example.internal:11434
nollama remote add inference-host http://inference-host.example.internal:11434

# Manage models across your fleet
nollama load model.gguf --node gpu-host
nollama pull org/model:quant --node inference-host
nollama status  # shows all nodes
```

### Hardware profiles

Known-good flag combinations, community-contributed:

```bash
# Asymmetric TurboQuant (K@q8, V@turbo3 — preserves retrieval at 131K+ context)
nollama load model.gguf --profile turboquant-asymmetric

# Maximum agent slots
nollama load model.gguf --profile agent-fleet

# Stack profiles
nollama load model.gguf --profile turboquant-asymmetric --profile agent-fleet
```

### Config file

Optional. Define what to autoload on startup:

```yaml
# /etc/nollama/config.yaml
model_dir: /mnt/models/gguf
bind: 0.0.0.0:11434

autoload:
  - model: gemma-4-26B-A4B-Q3_K_XL.gguf
    gpu: 0
    runtime: turboquant
    flags:
      ctx-size: 131072
      parallel: 8
      cache-type-k: q8_0
      cache-type-v: turbo3

  - model: Qwen3.6-35B-A3B-Q4_K_S.gguf
    gpu: 1
```

```bash
nollama serve --config /etc/nollama/config.yaml
```

---

## Commitments

nollama will never:

- **Store models in hashed blob directories.** Your GGUFs are files in a folder. Period.
- **Require a proprietary configuration format.** No Modelfile. CLI flags and an optional YAML config.
- **Offer cloud-hosted inference.** Local only. Forever.
- **Hide which flags are being passed.** `--dry-run` shows everything. Transparency is the feature.
- **Silently override GGUF-embedded metadata.** The GGUF has a chat template. llama-server reads it. We don't re-implement it.
- **Gate quantization formats behind an allow-list.** If llama-server supports it, nollama supports it.
- **Take VC money.** This project exists to serve its users, not investors.

---

## How it works

nollama is a lifecycle manager and traffic cop. It does not do inference. llama-server does all the real work, untouched, as a child process.

```
                    ┌──────────────────────────┐
                    │      nollama serve       │
                    │                          │
   Requests ──────►│  ┌────────┐ ┌────────┐   │
   (OpenAI API)    │  │llama-  │ │llama-  │   │
                   │  │server  │ │server  │   │
                   │  │model-a │ │model-b │   │
                   │  │GPU 0   │ │GPU 1   │   │
                   │  └────────┘ └────────┘   │
                    └──────────────────────────┘
```

When you `nollama load model.gguf`:

1. nollama reads the GGUF header (architecture, quant, context length, template)
2. nollama inventories your hardware (GPUs, VRAM, CPU, RAM)
3. nollama computes the right llama-server flags
4. nollama spawns llama-server as a child process
5. nollama proxies requests to it

You can see every flag with `--dry-run`. You can override any flag with `--`. nollama never touches the inference path.

---

## vs Ollama

| | Ollama | nollama |
|---|---|---|
| Model storage | Hashed blob store | Plain GGUF files |
| Chat templates | Hardcoded list, Go template syntax | Reads GGUF-embedded Jinja via llama-server |
| Config format | Modelfile (proprietary) | CLI flags + optional YAML |
| Inference engine | Forked ggml (custom, slower) | Stock llama-server (unmodified) |
| Performance | 30-50% overhead vs llama.cpp | 0% overhead (pure proxy) |
| Quantizations | 5 types | All (inherits llama-server) |
| Multi-node | No | Federation built-in |
| Fork support | No | Per-model runtime selection |
| Cloud | Yes (pivot in progress) | Never |
| Flag visibility | Hidden | `--dry-run` shows everything |
| Funding | Y Combinator (VC) | None |

---

## Contributing

nollama is MIT licensed and contributions are welcome.

**Hardware profiles** are the easiest way to contribute — if you've found a good flag combination for your GPU + model setup, submit it as a YAML file in `profiles/`.

**Bug reports** with real hardware details (GPU model, GGUF file, expected vs actual behavior) are extremely valuable.

**Code contributions** should follow the core principle: nollama manages processes and proxies requests. It never touches inference. If a feature requires modifying how llama-server works, it belongs upstream in llama.cpp, not here.

---

## Attribution

nollama exists because of [llama.cpp](https://github.com/ggerganov/llama.cpp), created by Georgi Gerganov and maintained by hundreds of contributors. Without their work, local LLM inference wouldn't exist. nollama manages llama-server processes and gets out of the way. The real work happens in llama.cpp.

---

## License

MIT

---

*Built by [Kit Porath](https://github.com/architkit). No VC. No cloud. No blob store. Just models on your hardware.*