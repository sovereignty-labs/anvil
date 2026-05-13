# nollama — PROJECT.md

**The model runner Ollama should have been.**

---

## Current State

**Phase:** Phase 1 — Core Runner (MVP). Foundation + flag computation + process management + serve daemon + proxy built. Pending integration testing on real hardware.

**Spec location:** `nollama-spec-v1.md` in project knowledge.

**What works:**
- `nollama inspect <model.gguf>` — reads GGUF metadata, detects hardware, shows recommendations
- `nollama load <model.gguf> --dry-run` — computes llama-server flags, shows full command without launching
- `nollama load <model.gguf>` — spawns llama-server as child process, serves OpenAI-compatible API
- `nollama unload <model>` — stops llama-server by model name or port
- `nollama status` — shows all running llama-server processes with model, port, GPU, PID, uptime
- `--` passthrough for arbitrary llama-server flags
- `--llama-server` flag and `NOLLAMA_LLAMA_SERVER` env var for binary path
- GPU selection with most-headroom preference, CPU fallback with `CUDA_VISIBLE_DEVICES=""` isolation
- Smart defaults: `--n-gpu-layers 99`, `--flash-attn on`, `--no-warmup`, `--jinja` (when template present), `--ctx-size` capped by VRAM
- 59 passing tests across all packages (pre-S3)
- Docker image builds on build-node, pushes to Gitea registry, deploys anywhere

**S3 code written (pending integration + testing):**
- `nollama serve --config` — daemon mode with autoload, SIGHUP reload, graceful shutdown
- `nollama models` — scan model_dir, list available GGUFs with metadata
- OpenAI-compatible API proxy with model-name routing (fuzzy match, aliases, single-model shortcut)
- Config package — YAML config loader with autoload, defaults, aliases, merged flags
- Model registry — ScanDir, FuzzyMatchModel
- SIGHUP reconciliation — add/remove models without restarting, no downtime on unchanged models
- Example configs: nollama.example.yaml, agent-host.yaml
- Updated systemd service file with ExecReload for SIGHUP

**Validated against real hardware (S2):**
- agent-host/deep-lab: RTX 5060 Ti 16GB + RTX PRO 4000 Blackwell 24GB, AMD Ryzen 9 5900X
- Successfully loaded Qwen3.6-27B IQ4_XS on CPU (14.4 GB), served inference via nollama's endpoint
- GGUF parser validated against Gemma 4 26B-A4B (MoE 128/8) and Qwen 3.6 27B

---

## Architecture Summary

One Go binary. Manages llama-server child processes with smart defaults derived from GGUF metadata and hardware detection. Pure HTTP reverse proxy on the inference hot path — zero overhead. Plain GGUFs in plain directories. MIT licensed.

---

## Build & Deploy Workflow

**No automated CI/CD.** Manual build on build-node, push to Gitea container registry, pull anywhere.

```bash
# Build (build-node)
cd ~/nollama && git pull origin main
docker build -t git.hirdforge.com/kit/nollama/nollama:latest .
docker push git.hirdforge.com/kit/nollama/nollama:latest

# Deploy (any machine)
docker pull git.hirdforge.com/kit/nollama/nollama:latest
docker create --name tmp git.hirdforge.com/kit/nollama/nollama:latest
docker cp tmp:/usr/local/bin/nollama ./nollama
docker rm tmp
export NOLLAMA_LLAMA_SERVER=/path/to/llama-server
./nollama serve --config /etc/nollama/config.yaml
```

**Code changes:** OpenCode on workstation (~/nollama), branch → PR → merge on Gitea.

---

## Build Phases

### Phase 1: Core Runner (MVP) ← CURRENT TARGET
- [x] Project scaffold (Go module, cobra CLI, package structure)
- [x] GGUF header parser (metadata extraction)
- [x] GPU detection via nvidia-smi
- [x] CPU/RAM detection via /proc
- [x] `nollama inspect` (show GGUF metadata + hardware recommendation)
- [x] Flag computation engine (GPU selection, VRAM estimation, context capping)
- [x] `--dry-run` flag (show computed flags without spawning)
- [x] `--llama-server` flag + `NOLLAMA_LLAMA_SERVER` env var
- [x] Process manager (Start/Stop/List with PID tracking, log files, signal handling)
- [x] `nollama load` (spawn llama-server with computed flags)
- [x] `nollama unload` (stop by model name or port)
- [x] `nollama status` (show running processes)
- [x] `--` passthrough for arbitrary llama-server flags
- [x] CPU fallback with CUDA_VISIBLE_DEVICES isolation
- [x] `nollama serve` daemon mode (S3 — pending integration)
- [x] `nollama models` (list local GGUFs) (S3 — pending integration)
- [x] OpenAI-compatible API proxy with model routing (S3 — pending integration)
- [x] Config file support (autoload on startup) (S3 — pending integration)
- [ ] `nollama pull` from HuggingFace
- [ ] `nollama runtime install` (download pre-built llama-server)
- [ ] `nollama runtime build` (compile from source, including forks)
- [ ] `nollama runtime list/use/add`

**Exit criteria:** New user: `nollama runtime install` → `nollama pull org/model:quant` → `nollama load model.gguf`. Three commands, zero to inference.

### Phase 2: Federation + MCP
- [ ] Management API per node
- [ ] `nollama remote add/rm/list/ping`
- [ ] Aggregated `nollama status` across remotes
- [ ] `nollama --node X load/unload/models`
- [ ] `nollama cp` (cross-node transfer)
- [ ] MCP server (stdio + SSE)
- [ ] MCP tools: status, load, unload, models, pull, inspect, runtimes

### Phase 3: Profiles + Community
- [ ] Built-in profiles (TurboQuant, agent-fleet, subconsciousness)
- [ ] Profile stacking
- [ ] Model aliasing
- [ ] LRU swap
- [ ] Prometheus `/metrics`

### Phase 4: Polish + Ecosystem
- [ ] AMD ROCm + Intel Arc detection
- [ ] Multi-GPU split
- [ ] `nollama bench`
- [ ] Container image
- [ ] Homebrew formula

---

## Decisions Log

| Session | Decision | Rationale |
|---------|----------|-----------|
| S0 (spec) | Go, single binary | Matches Hirdforge philosophy. Static, zero deps. |
| S0 | llama-server as unmodified child process | Never touch inference. Never fork. Never vendor. |
| S0 | Plain GGUF storage, no blob store | Direct counter to Ollama. `ls` works. |
| S0 | `nollama runtime install/build` manages llama-server binary | Ease of use is the core promise. Can't require users to compile. |
| S0 | Support multiple runtimes (mainline, TurboQuant, ik_llama) | Per-model `--runtime` flag. Different forks for different workloads. |
| S0 | MCP server for fleet management | Agents manage inference fleet programmatically. Control plane only. |
| S0 | OpenAI-compatible API only, no Ollama API compat | Industry standard. Ollama's custom API is their problem. |
| S0 | nollama is standalone, not a Hirdforge component | Two projects that compose cleanly. |
| S1 | cobra v1.8.1 for CLI | Stable, minimal deps. pflag included. |
| S1 | QuantDisplayName checks filename first, file_type enum second | Community quants like Q3_K_XL aren't in the GGUF file_type enum. |
| S1 | Hardware detection returns nil (not error) when nvidia-smi missing | No GPU is a valid state, not an error. |
| S2 | No automated CI/CD for now | Manual build on build-node → push to Gitea registry. Avoids act runner issues. |
| S2 | Deploy as native binary, not containerized | nollama needs direct GPU access and process management. Extract binary from container image. |
| S2 | OpenCode on workstation for code changes | Branch → PR → merge workflow via Gitea. |
| S2 | `--flash-attn on` not bare `--flash-attn` | Newer llama-server requires value parameter. |
| S2 | `CUDA_VISIBLE_DEVICES=""` on CPU fallback | Prevents llama-server auto-fit from hanging on occupied GPUs. |
| S2 | `--n-gpu-layers 0` on CPU fallback | Explicitly forces CPU-only inference. |
| S2 | Process manager state is in-memory only | No persistence across nollama restarts. Daemon mode (nollama serve) adds persistence later. |
| S3 | SIGHUP reloads config and reconciles models | Edit config, send HUP, models adjust without downtime on unchanged models. |
| S3 | Proxy single-model shortcut | When one model loaded, all requests route to it regardless of model field. Critical for Open WebUI compat. |
| S3 | Proxy body buffering for model extraction | Read full body to get model name, reconstruct for forwarding. SSE streaming via FlushInterval=100ms. |
| S3 | Config MergedFlags: defaults + per-model overlay | Set flash-attn once in defaults, override per-model only when needed. |
| S3 | gopkg.in/yaml.v3 for config parsing | Standard Go YAML library. No viper dependency needed for config alone. |
| S3 | FuzzyMatchModel: exact → prefix → contains | Forgiving model name matching for API requests and CLI commands. |

---

## Package Structure

```
nollama/
├── cmd/nollama/
│   ├── main.go              # CLI entry, load/unload/status implementations
│   ├── inspect.go           # inspect command
│   ├── serve.go             # serve daemon command (S3)
│   └── models.go            # models list command (S3)
├── internal/
│   ├── config/
│   │   ├── config.go        # YAML config loader (S3)
│   │   └── config_test.go   # Config tests (S3)
│   ├── model/
│   │   ├── gguf.go          # GGUF header parser
│   │   ├── gguf_test.go     # 6 tests
│   │   ├── registry.go      # Model directory scanner (S3)
│   │   └── registry_test.go # Registry tests (S3)
│   ├── hardware/
│   │   ├── hardware.go      # GPU (nvidia-smi) + CPU (/proc) detection
│   │   └── hardware_test.go # 3 tests
│   ├── process/
│   │   ├── flags.go         # Flag computation engine
│   │   ├── flags_test.go    # 21 tests
│   │   ├── manager.go       # Process lifecycle manager
│   │   └── manager_test.go  # 29 tests
│   ├── server/
│   │   ├── server.go        # Serve daemon (S3)
│   │   ├── proxy.go         # API proxy with model routing (S3)
│   │   └── proxy_test.go    # Proxy tests (S3)
│   ├── version/
│   │   └── version.go       # Build version info (ldflags)
│   ├── runtime/             # (empty — runtime management)
│   ├── federation/          # (empty — Phase 2)
│   ├── mcp/                 # (empty — Phase 2)
├── configs/
│   ├── nollama.example.yaml # Full example config (S3)
│   └── agent-host.yaml           # agent-host-specific config (S3)
├── systemd/
│   └── nollama.service      # Updated for nollama serve --config (S3)
├── S3-INTEGRATION.md        # Interface assumptions for builder (S3)
├── Dockerfile
├── docker-compose.yaml
├── go.mod
├── go.sum
├── Makefile
├── LICENSE (MIT)
├── README.md
└── .gitignore
```

---

## Kit's Test Environment

- **agent-host/deep-lab (LXC 203):** RTX PRO 4000 Blackwell 24GB + RTX 5060 Ti 16GB, AMD Ryzen 9 5900X 12-core, at 203.0.113.203
- **gpu-host (LXC 202):** RTX 3090 24GB at 203.0.113.17
- **workstation:** Kit's workstation, OpenCode runs here with nollama repo at ~/nollama
- **build-node:** Build machine, Docker builds happen here, pushes to Gitea registry
- **Models validated:** Gemma 4 26B-A4B Q5_K_S (17.5 GB), Qwen3.6-27B IQ4_XS (14.4 GB)
- **llama-server paths:** /mnt/deeplab/llama.cpp.mainline/build/bin/llama-server (mainline), /mnt/deeplab/llama.cpp/build/bin/llama-server (fork)

---

## Session History

### S0 — Spec (May 9, 2026)
- Wrote nollama-spec-v1.md (complete)
- Resolved all open questions

### S1 — Foundation Layer (May 9, 2026)
- Project scaffold, GGUF parser, hardware detection, `nollama inspect`
- 9 tests passing, PR #1 merged

### S2 — Flag Computation + Process Management (May 12, 2026)
**Built:**
- Flag computation engine (PR #5): GPU selection with most-headroom, CPU fallback, VRAM estimation, context capping, smart defaults
- Bug fixes (PR #6): Flag display loop, --llama-server persistent flag positioning
- Process manager (PR #7): Start/Stop/List, signal handling, log files, passthrough flag merging, 29 tests
- llama-server compat fixes (PR #8): --flash-attn requires value in newer versions
- CPU fallback fixes (PR #9): --n-gpu-layers 0 to force CPU mode
- GPU isolation (PR #10): CUDA_VISIBLE_DEVICES="" prevents auto-fit hanging

**Validated:** nollama loaded Qwen3.6-27B on CPU via deep-lab, served inference through OpenAI-compatible endpoint.

### S3 — Serve Daemon + Config + Proxy (May 12, 2026)
**Produced:**
- Config package (internal/config/): YAML config loader with autoload, defaults, aliases, merged flags. 8 tests.
- Model registry (internal/model/registry.go): ScanDir, FuzzyMatchModel. 10 tests.
- API proxy (internal/server/proxy.go): HTTP reverse proxy, model-name routing, fuzzy match, aliases, /v1/models, /health. 9 tests.
- Serve daemon (internal/server/server.go): Autoload from config, SIGHUP reconciliation, graceful shutdown.
- CLI commands: `nollama serve --config`, `nollama models --model-dir`
- Example configs: nollama.example.yaml (full reference), agent-host.yaml (Kit's first deployment target)
- Updated systemd service with ExecReload for SIGHUP
- Integration guide (S3-INTEGRATION.md) documenting interface assumptions

**Status:** Code written and reviewed, pending integration into repo and testing on real hardware.

---

## Next Session (S4) Priorities

1. **Integrate and test S3 code** — merge into repo, fix interface mismatches, run on agent-host/deep-lab with agent-host.yaml config. Validate: `nollama serve --config agent-host.yaml` loads model, proxies inference, responds to SIGHUP.

2. **`nollama pull` from HuggingFace** — download GGUFs with quant filter, resume support, original filenames. This completes the "three commands to inference" path for new users: `nollama runtime install` → `nollama pull` → `nollama load`.

3. **`nollama runtime install`** — download pre-built llama-server from GitHub Releases. Auto-detect platform + GPU. This is the remaining piece of the new-user experience.

4. **`nollama runtime build`** — compile from source for fork support (TheTom's TurboQuant, ik_llama.cpp). Per-model `--runtime` flag wiring into autoload config.

**Key validation for S4:** Kit runs `nollama serve --config agent-host.yaml` on deep-lab and it replaces the manual systemd service file. One config file, one command, model loaded, inference served, SIGHUP reloads cleanly.
