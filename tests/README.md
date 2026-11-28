# Tests in `tests`

## Where tests live

Four systems, split by what they can see; `python run_tests.py` runs all of them.

| place | style | covers |
|---|---|---|
| `tests/check_*.py` (this folder) | plain scripts, own `check` harness, `python -m tests.<name>` | the trainer: `graph`, `engine`, `families`, `nodes`, labs. Must not import `backend` (the layer check enforces it) |
| `backend/tests/test_*.py` | plain scripts, `backend.tests.support` helpers, run directly or by `backend/tests/run_all.py` | the server: use cases, API, persistence, wiring. The `sys.path` line at the top of each file is what makes it runnable directly |
| `frontend/tests/*.test.mjs` | `node --test` | the pure parts of the page code (board model, selection, chart maths); DOM is stubbed |
| `tests/browser/check_*.py` | Playwright, `python -m tests.browser [name]` (needs `pip install playwright` and its Chromium) | the pages in a real Chromium against a real server and real graph child processes, on tiny CPU stand-ins (`tiny_profile/sitecustomize.py` swaps the model families and fakes a 12 GB card): apps and their graphs, training runs (status, Save now, history, queue, Stop, host memory ceiling), image generation, a server stopped, crashed or lost while a run trains (the run is adopted, stopped, or recorded), start/settings/help/datasets pages, the app page editor, node pictures, resize, the live preview in Preview Image. About 10 minutes in all; nothing needs a card |

A check that needs both the trainer and the server belongs in `backend/tests`, because that is the side allowed to import both.

Plain scripts (`python -m tests.<name>`), grouped by what they protect. Each file's docstring states its purpose. Run only the group that matches what changed; nothing here needs a GPU. `python -m tests.run_all [name-part]` runs every `check_*` script (or the ones matching) in its own process and prints one line each; the whole set takes about five minutes, `check_resume` and `check_accept` being the slowest.

| group | files | protects |
|---|---|---|
| graph | `check_graph_core`, `check_addons` | declarations, validation, runner, events, settings; addons (slots, wires, reuse, event scopes, the outside API) |
| files and families | `check_detection`, `check_families_torch`, `check_sdxl`, `check_adapters` | recognizing model files, SDXL building blocks, adapter plans and file formats |
| data | `check_data` | epoch barrier and event order, per-VAE cache refusal, prompt determinism and resume, bucketing, dataset nodes |
| engine | `check_engine`, `check_memory` | training steps on a tiny UNet: only adapters train, accumulation, stopping, fused equals plain, checkpointing equals plain; budget, pressure levers, eviction order and refusals on a scripted fake device |
| performance lab | `check_perf` | kernel categories, cost and rating arithmetic, findings rules, a profiling window over the tiny trainer |
| acceptance lab | `check_accept` | regions, block recorder, quantization probe (a planted fault is found in its block), artifact parsing, every check passing and failing, suites and verdicts, the runner with a fake command, visual aids |
| node surface | `check_node_surface` | field groups: nodes with many fields group them all, small groups, few fields visible before opening anything |
| regions and lab | `check_regions`, `check_lab` | timesteps per image (draws, copies, policies, ladder, streaming, calibration, self-perceptual loss); the experiment lab (samplers, grid, templates, reports) |
| generation | `check_generate` | the flow sampling scheme (shifted positions, exact recovery, refusals), the `sdxl-image` starter graph (prompt to saved PNG) on tiny stand-ins: adapter list syntax, seeds and the recipe inside each file, an image alone equals the same image in a batch, stopping, size refusal |
| restart | `check_resume` | a run cut and continued in a new process against one never cut (`check_lab` covers the lab's own stop at a cut), for the lab's recipes: adapters and losses must be identical |
| whole pipeline | `check_pipeline`, `check_anima_pipeline` | the `sdxl-lora` and `anima-lora` starter graphs (dataset to saved adapter) through the graph runner on tiny stand-ins (`pipeline_nodes.py`): ports, event order, stopping, files that come out |
| Anima | `check_anima` | key layout against the real files' headers, DiT, adapter, Qwen3, tokenizers, Wan VAE for images, flow schedule and objective, adapter keys; `reference/equivalence_*` compare against ComfyUI/Hugging Face code when it is present |

Hardware checks live in `scripts/xpu_smoke.py` (real weights, real device) and are run by the owner.
