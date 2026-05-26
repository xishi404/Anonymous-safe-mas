# Routing for Safety

Code for *Learning to Route Safely in Multi-Agent LLM Systems*.

## Repository structure

```
core/
  router/             constrained-MORL router
  graph/              multi-agent scaffold
  agent/              agent runner
  llm/                LLM client wrappers
  roles/              role pool definitions
  prompts/            task profile
  utils/              helper utilities
experiments/
  eval_mmlupro.py     MMLU-Pro evaluation entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Model endpoints are configured in `core/llm/llm_profile.py`. The file maps each agent-pool model name to an OpenAI-compatible base URL — point it at your local serving setup (e.g. vLLM, Ollama, TGI) or any other OpenAI-compatible backend.

For ad-hoc endpoints not listed in the profile, the runner falls back to the `URL` and `KEY` environment variables (or a `.env` file in the repository root).

## Data

The evaluation script expects the MMLU-Pro test split. Download it from the [official release](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) and place the file at:

```
data/mmlupro/test.jsonl
```

The dataset loader is imported as `datasets.mmlupro_dataset`.

## Usage

```bash
python -m experiments.eval_mmlupro \
    --ckpt path/to/checkpoint.pth \
    --out results.json
```

Outputs a JSON file with per-item predictions and a summary block (`solved / total / accuracy`).

## License

MIT.
