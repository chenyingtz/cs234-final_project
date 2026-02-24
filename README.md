# SRL: Supervised Reinforcement Learning for LLM Reasoning

End-to-end implementation of **Supervised Reinforcement Learning (SRL)** for LLM step-wise reasoning: train Qwen2.5-7B-Instruct on s1K-1.1 expert trajectories, then evaluate on AIME24.

## Architecture

1. **Data prep**: Load s1K-1.1 from HuggingFace; parse expert solutions into numbered steps (`1. …`, `2. …`); create N-1 training instances per solution (steps 2..N).
2. **SRL instances**: Each instance has `context = problem + expert steps 1..k-1`, `target = expert step k`.
3. **Prompt template**: System + user chat; user includes problem and previous steps; model must output `<think>...</think> [ACTION STEP]`.
4. **Reward**: Computed only on the parsed action step (ignore `<think>`). Baseline: `difflib.SequenceMatcher` ratio between model step and expert step.
5. **Invalid output**: Missing think tags, multiple steps, or parse failure → reward = -1.
6. **GRPO**: Sample G rollouts per prompt; compute group-normalized advantages `A_i = (r_i - mean(r)) / (std(r) + eps)`; apply PPO-style clipped policy gradient.
7. **Dynamic sampling**: Filter prompts whose rollout rewards have `std < eps_std` (weak signal); keep sampling until batch is full.
8. **Optional KL penalty** against reference model (configurable; default 0).
9. **Checkpointing**: Save model + tokenizer every N steps; support `--resume`.
10. **Evaluation**: lm-evaluation-harness task `aime24`; greedy decoding and Avg@N (N=1, 32) with HF or vLLM backend.
11. **Answer extraction**: Utility for AIME-style integer 0–999 from model output (`\boxed{...}`, etc.).
## Setup

```bash
pip install -r requirements.txt
```

## Data Prep

Create step-wise SRL training instances from s1K-1.1:

```bash
python -m src.data_prep --output data/srl_instances.jsonl
```

## Training

### Full training (Qwen2.5-7B)

```bash
python -m src.train_srl --config configs/srl_qwen7b.yaml
```

Or with explicit args:

```bash
python -m src.train_srl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/srl_instances.jsonl \
  --output-dir checkpoints/srl \
  --num-steps 500 \
  --batch-size 4 \
  --group-size 4 \
  --checkpoint-every 100
```

### Resume

```bash
python -m src.train_srl --resume checkpoints/srl/step_100 --data data/srl_instances.jsonl --num-steps 500
```

## Evaluation (AIME24)

### Base model (greedy)

```bash
python -m src.eval_aime24 --model Qwen/Qwen2.5-7B-Instruct --backend hf --mode greedy --output-dir results/aime24_greedy_base
```

### Base model (Avg@32)

```bash
python -m src.eval_aime24 --model Qwen/Qwen2.5-7B-Instruct --backend hf --mode avg32 --output-dir results/aime24_avg32_base
```

### SRL fine-tuned checkpoint (greedy + Avg@32)

```bash
python -m src.eval_aime24 --model-path checkpoints/srl/step_500 --backend hf --mode greedy --output-dir results/aime24_greedy_srl
python -m src.eval_aime24 --model-path checkpoints/srl/step_500 --backend hf --mode avg32 --output-dir results/aime24_avg32_srl
```

Or use the shell wrapper:

```bash
bash scripts/run_eval_srl_checkpoint.sh checkpoints/srl/step_500
```

### vLLM backend (requires separate vLLM server)

Start vLLM first, then:

```bash
python -m src.eval_aime24 --model Qwen/Qwen2.5-7B-Instruct --backend vllm --mode greedy
```

## Tests

```bash
pytest tests/ -v
```

## Project Layout

```
xzt/
├── src/
│   ├── data_prep.py    # s1K-1.1 → SRL JSONL
│   ├── prompts.py      # SRL chat templates
│   ├── reward.py       # SequenceMatcher + parse_srl_output
│   ├── grpo_trainer.py # GRPO loop, dynamic sampling
│   ├── train_srl.py    # Training entrypoint
│   ├── eval_aime24.py  # lm-eval AIME24 wrapper
│   └── utils.py        # seed, JSONL, extract_aime_answer
├── configs/
│   ├── srl_qwen7b.yaml
│   └── smoke_test.yaml
├── scripts/
│   ├── run_eval_aime24.sh
│   └── run_eval_srl_checkpoint.sh
├── tests/
│   ├── test_reward.py
│   ├── test_data_prep.py
│   └── test_utils.py
├── requirements.txt
└── README.md
```

## Exact Commands Summary

| Stage | Command |
|-------|---------|
| Data prep | `python -m src.data_prep --output data/srl_instances.jsonl` |
| Train (full) | `python -m src.train_srl --config configs/srl_qwen7b.yaml` |
| Eval base greedy | `python -m src.eval_aime24 --model Qwen/Qwen2.5-7B-Instruct --mode greedy` |
| Eval base Avg@32 | `python -m src.eval_aime24 --model Qwen/Qwen2.5-7B-Instruct --mode avg32` |
| Eval SRL greedy | `python -m src.eval_aime24 --model-path checkpoints/srl/step_500 --mode greedy` |
| Eval SRL Avg@32 | `python -m src.eval_aime24 --model-path checkpoints/srl/step_500 --mode avg32` |
| Tests | `pytest tests/ -v` |
