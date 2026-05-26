"""Evaluate a trained router checkpoint on MMLU-Pro."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("SAFETY_AWARE_ROUTING", "1")

import torch
from core.router.router import Router
from core.graph.graph import Graph
from core.llm.llm_profile import llm_profile
from core.agent.reasoning_profile import reasoning_profile
from core.prompts.tasks_profile import tasks_profile
from core.utils.utils import fix_random_seed
from core.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets.mmlupro_dataset import MmluProDataset, mmlu_pro_is_correct


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_agent", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    router = Router(hidden_dim=256, max_agent=args.max_agent, device=device).to(device)
    router.load_state_dict(torch.load(args.ckpt, map_location=device))
    router.eval()

    test_set = MmluProDataset(split="test")
    tasks = tasks_profile()
    llms = llm_profile()
    reasonings = reasoning_profile()

    n_correct, n_total = 0, 0
    items = []

    with torch.no_grad():
        for item in test_set:
            queries = [item["question"]]
            selected_llms, selected_roles, selected_collabs, agent_num, *_ = router(
                queries, tasks, llms, reasonings,
                prompt_file="core/roles/finalnode/mmlupro.json",
            )
            graph = Graph(
                domain="Commonsense",
                llm_names=[m["Name"] for m in selected_llms[0]],
                agent_names=[r["Name"] for r in selected_roles[0]],
                decision_method=selected_collabs[0]["Name"],
            )
            answers = graph.run([item["question"]])
            pred = answers[0]
            correct = mmlu_pro_is_correct(pred, item["answer"])
            n_correct += int(correct)
            n_total += 1
            items.append({"id": item.get("id"), "pred": pred, "gold": item["answer"], "correct": correct})

    summary = {
        "config": "router[mmlupro]",
        "n_queries": n_total,
        "utility": {"solved": n_correct, "total": n_total, "accuracy": n_correct / max(1, n_total)},
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "items": items}, f, indent=2)


if __name__ == "__main__":
    main()
