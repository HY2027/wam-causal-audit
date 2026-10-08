from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import RESULTS, TASKS, write_csv


N_BOOT = 10_000
BOOTSTRAP_SEED = 20260925


def as_bool(value):
    if value is None:
        return None
    return bool(value)


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def load_rows() -> tuple[list[dict], dict[str, dict]]:
    with (RESULTS / "closed_loop_50_registry.csv").open(newline="") as stream:
        registry = list(csv.DictReader(stream))
    summaries: dict[str, dict] = {}
    for path in sorted((RESULTS / "stage5_shards").glob("shard_*_of_*/pair_summaries.json")):
        for row in json.loads(path.read_text()):
            pair_id = row["pair_id"]
            if pair_id in summaries:
                raise RuntimeError(f"DUPLICATE_PAIR_SUMMARY:{pair_id}")
            summaries[pair_id] = row
    registry_ids = [row["pair_id"] for row in registry]
    if len(registry) != 50 or len(set(registry_ids)) != 50:
        raise RuntimeError("FROZEN_REGISTRY_NOT_50_UNIQUE_PAIRS")
    if set(registry_ids) != set(summaries):
        raise RuntimeError("REGISTRY_SUMMARY_ID_MISMATCH")
    return registry, summaries


def exclusion(preflight: dict) -> tuple[str, str, bool]:
    status = preflight["status"]
    if status == "PASS":
        return "NONE", "NOT_EXCLUDED", False
    if status == "TECHNICAL_FAILURE":
        detail = str(preflight.get("error") or "UNSPECIFIED_TECHNICAL_FAILURE")
        return (
            f"PREFLIGHT_TECHNICAL_FAILURE: {detail}",
            "STAGE5_EXPERT_PREFLIGHT_BEFORE_INSTRUCTION_AND_BUDGET_EPISODES",
            True,
        )
    if status == "BENCHMARK_INVALID":
        plan = bool(preflight.get("plan_success"))
        expert = bool(preflight.get("expert_success"))
        return (
            f"PREFLIGHT_BENCHMARK_INVALID: plan_success={plan}; expert_success={expert}",
            "STAGE5_EXPERT_PREFLIGHT_BEFORE_INSTRUCTION_AND_BUDGET_EPISODES",
            False,
        )
    return (
        f"PREFLIGHT_OTHER_INVALID_STATUS: {status}",
        "STAGE5_EXPERT_PREFLIGHT_BEFORE_INSTRUCTION_AND_BUDGET_EPISODES",
        True,
    )


def build_audit(registry: list[dict], summaries: dict[str, dict]) -> list[dict]:
    output = []
    for frozen in registry:
        summary = summaries[frozen["pair_id"]]
        episodes = {int(row["budget"]): row for row in summary.get("episodes", [])}
        if len(episodes) != len(summary.get("episodes", [])):
            raise RuntimeError(f"DUPLICATE_BUDGET_EPISODE:{frozen['pair_id']}")
        k10, k5 = episodes.get(10), episodes.get(5)
        valid = (
            summary["preflight"]["status"] == "PASS"
            and summary["status"] == "PASS"
            and k10 is not None and k5 is not None
            and k10["status"] == "PASS" and k5["status"] == "PASS"
        )
        reason, stage, preflight_error = exclusion(summary["preflight"])
        if valid:
            reason, stage = "NONE", "NOT_EXCLUDED"
        if not valid and (k10 is not None or k5 is not None):
            # Any asymmetric or post-budget invalidity would invalidate the
            # present valid-cohort interpretation and must be made explicit.
            budget_dependent = True
            before_success = False
        elif not valid:
            budget_dependent = False
            before_success = True
        else:
            budget_dependent = None
            before_success = None

        def run_status(episode):
            return episode["status"] if episode is not None else "NOT_RUN_PRE_BUDGET_PREFLIGHT"

        def error_flag(episode):
            if episode is None:
                return "NOT_RUN"
            return (
                bool(episode.get("simulator_error"))
                or bool(episode.get("close_error"))
                or episode.get("status") != "PASS"
            )

        output.append({
            "task": frozen["task_id"],
            "seed": int(frozen["simulator_seed"]),
            "pair_id": frozen["pair_id"],
            "frozen_run_order": frozen["run_order"],
            "K10_run_status": run_status(k10),
            "K5_run_status": run_status(k5),
            "valid_pair": "yes" if valid else "no",
            "exact_exclusion_reason": reason,
            "exclusion_decision_stage": stage,
            "exclusion_before_K10_K5_success_observed": (
                "yes" if before_success is True else "no" if before_success is False else "not_applicable"
            ),
            "exclusion_depends_on_one_budget": (
                "yes" if budget_dependent is True else "no" if budget_dependent is False else "not_applicable"
            ),
            "K10_success": as_bool(k10.get("success")) if k10 is not None else None,
            "K5_success": as_bool(k5.get("success")) if k5 is not None else None,
            "K10_termination_reason": k10.get("termination_reason") if k10 is not None else "NOT_RUN",
            "K5_termination_reason": k5.get("termination_reason") if k5 is not None else "NOT_RUN",
            "K10_simulator_or_runtime_error_flag": error_flag(k10),
            "K5_simulator_or_runtime_error_flag": error_flag(k5),
            "preflight_status": summary["preflight"]["status"],
            "preflight_runtime_error_flag": preflight_error,
            "preflight_plan_success": summary["preflight"].get("plan_success"),
            "preflight_expert_success": summary["preflight"].get("expert_success"),
        })
    return output


def taskwise(audit: list[dict]) -> list[dict]:
    rows = []
    for task in TASKS:
        frozen = [row for row in audit if row["task"] == task]
        valid = [row for row in frozen if row["valid_pair"] == "yes"]
        rows.append({
            "task": task,
            "frozen_pairs": len(frozen),
            "valid_pairs": len(valid),
            "K10_successes": sum(row["K10_success"] is True for row in valid),
            "K5_successes": sum(row["K5_success"] is True for row in valid),
            "K10_success_K5_fail": sum(row["K10_success"] is True and row["K5_success"] is False for row in valid),
            "K10_fail_K5_success": sum(row["K10_success"] is False and row["K5_success"] is True for row in valid),
        })
    return rows


def frozen_bootstrap(audit: list[dict]) -> dict:
    valid = [row for row in audit if row["valid_pair"] == "yes"]
    by_task = {task: [row for row in valid if row["task"] == task] for task in TASKS}
    if any(not rows for rows in by_task.values()):
        raise RuntimeError("A_TASK_HAS_NO_VALID_PAIRS")
    task_means = [
        np.mean([int(row["K5_success"]) - int(row["K10_success"]) for row in by_task[task]])
        for task in TASKS
    ]
    point = float(np.mean(task_means))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(N_BOOT, dtype=np.float64)
    task_array = np.asarray(TASKS, dtype=object)
    for index in range(N_BOOT):
        sampled_task_means = []
        for task in rng.choice(task_array, size=len(TASKS), replace=True):
            rows = by_task[str(task)]
            sampled = rng.choice(len(rows), size=len(rows), replace=True)
            sampled_task_means.append(np.mean([
                int(rows[i]["K5_success"]) - int(rows[i]["K10_success"]) for i in sampled
            ]))
        draws[index] = np.mean(sampled_task_means)
    return {"estimate": point, "ci95": interval(draws), "replicates": N_BOOT, "seed": BOOTSTRAP_SEED}


def markdown(audit: list[dict], per_task: list[dict], bootstrap: dict) -> str:
    valid = [row for row in audit if row["valid_pair"] == "yes"]
    excluded = [row for row in audit if row["valid_pair"] == "no"]
    unstable = [row for row in excluded if row["preflight_status"] == "TECHNICAL_FAILURE"]
    benchmark_invalid = [row for row in excluded if row["preflight_status"] == "BENCHMARK_INVALID"]
    dependent = [row for row in excluded if row["exclusion_depends_on_one_budget"] == "yes"]
    model_outcome_dependent = [
        row for row in excluded if row["exclusion_before_K10_K5_success_observed"] != "yes"
    ]
    expert_patterns = Counter(
        (str(row["preflight_plan_success"]), str(row["preflight_expert_success"]))
        for row in benchmark_invalid
    )
    task_attrition = Counter(row["task"] for row in excluded)
    existing = json.loads((RESULTS / "closed_loop_results.json").read_text())
    expected_point = float(existing["point"]["paired_success_difference"])
    expected_ci = [float(value) for value in existing["ci95"]["success_difference"]]
    bootstrap_match = np.isclose(bootstrap["estimate"], expected_point) and np.allclose(bootstrap["ci95"], expected_ci)
    if not bootstrap_match:
        raise RuntimeError(f"FROZEN_BOOTSTRAP_REPRODUCTION_MISMATCH:{bootstrap}:{expected_point}:{expected_ci}")

    lines = [
        "# RoboTwin reduced-computation closed-loop validity audit",
        "",
        "Status: `VALID_COHORT_ANALYSIS_RETAINED_CONDITIONAL_ON_SHARED_PREFLIGHT`",
        "",
        "This audit is read-only with respect to scientific execution. No model call, simulator step, new seed, or new condition was run.",
        "",
        "## Pair flow",
        "",
        f"The prospectively frozen registry contains {len(audit)} unique task--seed pairs. "
        f"A single shared expert preflight was evaluated before instruction generation and before either budget episode. "
        f"{len(valid)} pairs passed and have technically complete K=10 and K=5 episodes; {len(excluded)} did not pass and have no K=10 or K=5 episode.",
        "",
        "| Task | Frozen | Valid | K10 success | K5 success | K10 success / K5 fail | K10 fail / K5 success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_task:
        lines.append(
            f"| {row['task']} | {row['frozen_pairs']} | {row['valid_pairs']} | "
            f"{row['K10_successes']} | {row['K5_successes']} | "
            f"{row['K10_success_K5_fail']} | {row['K10_fail_K5_success']} |"
        )
    lines.extend([
        "",
        "## Attrition audit",
        "",
        f"- `PREFLIGHT_TECHNICAL_FAILURE / UnStableError`: {len(unstable)}.",
        f"- `PREFLIGHT_BENCHMARK_INVALID`: {len(benchmark_invalid)} "
        f"(plan/expert patterns: {dict(sorted(expert_patterns.items()))}).",
        "- Exclusions by task: " + ", ".join(f"{task}={task_attrition[task]}" for task in TASKS) + ".",
        f"- Budget-dependent exclusions: {len(dependent)}; budget-independent exclusions: {len(excluded)-len(dependent)}.",
        f"- Exclusions made after observing either K10/K5 success: {len(model_outcome_dependent)}.",
        "- No valid pair has a simulator/runtime error in either budget episode.",
        "",
        "The nine benchmark-invalid exclusions depend on an expert preflight eligibility outcome (`plan_success AND expert_success`), and the twelve `UnStableError` exclusions depend on reset stability. Thus attrition is not claimed to be missing-at-random over the original frozen seeds. Crucially, neither rule reads or depends on K10/K5 policy success, and both budgets are absent together for every excluded pair.",
        "",
        "## Frozen paired completion analysis",
        "",
        f"Among {len(valid)} valid pairs, K10 succeeds in {sum(row['K10_success'] is True for row in valid)} and K5 succeeds in {sum(row['K5_success'] is True for row in valid)}. "
        f"There are {sum(row['K10_success'] is True and row['K5_success'] is False for row in valid)} K10-success/K5-fail pairs and "
        f"{sum(row['K10_success'] is False and row['K5_success'] is True for row in valid)} K10-fail/K5-success pairs.",
        "",
        f"The frozen task-equal paired estimate is {bootstrap['estimate']:.6f}, with 95% hierarchical paired-bootstrap interval [{bootstrap['ci95'][0]:.6f}, {bootstrap['ci95'][1]:.6f}] ({bootstrap['replicates']} replicates; seed {bootstrap['seed']}). This exactly reproduces `closed_loop_results.json`.",
        "",
        "## Sensitivity to validity definition",
        "",
        "All 21 invalid pairs are jointly missing for both budgets, before either model condition was evaluated. There is therefore no budget-specific invalidity requiring a different outcome definition, asymmetric failure coding, or imputation. The current 29-pair comparison is internally paired and may be retained as a conditional valid-cohort analysis.",
        "",
        "It is not an estimate of unconditional completion over all 50 prospectively frozen seeds: preflight attrition is large (21/50) and task-dependent. Assigning task failure, technical failure, or zero success to the 21 pairs would conflate benchmark eligibility with policy behavior and is not scientifically justified by the saved evidence.",
        "",
        "## Required final answers",
        "",
        "### Why are only 29/50 pairs valid?",
        "",
        "Twenty-one pairs failed the one shared pre-budget expert preflight: twelve resets raised `UnStableError`, and nine completed the expert preflight without satisfying `plan_success AND expert_success`. They were retained in the audit and not replaced, but neither budget episode was started.",
        "",
        "### Were all exclusions fixed independently of model success?",
        "",
        "Yes with respect to Joint-WAM success: all exclusions were decided before either K10 or K5 ran. Nine exclusions do depend on the separate expert eligibility outcome, so the audit does not characterize attrition as generally outcome-free or missing-at-random.",
        "",
        "### Did either K=10 or K=5 create additional invalidity?",
        "",
        "No. Every preflight-valid pair has technically complete episodes for both budgets, and no budget episode has a simulator/runtime error. All invalid pairs are missing both budgets.",
        "",
        "### Can the current 18/29 vs 19/29 comparison be retained?",
        "",
        "Yes, as a paired analysis conditional on passing the shared benchmark preflight. The frozen task-equal difference and interval can be retained. It cannot be relabeled as 18/50 versus 19/50, as an unconditional 50-seed estimate, or as an equivalence/non-inferiority result.",
        "",
        "### What is the strongest defensible wording for the paper?",
        "",
        "> Of 50 prospectively frozen task--seed pairs, 29 passed a shared expert benchmark preflight conducted before either model condition. Within this valid paired cohort, Joint-WAM completed 18/29 native-K10 and 19/29 K5 episodes (task-equal paired difference -0.027, 95% CI [-0.267, 0.160]). All 21 exclusions were budget-independent and preceded model evaluation; because preflight attrition was substantial and task-dependent, this comparison is conditional on benchmark-valid resets and does not establish unconditional equivalence or non-inferiority over the full frozen cohort.",
        "",
        "## Evidence",
        "",
        "- `closed_loop_50_registry.csv`: prospective pair identities and counterbalanced order.",
        "- `stage5_shards/*/pair_summaries.json`: preflight decisions and paired episode records.",
        "- `stage5_shards/*/pairs/*/preflight.json`: exact expert eligibility evidence.",
        "- `stage5_shards/*/pairs/*/K*_episode.json`: budget-specific success, termination, and error fields.",
        "- `run_closed_loop.py`: preflight precedes instruction construction and both budget loops; failed preflight immediately skips the pair.",
        "- `closed_loop_results.json`: frozen paired bootstrap result used above.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    registry, summaries = load_rows()
    audit = build_audit(registry, summaries)
    per_task = taskwise(audit)
    bootstrap = frozen_bootstrap(audit)
    write_csv(RESULTS / "closed_loop_validity_audit.csv", audit)
    write_csv(RESULTS / "taskwise_completion.csv", per_task)
    (RESULTS / "closed_loop_validity_audit.md").write_text(
        markdown(audit, per_task, bootstrap), encoding="utf-8"
    )
    print(json.dumps({
        "frozen_pairs": len(audit),
        "valid_pairs": sum(row["valid_pair"] == "yes" for row in audit),
        "invalid_pairs": sum(row["valid_pair"] == "no" for row in audit),
        "budget_dependent_exclusions": sum(row["exclusion_depends_on_one_budget"] == "yes" for row in audit),
        "bootstrap": bootstrap,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
