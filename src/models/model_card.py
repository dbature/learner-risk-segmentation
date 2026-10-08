"""Writes docs/model_card.md from the run's own reports, following the
section structure of Mitchell et al. (2019). Every number in the card is read
from reports/*.json, so the card cannot drift from the model it describes."""
from __future__ import annotations

from pathlib import Path


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


PRETTY = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}


def write_model_card(path: Path, metrics: dict, fairness: dict, explanations: dict) -> Path:
    sel = metrics["selected"]
    c = metrics["candidates"][sel]
    t = c["test"]
    b = metrics["baselines"]
    sens = metrics["sensitivity"]
    reg = metrics["registry"]
    top = ", ".join(r["label"] for r in explanations["global_importance"][:5])
    after, before = fairness["after"], fairness["before"]
    comp = fairness["comparison"]
    eg_cost = comp["unmitigated"]["auc"] - comp["fairlearn_exponentiated_gradient"]["auc"]
    names = {"gender": "Sex (recorded as gender)", "age_band": "Age band", "disability": "Disability",
             "imd_band": "IMD band"}

    failed = [a for a, v in fairness["gate_after"].items() if not v["pass"]]
    imd = {g["group"]: g["recall"] for g in after["imd_band"]["groups"]}
    imd_lo, imd_hi = imd.get("0-10%"), imd.get("90-100%")

    def frow(a):
        ga = fairness["gate_after"][a]
        return (f"| {names[a]} | {_f(before[a]['recall_gap'])} | {_f(after[a]['recall_gap'])} | "
                f"{_f(after[a]['disparate_impact_ratio'], 2)} | {_f(after[a]['heterogeneity_p'], 2)} | "
                f"{ga['rule']} | {'pass' if ga['pass'] else 'FAIL'} |")

    lines = [
        "# Model Card: Learner Non-Completion Risk at Day 30",
        "",
        f"Registered as `{reg['name']}` version {reg['version']}, alias `{reg['alias']}`, stage **{reg['stage']}**. "
        "Generated automatically from `reports/*.json` by `src/models/model_card.py`.",
        "",
        "## Model details",
        f"* Algorithm: {PRETTY.get(sel, sel)} (Chen and Guestrin, 2016) inside a scikit-learn pipeline (median imputation with missing-value "
        "indicators, scaling, one-hot encoding of module).",
        "* Tuned hyperparameters: " + ", ".join(f"{k.replace('clf__', '')} = {v}" for k, v in c["params"].items()) + ".",
        "* Developed by Desmond Amos Bature, BAN6800 Data Analytics Capstone, Nexford University, October 2026.",
        "* Training data: Open University Learning Analytics Dataset (Kuzilek et al., 2017), processed by the "
        "Module 3 pipeline. Learner identifiers are pseudonymised.",
        "",
        "## Intended use",
        "* Rank learners still enrolled on day 30 of a course by risk of not completing (withdrawing or failing), "
        "so that a coach with capacity for about a quarter of the cohort reaches the learners most likely to need help.",
        "* Decision support for a named human. The model never contacts a learner and never removes support.",
        "* Out of scope: admissions, grading, any sanction, and learners who left before day 30 "
        "(they are a separate registration-time segment).",
        "",
        "## Factors",
        "Fairness is audited across sex, age band, disability and IMD (deprivation) band. "
        "None of these is a model input. Sex is used only by the post-processing step that equalises recall.",
        "",
        "## Metrics",
        "Ranking quality (ROC AUC, average precision) and the operating point a coach uses: flag the riskiest 25%, "
        "then measure recall (share of non-completers reached) and precision (share of flags that are right).",
        "",
        "## Evaluation data",
        f"Grouped 80/20 hold-out: {metrics['split']['n_test']:,} registrations, no learner on both sides. "
        f"Non-completion rate {_pct(metrics['split']['test_base_rate'])}. "
        f"Time-aware check: trained on 2013B to 2014B, tested on {sens['time_aware']['n_test']:,} new learners in 2014J.",
        "",
        "## Quantitative analysis",
        "| Model | ROC AUC | Avg precision | Recall at 25% | Precision at 25% |",
        "|---|---|---|---|---|",
        f"| Majority class | {_f(b['baseline_majority']['roc_auc'])} | {_f(b['baseline_majority']['average_precision'])} | 0.0% | n/a |",
        f"| Random flagging | {_f(b['baseline_random']['roc_auc'])} | {_f(b['baseline_random']['average_precision'])} | "
        f"{_pct(b['baseline_random']['recall'])} | {_pct(b['baseline_random']['precision'])} |",
        f"| Registration data only | {_f(b['baseline_registration_lr']['roc_auc'])} | "
        f"{_f(b['baseline_registration_lr']['average_precision'])} | {_pct(b['baseline_registration_lr']['recall'])} | "
        f"{_pct(b['baseline_registration_lr']['precision'])} |",
        f"| First assessment only | {_f(b['baseline_first_assessment_lr']['roc_auc'])} | "
        f"{_f(b['baseline_first_assessment_lr']['average_precision'])} | {_pct(b['baseline_first_assessment_lr']['recall'])} | "
        f"{_pct(b['baseline_first_assessment_lr']['precision'])} |",
    ]
    for name, cand in metrics["candidates"].items():
        ct = cand["test"]
        mark = " (selected)" if name == sel else ""
        lines.append(f"| {PRETTY.get(name, name)}{mark} | {_f(ct['roc_auc'])} | {_f(ct['average_precision'])} | "
                     f"{_pct(ct['recall'])} | {_pct(ct['precision'])} |")
    lines += [
        "",
        f"Time-aware ROC AUC {_f(sens['time_aware']['roc_auc'])}; seed stability "
        f"{', '.join(_f(s['auc']) for s in sens['seed_stability'])}; with 25% input noise on click counts, "
        f"{_pct(sens['input_noise'][-1]['flag_flip_rate'])} of flags change.",
        "",
        "### Fairness (test set, recall at 25% capacity)",
        "| Attribute | Recall gap before | Recall gap after | Disparate impact ratio | Chi-square p | Gate rule | Gate |",
        "|---|---|---|---|---|---|---|",
        *[frow(a) for a in ["gender", "age_band", "disability", "imd_band"]],
        "",
        f"Groups with fewer than {fairness['min_positives_gated']} positive cases in the test set are reported but "
        "not gated (age 55 and over; missing IMD).",
        "",
        f"IMD band: the literal Module 1 rule (largest minus smallest recall, {_f(after['imd_band']['recall_gap'])}) cannot "
        f"test ten bands. A perfectly fair model would show a median gap of {_f(after['imd_band']['power_check']['null_median_gap'])} "
        f"at this sample size and pass it only {_pct(after['imd_band']['power_check']['probability_fair_model_passes'])} of the time. "
        "The gate for many-level attributes is therefore a chi-square test of equal recall, pending Ethics Committee ratification. "
        + ("IMD passes it." if fairness["gate_after"]["imd_band"]["pass"] else
           f"IMD fails it (p = {_f(after['imd_band']['heterogeneity_p'])}): recall is highest in the most deprived bands "
           f"({_pct(imd_lo)} in 0-10%) and lowest in the least deprived ({_pct(imd_hi)} in 90-100%)."),
        "",
        "## Release decision",
        ("All fairness gates pass." if fairness["all_gates_pass"] else
         "The fairness gate fails on " + ", ".join(names[a] for a in failed) + ". The model stays in **Staging** and "
         "`src/models/train.promote()` refuses Production until the gate passes or the Ethics Committee formally "
         "changes the rule."),
        "",
        "## Ethical considerations",
        "* The model learns from behaviour that is itself socially patterned: deprived learners are flagged more "
        "often because they do not complete more often. Recall parity, not equal flag rates, is the commitment.",
        "* Mitigation uses sex at the point of decision. That is differential treatment and needs Ethics Committee "
        "approval before release. The in-processing alternative that avoids it costs "
        f"{_f(eg_cost, 2)} AUC.",
        "* A flag is an invitation to support, never a judgement. Coaches see reasons (SHAP), not just scores.",
        "",
        "## Caveats and recommendations",
        f"* Recall at 25% capacity is {_pct(t['recall'])} before mitigation and "
        f"{_pct(fairness['comparison']['capacity_equal_opportunity (selected)']['recall'])} after, short of the Module 2 target of 70%. The target was set "
        "before learners who had already left were removed from the population (Module 3). Acceptance criteria are reset "
        "against the baselines above.",
        f"* Top drivers: {top}. Early assessment behaviour dominates, so modules with nothing due by day 30 get weaker predictions.",
        "* The data is UK adult distance learning used as an analogue. Revalidate before any use on another population.",
        "* Monitor monthly: score drift, recall by group on completed cohorts, and the share of learners with no early assessment.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
