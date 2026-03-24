import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .question1 import ensure_dir, normalize_text


TOKEN_STRIP = ".,!?;:\"'`[]{}()<>|/\\"

HI_TO_DIGIT = {
    "शून्य": "0",
    "एक": "1",
    "दो": "2",
    "तीन": "3",
    "चार": "4",
    "पांच": "5",
    "पाँच": "5",
    "छह": "6",
    "सात": "7",
    "आठ": "8",
    "नौ": "9",
    "दस": "10",
    "ग्यारह": "11",
    "बारह": "12",
    "तेरह": "13",
    "चौदह": "14",
    "पंद्रह": "15",
    "सोलह": "16",
    "सत्रह": "17",
    "अठारह": "18",
    "उन्नीस": "19",
    "बीस": "20",
}

DIGIT_TO_HI = {value: key for key, value in HI_TO_DIGIT.items()}


@dataclass
class AlignmentOp:
    op: str
    ref_idx: int | None
    hyp_idx: int | None
    ref_token: str
    hyp_token: str


@dataclass
class LatticeBin:
    bin_id: str
    kind: str  # ref | ins
    anchor_index: int
    alternatives: list[str]
    allow_skip: bool
    support: dict[str, int]
    decision_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question-4 lattice-based ASR evaluation.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/question4_transcripts.csv"),
        help="CSV containing one reference and multiple model outputs per utterance.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/q4"),
        help="Output directory for Q4 artifacts.",
    )
    parser.add_argument(
        "--reference-col",
        type=str,
        default="reference",
        help="Reference transcript column name.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="utterance_id",
        help="Utterance id column. If missing, row index is used.",
    )
    parser.add_argument(
        "--model-cols",
        type=str,
        default="",
        help="Comma-separated model output columns. If empty, auto-detect text columns.",
    )
    parser.add_argument(
        "--expected-model-count",
        type=int,
        default=5,
        help="Expected number of ASR model columns (Q4 uses five models). Set 0 to disable this check.",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=int,
        default=3,
        help="Min model support to trust model agreement over reference at a position.",
    )
    parser.add_argument(
        "--alternative-support-threshold",
        type=int,
        default=1,
        help="Min support to include an alternative token in a bin. Use 1 to include all model-proposed alternatives.",
    )
    parser.add_argument(
        "--insertion-support-threshold",
        type=int,
        default=1,
        help="Min support to create an insertion bin between reference positions. Use 1 to include all model-proposed insertions.",
    )
    return parser.parse_args()


def clean_token(token: str) -> str:
    token = token.strip().strip(TOKEN_STRIP).replace("।", "")
    return token


def tokenize(text: str) -> list[str]:
    norm = normalize_text(text)
    tokens = [clean_token(token) for token in norm.split()]
    return [token for token in tokens if token]


def expand_numeric_variants(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for token in list(tokens):
        if token.isdigit() and token in DIGIT_TO_HI:
            out.add(DIGIT_TO_HI[token])
        if token in HI_TO_DIGIT:
            out.add(HI_TO_DIGIT[token])
    return out


def align_tokens(ref_tokens: list[str], hyp_tokens: list[str]) -> tuple[int, int, int, int, list[AlignmentOp]]:
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + sub_cost,  # substitution/equal
            )

    i, j = n, m
    ops_rev: list[AlignmentOp] = []
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                if sub_cost == 0:
                    ops_rev.append(
                        AlignmentOp(
                            op="equal",
                            ref_idx=i - 1,
                            hyp_idx=j - 1,
                            ref_token=ref_tokens[i - 1],
                            hyp_token=hyp_tokens[j - 1],
                        )
                    )
                else:
                    substitutions += 1
                    ops_rev.append(
                        AlignmentOp(
                            op="replace",
                            ref_idx=i - 1,
                            hyp_idx=j - 1,
                            ref_token=ref_tokens[i - 1],
                            hyp_token=hyp_tokens[j - 1],
                        )
                    )
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            ops_rev.append(
                AlignmentOp(
                    op="delete",
                    ref_idx=i - 1,
                    hyp_idx=None,
                    ref_token=ref_tokens[i - 1],
                    hyp_token="",
                )
            )
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            ops_rev.append(
                AlignmentOp(
                    op="insert",
                    ref_idx=i,  # insertion before ref index i
                    hyp_idx=j - 1,
                    ref_token="",
                    hyp_token=hyp_tokens[j - 1],
                )
            )
            j -= 1
            continue

        raise RuntimeError("Alignment backtrace failed.")

    ops = list(reversed(ops_rev))
    distance = dp[n][m]
    return distance, substitutions, deletions, insertions, ops


def detect_model_columns(df: pd.DataFrame, reference_col: str, id_col: str, model_cols_arg: str) -> list[str]:
    if model_cols_arg.strip():
        cols = [item.strip() for item in model_cols_arg.split(",") if item.strip()]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Model columns not found in input file: {missing}")
        return cols

    candidate_cols = [col for col in df.columns if col not in {reference_col, id_col}]
    text_like = [col for col in candidate_cols if df[col].dtype == "object"]
    preferred = [
        col
        for col in text_like
        if any(keyword in col.lower() for keyword in ("model", "asr", "pred", "output", "hyp"))
    ]
    if preferred:
        return preferred
    # Safe fallback: if exactly five text columns remain (Q4 setting), treat them as model outputs.
    if len(text_like) == 5:
        return text_like
    raise ValueError(
        "Could not auto-detect model output columns safely. Pass --model-cols explicitly "
        "(for Q4 this should list five model columns)."
    )


def build_lattice(
    reference_tokens: list[str],
    model_tokens_by_name: dict[str, list[str]],
    agreement_threshold: int,
    alternative_support_threshold: int,
    insertion_support_threshold: int,
) -> tuple[list[LatticeBin], dict[str, Any]]:
    n = len(reference_tokens)
    model_names = list(model_tokens_by_name.keys())
    num_models = len(model_names)

    ref_votes: list[Counter] = [Counter() for _ in range(n)]
    insert_votes: dict[int, Counter] = defaultdict(Counter)
    deletion_counts: list[int] = [0] * n

    for model_name, tokens in model_tokens_by_name.items():
        _, _, _, _, ops = align_tokens(reference_tokens, tokens)
        for op in ops:
            if op.op in {"equal", "replace"} and op.ref_idx is not None:
                ref_votes[op.ref_idx][op.hyp_token] += 1
            elif op.op == "delete" and op.ref_idx is not None:
                deletion_counts[op.ref_idx] += 1
            elif op.op == "insert" and op.ref_idx is not None:
                insert_votes[op.ref_idx][op.hyp_token] += 1

    bins: list[LatticeBin] = []
    trust_events: list[dict[str, Any]] = []

    def add_insertion_bin(position: int) -> None:
        if position not in insert_votes:
            return
        votes = insert_votes[position]
        strong_tokens = {token: count for token, count in votes.items() if count >= insertion_support_threshold}
        if not strong_tokens:
            return

        alts = expand_numeric_variants(set(strong_tokens.keys()))
        support = {token: int(votes.get(token, 0)) for token in alts}
        bins.append(
            LatticeBin(
                bin_id=f"ins_{position}_{len(bins)}",
                kind="ins",
                anchor_index=position,
                alternatives=sorted(alts),
                allow_skip=True,
                support=support,
                decision_reason=f"insertion_supported_by>={insertion_support_threshold}_models",
            )
        )
        trust_events.append(
            {
                "type": "insertion_consensus",
                "position": position,
                "tokens": strong_tokens,
                "rule": "Insertion trusted when same token is inserted by multiple models.",
            }
        )

    for idx in range(n + 1):
        add_insertion_bin(idx)
        if idx >= n:
            continue

        ref_token = reference_tokens[idx]
        votes = ref_votes[idx]
        ref_support = int(votes.get(ref_token, 0))
        top_token = ""
        top_support = 0
        if votes:
            top_token, top_support = votes.most_common(1)[0]

        alts: set[str] = {ref_token}
        for token, count in votes.items():
            if count >= alternative_support_threshold:
                alts.add(token)
        alts = expand_numeric_variants(alts)

        allow_skip = False
        reasons: list[str] = []

        if top_support >= agreement_threshold and top_token and top_token != ref_token and ref_support <= 1:
            allow_skip = True
            alts.add(top_token)
            reasons.append("majority_non_reference_substitution")

        if deletion_counts[idx] >= agreement_threshold:
            allow_skip = True
            reasons.append("majority_deletion_support")

        support = {token: int(votes.get(token, 0)) for token in alts}
        support.setdefault(ref_token, ref_support)
        bins.append(
            LatticeBin(
                bin_id=f"ref_{idx}",
                kind="ref",
                anchor_index=idx,
                alternatives=sorted(alts),
                allow_skip=allow_skip,
                support=support,
                decision_reason="|".join(reasons) if reasons else "reference_preserved",
            )
        )

        if reasons:
            trust_events.append(
                {
                    "type": "reference_override",
                    "position": idx,
                    "reference_token": ref_token,
                    "top_model_token": top_token,
                    "top_model_support": int(top_support),
                    "reference_support": int(ref_support),
                    "deletion_support": int(deletion_counts[idx]),
                    "reasons": reasons,
                    "rule": "Trust model agreement when majority evidence contradicts reference.",
                }
            )

    lattice_meta = {
        "num_reference_tokens": n,
        "num_models": num_models,
        "num_bins": len(bins),
        "trust_events": trust_events,
        "agreement_threshold": agreement_threshold,
        "alternative_support_threshold": alternative_support_threshold,
        "insertion_support_threshold": insertion_support_threshold,
    }
    return bins, lattice_meta


def lattice_edit_distance(hyp_tokens: list[str], bins: list[LatticeBin]) -> tuple[float, float]:
    b = len(bins)
    m = len(hyp_tokens)
    inf = 10**9
    dp = [[inf] * (m + 1) for _ in range(b + 1)]
    dp[0][0] = 0
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, b + 1):
        bin_item = bins[i - 1]
        skip_cost = 0 if bin_item.allow_skip else 1
        dp[i][0] = dp[i - 1][0] + skip_cost
        for j in range(1, m + 1):
            # Match/substitute against bin alternatives.
            token_cost = 0 if hyp_tokens[j - 1] in bin_item.alternatives else 1
            best = dp[i - 1][j - 1] + token_cost

            # Skip current bin (deletion wrt lattice).
            best = min(best, dp[i - 1][j] + skip_cost)

            # Insert token from hypothesis.
            best = min(best, dp[i][j - 1] + 1)

            dp[i][j] = best

    distance = float(dp[b][m])
    return distance, distance


def serialize_bins(bins: list[LatticeBin]) -> list[dict[str, Any]]:
    return [
        {
            "bin_id": item.bin_id,
            "kind": item.kind,
            "anchor_index": item.anchor_index,
            "alternatives": item.alternatives,
            "allow_skip": item.allow_skip,
            "support": item.support,
            "decision_reason": item.decision_reason,
        }
        for item in bins
    ]


def build_pseudocode_markdown() -> str:
    return """# Q4 Lattice Method (Theory + Pseudocode)

## Alignment Unit Choice: Word
- Word-level alignment keeps interpretation intuitive for WER and error attribution.
- It captures substitutions, insertions, and deletions directly without needing subword decoding logic.

## Core Idea
- Replace single rigid reference string with a sequence of lattice bins.
- Each bin contains valid alternatives collected from model agreement + reference token.
- Reference can be down-weighted (optional skip) when model consensus strongly disagrees.

## Trust Rule
- Trust model consensus over reference when:
  - majority models propose the same non-reference token at a position, and
  - reference support is weak, or
  - majority models delete the reference token.
- For insertions, create optional insertion bins from model-supported tokens (threshold configurable; default includes all model-supported insertions).

## Pseudocode
```text
for each utterance:
    ref_tokens = tokenize(reference)
    for each model_output:
        hyp_tokens = tokenize(model_output)
        ops = align(ref_tokens, hyp_tokens)
        collect per-position token votes
        collect insertion votes
        collect deletion votes

    bins = []
    for each position between/at ref tokens:
        if insertion token meets configured support threshold:
            bins.append(optional insertion bin)
        build reference bin with:
            alternatives = {reference_token} U strongly-supported model alternatives
            allow_skip = True if strong disagreement with reference
            decision metadata

    for each model_output:
        lattice_distance = min_edit_distance(hyp_tokens, bins)
        rigid_distance = edit_distance(hyp_tokens, ref_tokens)
        record WERs and deltas
```

## Lattice WER DP
- DP state `dp[i][j]`: minimum edit cost matching first `i` lattice bins to first `j` hypothesis tokens.
- Transitions:
  - consume bin + token (0 cost if token in bin alternatives else substitution cost 1)
  - skip bin (0 if bin optional, else deletion cost 1)
  - insert hypothesis token (cost 1)
"""


def build_report_markdown(
    model_summary_df: pd.DataFrame,
    overall_summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Question-4 Report (Lattice-Based ASR Evaluation)")
    lines.append("")
    lines.append("## Alignment Unit")
    lines.append("- Chosen unit: **word-level** alignment.")
    lines.append("- Rationale: direct compatibility with WER and clear handling of insertions/deletions/substitutions.")
    lines.append("")
    lines.append("## Reference Trust Strategy")
    lines.append("- Build bins from reference + model alternatives at each position.")
    lines.append("- Override strict reference when strong model agreement contradicts it.")
    lines.append("- Add optional insertion bins from model-supported insertions (default includes all model-supported tokens).")
    lines.append("")
    lines.append("## Model WER Comparison")
    lines.append(
        "| Model | Rigid WER (%) | Lattice WER (%) | Delta (pp) | Improved Utterances | Unchanged Utterances | Worsened Utterances |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in model_summary_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['rigid_wer_percent']:.3f} | {row['lattice_wer_percent']:.3f} | "
            f"{row['delta_wer_percent']:.3f} | {int(row['improved_utterances'])} | {int(row['unchanged_utterances'])} | "
            f"{int(row['worsened_utterances'])} |"
        )
    lines.append("")
    lines.append("## Fairness Outcome")
    lines.append(f"- Total utterances: {overall_summary['utterances']}")
    lines.append(f"- Models evaluated: {overall_summary['models']}")
    lines.append(f"- Total utterance-model pairs improved: {overall_summary['total_improved_pairs']}")
    lines.append(f"- Total utterance-model pairs unchanged: {overall_summary['total_unchanged_pairs']}")
    lines.append(f"- Total utterance-model pairs worsened: {overall_summary['total_worsened_pairs']}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Positive deltas indicate models that were likely unfairly penalized by rigid reference.")
    lines.append("- Near-zero deltas indicate models already aligned with robust reference wording.")
    lines.append("- Negative deltas (worsened cases) should be minimal; inspect those utterances to tune thresholds.")
    lines.append("")
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> None:
    if args.agreement_threshold < 1:
        raise ValueError("--agreement-threshold must be >= 1.")
    if args.alternative_support_threshold < 1:
        raise ValueError("--alternative-support-threshold must be >= 1.")
    if args.insertion_support_threshold < 1:
        raise ValueError("--insertion-support-threshold must be >= 1.")

    input_file = args.input_file.resolve()
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}. Provide a CSV with reference + model outputs."
        )

    work_dir = ensure_dir(args.work_dir.resolve())
    df = pd.read_csv(input_file)
    if args.reference_col not in df.columns:
        raise ValueError(f"Reference column `{args.reference_col}` not found in input file.")

    model_cols = detect_model_columns(
        df=df,
        reference_col=args.reference_col,
        id_col=args.id_col,
        model_cols_arg=args.model_cols,
    )
    if args.expected_model_count > 0 and len(model_cols) != args.expected_model_count:
        raise ValueError(
            f"Q4 expects exactly {args.expected_model_count} ASR model columns, but found {len(model_cols)}: "
            f"{model_cols}. Pass the correct five columns via --model-cols."
        )
    if len(model_cols) < 2:
        raise ValueError("Need at least 2 model columns for lattice consensus.")

    rows: list[dict[str, Any]] = []
    lattice_jsonl_path = work_dir / "lattice_bins_per_utterance.jsonl"
    with lattice_jsonl_path.open("w", encoding="utf-8") as fout:
        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Constructing lattices and scoring"):
            utterance_id = row[args.id_col] if args.id_col in df.columns else f"row_{row_idx}"
            reference = str(row[args.reference_col])
            ref_tokens = tokenize(reference)
            ref_len = max(1, len(ref_tokens))

            model_tokens_by_name: dict[str, list[str]] = {}
            for model_col in model_cols:
                model_tokens_by_name[model_col] = tokenize(str(row[model_col]))

            bins, lattice_meta = build_lattice(
                reference_tokens=ref_tokens,
                model_tokens_by_name=model_tokens_by_name,
                agreement_threshold=args.agreement_threshold,
                alternative_support_threshold=args.alternative_support_threshold,
                insertion_support_threshold=args.insertion_support_threshold,
            )

            fout.write(
                json.dumps(
                    {
                        "utterance_id": utterance_id,
                        "reference": reference,
                        "reference_tokens": ref_tokens,
                        "lattice_bins": serialize_bins(bins),
                        "lattice_meta": lattice_meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for model_col in model_cols:
                hyp_tokens = model_tokens_by_name[model_col]
                rigid_distance, sub, dele, ins, _ = align_tokens(ref_tokens, hyp_tokens)
                lattice_distance, _ = lattice_edit_distance(hyp_tokens, bins)

                rigid_wer = rigid_distance / ref_len
                lattice_wer = lattice_distance / ref_len
                delta = rigid_wer - lattice_wer

                rows.append(
                    {
                        "utterance_id": utterance_id,
                        "model": model_col,
                        "reference_length_tokens": ref_len,
                        "rigid_errors": rigid_distance,
                        "rigid_substitutions": sub,
                        "rigid_deletions": dele,
                        "rigid_insertions": ins,
                        "rigid_wer_percent": rigid_wer * 100.0,
                        "lattice_errors": lattice_distance,
                        "lattice_wer_percent": lattice_wer * 100.0,
                        "delta_wer_percent": delta * 100.0,
                        "improved": int(delta > 1e-9),
                        "unchanged": int(math.isclose(delta, 0.0, abs_tol=1e-9)),
                        "worsened": int(delta < -1e-9),
                    }
                )

    per_utt_df = pd.DataFrame(rows)
    per_utt_path = work_dir / "per_utterance_model_scores.csv"
    per_utt_df.to_csv(per_utt_path, index=False)

    model_summary_rows: list[dict[str, Any]] = []
    for model_name, frame in per_utt_df.groupby("model"):
        total_ref = float(frame["reference_length_tokens"].sum())
        rigid_errors = float(frame["rigid_errors"].sum())
        lattice_errors = float(frame["lattice_errors"].sum())
        rigid_wer = (rigid_errors / total_ref) * 100.0 if total_ref > 0 else 0.0
        lattice_wer = (lattice_errors / total_ref) * 100.0 if total_ref > 0 else 0.0
        model_summary_rows.append(
            {
                "model": model_name,
                "utterances": int(len(frame)),
                "rigid_wer_percent": round(rigid_wer, 3),
                "lattice_wer_percent": round(lattice_wer, 3),
                "delta_wer_percent": round(rigid_wer - lattice_wer, 3),
                "improved_utterances": int(frame["improved"].sum()),
                "unchanged_utterances": int(frame["unchanged"].sum()),
                "worsened_utterances": int(frame["worsened"].sum()),
            }
        )

    model_summary_df = pd.DataFrame(model_summary_rows).sort_values("model").reset_index(drop=True)
    model_summary_path = work_dir / "model_summary.csv"
    model_summary_df.to_csv(model_summary_path, index=False)

    overall_summary = {
        "utterances": int(df.shape[0]),
        "models": int(len(model_cols)),
        "total_pairs": int(per_utt_df.shape[0]),
        "total_improved_pairs": int(per_utt_df["improved"].sum()),
        "total_unchanged_pairs": int(per_utt_df["unchanged"].sum()),
        "total_worsened_pairs": int(per_utt_df["worsened"].sum()),
        "input_file": input_file.as_posix(),
        "reference_col": args.reference_col,
        "model_cols": model_cols,
        "expected_model_count": int(args.expected_model_count),
        "agreement_threshold": args.agreement_threshold,
        "alternative_support_threshold": args.alternative_support_threshold,
        "insertion_support_threshold": args.insertion_support_threshold,
    }
    (work_dir / "summary.json").write_text(json.dumps(overall_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pseudocode_md = build_pseudocode_markdown()
    (work_dir / "lattice_theory_pseudocode.md").write_text(pseudocode_md, encoding="utf-8")

    report_md = build_report_markdown(model_summary_df, overall_summary)
    report_path = work_dir / "question4_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print("\nCompleted Question-4 pipeline.")
    print(f"Artifacts written to: {work_dir}")
    print(f"Main report: {report_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
