# Q4 Lattice Method (Theory + Pseudocode)

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
