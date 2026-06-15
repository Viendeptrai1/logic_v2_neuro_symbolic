# PROJECT: Neuro-Symbolic Solver for a Logic & Physics Reasoning Competition (LLM < 8B)

This folder is an experimental V2 track for Type 1 logic reasoning. It is intentionally separate from the current deploy pipeline and current LoRA experiment outputs.

## 0. Mission And Hard Constraints

- Goal: maximize competition score on two task types.
- Model size limit: any LLM must be < 8B params.
- Base model: `Qwen2.5-Coder-7B-Instruct`.
- This folder focuses on Type 1: logic / first-order reasoning.
- Type 2 physics numeric solving is handled by a separate, already-working module. Keep an interface stub for it, but do not rebuild it here.
- Core philosophy: the LLM only translates natural language into a formal logic representation.
- All reasoning, answer selection, premise selection, and explanation generation must be done by a deterministic symbolic engine.
- Never let the LLM guess the final answer except in an explicit fallback path.
- System accuracy is approximately `P(translation correct)`.

## 1. Inference I/O

### Input

```json
{
  "query_id": "T1_0031",
  "type": "type1",
  "query": "Based on the premises, which conclusion is correct?\nA. ...\nB. ...\nC. ...\nD. ...",
  "premises": ["If a manuscript is scanned at 600 dpi and ...", "..."],
  "options": ["A", "B", "C", "D"]
}
```

- For MCQ, option texts are embedded inside `query` as `A.`, `B.`, etc.
- For truth questions, `options` holds exact answer tokens such as `["Yes", "No", "Uncertain"]`.
- Always output one of the strings present in `options` verbatim.
- Do not invent `Unknown` if the options say `Uncertain`.

### Output

```json
{
  "query_id": "T1_0031",
  "answer": "B",
  "unit": "",
  "explanation": "Natural-language justification consistent with the proof.",
  "premises_used": [0, 1],
  "reasoning": {
    "type": "fol",
    "steps": ["Rule: ...", "Fact: ...", "Conclusion: ..."]
  }
}
```

- `premises_used` must be 0-based for competition output.

## 2. Provided Training Data

Source dataset:

```text
Data_Processed/Logic/Logic_Based_Educational_Queries.json
```

Expected local copy for this V2 track:

```text
logic_v2_neuro_symbolic/data/Logic_Based_Educational_Queries.json
```

Per-item fields:

- `premises-NL`: natural-language premises, translator input.
- `premises-FOL`: gold FOL for each premise, aligned by index.
- `questions`: one or two questions per item.
- `answers`: aligned to `questions`.
- `idx`: aligned to `questions`; gold premise indices, 1-based in the dataset.
- `explanation`: gold explanation per question.

Known dataset stats:

- 411 items.
- 808 total questions.
- 449 open Yes/No/Unknown questions.
- 359 MCQ questions with 3-4 options A-D.
- Premises per item: min 3, max 36, average about 11.

Important parser requirement:

- Two FOL notations appear: unicode-style and ASCII/functional style.
- The parser must accept both and normalize to one internal AST.

## 3. Logical Taxonomy To Support

Build and unit-test every item below.

- Universal quantifier: all / forall.
- Negation.
- Existential quantifier, including distinctness and counting.
- Conjunction in antecedents.
- Numeric and threshold predicates: `>= 80`, `< 50`, `= 100`, symbolic thresholds.
- Disjunction in antecedents or consequents.
- Biconditional.
- Nested function terms.
- Inequality / distinctness.

Required reasoning patterns:

- Multi-hop modus ponens chains.
- Contraposition.
- Hypothetical syllogism.
- Conditional conclusions.
- Universal instantiation.
- Existential reasoning.
- Disjunctive syllogism.
- Biconditional rewriting.
- Fewest-premises / minimal-support selection.
- Meta-questions asking which premises support a conclusion.

## 4. Architecture V2

Pipeline:

```text
Input
  -> A. Translate with Qwen+LoRA
       premises-NL -> premises-FOL AST
       question/options -> goal spec
  -> B. Normalize
       canonical predicate and constant names
       merge synonyms
       accept unicode and functional FOL notations
  -> C. Deterministic FOL reasoner
       decide entailment for each candidate
       extract proof object and minimal premise support
  -> D. Resolver
       map proof result to exact option token
  -> E. Output builder
       answer, premises_used, explanation, reasoning.steps
  -> F. Validate schema
       retry/fallback only when needed
```

Only Stage A uses the LLM.

## 5. Stage A Translator IR

The translator must output strict JSON:

```json
{
  "predicates": {
    "WT": "well-tested",
    "O": "optimized"
  },
  "constants": ["Atlas"],
  "premises_fol": [
    {"id": 0, "fol": "forall x. WT(x) -> O(x)"}
  ],
  "goal": {
    "kind": "mcq",
    "options": {
      "A": "forall x. (~O(x) -> ~WT(x))"
    }
  }
}
```

For truth questions:

```json
{
  "goal": {
    "kind": "truth",
    "statement": "forall x. WS(x) -> O(x)",
    "answer_tokens": ["Yes", "No", "Uncertain"]
  }
}
```

Key failure mode to fight:

- Predicate names must stay consistent across all premises and options inside one item.
- Enforce this with the declared `predicates` registry and Stage B reconciliation.

## 6. Stage C Reasoner

Preferred approaches:

1. Resolution-based FOL theorem prover with equality and simple arithmetic theory.
2. SMT offload for numeric comparisons and cardinality when needed.
3. Bounded forward/backward chaining only as a fast path for pure Horn items.

Truth-question decision:

- `prove(statement)` succeeds -> `Yes`.
- `prove(not statement)` succeeds -> `No`.
- neither succeeds within budget -> `Unknown` / `Uncertain`, mapped to exact token in `answer_tokens`.

MCQ decision:

- Evaluate each option formula.
- Usually choose the uniquely entailed option.
- For "fewest premises", choose the entailed option with minimal premise support.

Reasoner must return a proof object:

- ordered steps
- exact premise ids used
- minimal support when possible

## 7. Output Builder

- `premises_used` equals proof premise ids, converted to competition 0-based indexing.
- `explanation` and `reasoning.steps` are rendered deterministically from the proof object.
- Do not use free-form LLM explanation outside fallback.

## 8. Robustness

- Validate translator JSON schema.
- Retry Stage A on parse failure.
- Self-consistency: sample Stage A `k=3-5`, run deterministic stages for each, majority vote final token.
- Use winning sample proof for explanation.
- Fallback only when reasoner returns no proof, times out, votes tie, or translator repeatedly fails.
- Log every fallback.

## 9. Training Plan

Build:

```text
train/prepare_sft.py
train/train_lora.py
```

Training data conversion:

- Input: `premises-NL` plus question/options.
- Target: assistant strict IR JSON using gold `premises-FOL`.

LoRA starting config:

- rank `r=16-32`
- alpha `2r`
- targets: attention `q,k,v,o` and MLP projections
- learning rate `1e-4` to `2e-4`
- epochs `1-3`
- bf16
- packed sequences
- base model frozen

Evaluation split:

- Hold out a dev split stratified by construct.
- Every construct in the taxonomy must appear in dev.
- Add round-trip checks for translation quality.

## 10. Evaluation Harness

Build:

```text
eval/evaluate.py
eval/per_construct_report.py
```

Report:

- exact-match accuracy on `answers`
- `premises_used` F1 vs gold `idx`
- per-construct accuracy breakdown
- fallback rate
- translation parse-failure rate

Construct buckets:

- negation
- existential/counting
- numeric
- disjunction
- biconditional
- contraposition
- fewest-premises
- conditional conclusion

## 11. Acceptance Test Cases

Must pass:

1. Multi-hop Horn plus missing final condition.
2. Contraposition MCQ.
3. Biconditional: `FailTest <-> not PassTest`.
4. Existential counting with distinct entities.
5. Numeric threshold.
6. Disjunction antecedent.
7. Fewest-premises MCQ.
8. No entailment either way -> `Unknown` / `Uncertain`.

## 12. Suggested Repo Layout

Target mature layout:

```text
src/
  translate/
  fol/
  resolve/
  output/
  pipeline.py
train/
  prepare_sft.py
  train_lora.py
eval/
  evaluate.py
  per_construct_report.py
data/
  Logic_Based_Educational_Queries.json
```

Current bootstrap layout:

```text
logic_v2_neuro_symbolic/
  PROJECT_CONTEXT.md
  neuro_symbolic_logic_v2.ipynb
  data/
```

## 13. Must-Not

- Do not let the LLM produce final answer, `premises_used`, or explanation outside explicit fallback.
- Do not treat numeric thresholds as opaque string labels.
- Do not hardcode topic-specific rules.
- Do not output an answer token that is not present in the query's `options`.

## 14. Deliverables

- working `pipeline.py`
- training scripts
- evaluation harness with per-construct report
- short README explaining inference and eval reproduction

