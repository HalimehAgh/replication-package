# Manual Coding Scheme — ML-Specific Code Smell Lifecycle Study

This document defines the coding scheme used for manual inspection of commits
in which ML-specific code smells were introduced or removed. It is part of
the replication package for the ICSE paper on ML-specific code smell lifecycles
and maintainability associations.

---

## Coding Process

Two coders independently coded **80 unique commits**, each commit across all 8 dimensions.
In total, **88 commit-dimension observations** were coded (some commits appear across
multiple dimensions). Inter-rater reliability was assessed before discussion
using Cohen's κ on the 88 × 8 = **704 coding decisions**.

---

## Inter-Rater Reliability

| Dimension | Agreement | κ | Interpretation |
|---|---|---|---|
| `pure_smell_fix` | 100.0% | 1.000 | Almost Perfect |
| `error_handling` | 97.7% | 0.952 | Almost Perfect |
| `type_hints` | 89.8% | 0.762 | Substantial |
| `confound_type` | 76.1% | 0.641 | Substantial |
| `defensive_checks` | 80.7% | 0.611 | Substantial |
| `documentation` | 78.4% | 0.568 | Moderate |
| `method_extraction` | 77.3% | 0.506 | Moderate |
| `refactored_nearby` | 73.9% | 0.482 | Moderate |
| **Overall** | **84.2%** | **0.690** | **Substantial** |

Interpretation scale: Landis & Koch (1977). Disagreements (n = 111) were
resolved through discussion between the two coders. The final resolved codings
are in `final_coding_commits.csv`.

The lower agreement on `refactored_nearby` (κ = 0.482) and `method_extraction`
(κ = 0.506) reflects the inherent subjectivity in assessing code changes in
the vicinity of a smell and distinguishing relocation of existing code from
all-new additions.

---

## Binary Columns (Yes / No)

### `pure_smell_fix`
Mark **YES** only if the commit **exclusively** fixes the smell and nothing
else — no surrounding refactoring, no new methods, no style changes. Even one
unrelated line change makes this **NO**. In practice this will almost always
be **NO**.

| YES Example | NO Example |
|---|---|
| Old code: `df.sort_values()` called but result not saved (in-place misuse). New code: `df = df.sort_values()`. No other changes in the diff. | Fix is surrounded by a new helper method or a refactored loop → NO |

---

### `method_extraction`
Mark **YES** if new named functions or methods are created that did not exist
before, **and** some code was **moved** into them from existing functions.
**Both** conditions required: (1) new named function/method added **AND**
(2) existing code relocated into it. Adding a new class or entirely new
standalone function with all-new code does **NOT** count.

| YES Example | NO Example |
|---|---|
| A long inline block inside `compute()` is extracted into a new helper `_validate_inputs()`. The original `compute()` now calls `_validate_inputs()`. | A brand new class `ScoreBasedModel` is added with all-new code (nothing moved from existing functions) → NO |

**Dataset example (YES):** Commit `084f545f` (mindsdb/mindsdb,
`columns_and_datatype_not_explicitly_set`) — existing model-loading logic was
moved from `from_pretrained_model()` into a new `load_model()` helper as part
of an architectural rewrite. `method_extraction = Yes`.

**Dataset example (NO):** Commit `8e9401fb`
(analyticalmindsltd/smote-variants, `in_place_apis_misused`) — only
formatting and style changes; no new functions created. `method_extraction = No`.

---

### `error_handling`
Mark **YES** if **new** `try/except` blocks, `raise` statements, or `except`
clauses are **added** in this commit that were not present before. Simply
retaining existing error handling does not count. If existing `try/except` is
**removed** and nothing is added, mark **NO**.

| YES Example | NO Example |
|---|---|
| New: `try: df.to_parquet(...) except Exception: df.to_csv(...)`. This `try/except` did not exist in the before version. | The old code already had `try/except` and the new code keeps it unchanged → NO |

**Dataset example (YES):** Commit `084f545f` (mindsdb/mindsdb) — new
`raise ValueError` and `raise RuntimeError` statements added throughout the
rewritten module. `error_handling = Yes`.

---

### `type_hints`
Mark **YES** if Python type annotations are explicitly **added** to
function/method signatures (parameters or return types) in this commit.
Annotations must be in the function signature (e.g. `def f(x: int) -> str`).
Type hints inside docstrings or comments do **NOT** count. Dataclass field
types count.

| YES Example | NO Example |
|---|---|
| Before: `def process(self, files, batch_size=1)`. After: `def process(self, files: List[str], batch_size: int = 1) -> List[str]`. | A docstring says `'Args: files (list)'` but the signature has no annotations → NO |

**Dataset example (YES):** Commit `19367c9c` (NVIDIA/NeMo,
`pytorch_call_method_misused`) — new methods added with fully typed signatures
(e.g. `def forward(self, input_ids: torch.Tensor) -> torch.Tensor`).
`type_hints = Yes`.

**Dataset example (NO):** Commit `82828282` — new methods added but signatures
have no annotations. `type_hints = No`.

---

### `documentation`
Mark **YES** if **new** docstrings or inline comments are **added** in this
commit. A docstring must be a string literal immediately after a `def`/`class`
statement. Inline comments start with `#`. Simply reformatting or rewording
existing docstrings does **NOT** count — new text must be added.

| YES Example | NO Example |
|---|---|
| A new method `_load_sdf_file()` is added with a docstring `'Load SDF file into dataframe.'` that did not exist before. | An existing docstring has a typo fixed (`discription` → `description`) → NO |

**Dataset example (YES):** Commit `317a6c74` (streamlit/streamlit,
`columns_and_datatype_not_explicitly_set`) — new inline `#` comments added
throughout the test function body. `documentation = Yes`.

**Dataset example (NO):** Commit `8a5d7499`
(ROCmSoftwarePlatform/tensorflow-upstream) — test methods commented out; no
new docstrings or comments added. `documentation = No`.

---

### `refactored_nearby`
Mark **YES** **only** if code that is **not** directly related to the smell is
modified in the **same function or class** where the smell appears. The key
test: *would this change exist even if the smell did not?* If the answer is
yes, mark **YES**. This column is specifically about changes in the
**vicinity** of the smell, not anywhere in the file.

| YES Example | NO Example |
|---|---|
| The smell is in `predict()`. The same `predict()` function also gets a new `elif` branch for a new mode. The branch has nothing to do with the smell → YES. | A completely separate function in the same file is rewritten, but the function containing the smell is only changed for the smell fix → NO |

**Dataset example (YES):** Commit `825c227c` (ray-project/ray,
`in_place_apis_misused`) — the class containing the smell was substantially
restructured with new methods unrelated to the smell. `refactored_nearby = Yes`.

**Dataset example (NO):** Commit `03cb911e` (calico/basenji,
`memory_not_freed`) — only a docstring added nearby; the smell-containing
function was not otherwise modified. `refactored_nearby = No`.

---

### `defensive_checks`
Mark **YES** if **new** input validation or safety checks are **added** in
this commit — specifically: `if/raise` patterns checking preconditions,
`assert` statements validating inputs, `.get()` replacing direct dict access
with defaults, `None` checks before use, or bounds checking. Must be
**new**, not pre-existing.

| YES Example | NO Example |
|---|---|
| New: `if weights.ndim != 1: raise ValueError('...')`. This check did not exist before. | An existing `assert` statement is retained unchanged → NO |

**Dataset example (YES):** Commit `084f545f` (mindsdb/mindsdb) — multiple new
`if ... is None: raise ValueError(...)` guards added to the rewritten module.
`defensive_checks = Yes`.

**Dataset example (NO):** Commit `82828282` — no new validation checks; only
structural changes to the training loop. `defensive_checks = No`.

---

## Nominal Column

### `confound_type`
Assign **one** of five values. Ask: what is the **primary driver** of the
metric change — not the smell fix, but everything else in the commit?

| Value | Definition |
|---|---|
| `pure_fix` | Only the smell was addressed, nothing else changed. |
| `feature_addition` | New end-user functionality was added (new API, new mode, new algorithm). The smell fix is a side effect. |
| `architectural_rewrite` | The module's design was fundamentally restructured (class hierarchy changed, methods reorganised, design pattern changed). No major new functionality — the restructuring itself is the point. |
| `style_normalization` | The dominant change is formatting, naming, PEP 8 compliance, or reformatting with no structural or functional change. |
| `test_refactoring` | The smell is in test code and the dominant change is restructuring the tests (fixtures, parametrisation, test organisation). |

**Decision rule:** If both a rewrite AND new features are present, ask which
drove the metric change more. New classes/methods with new algorithms →
`feature_addition`. Reorganising existing code into a different structure →
`architectural_rewrite`.

| Value | Dataset Example |
|---|---|
| `architectural_rewrite` | Commit `084f545f` (mindsdb/mindsdb) — entire module restructured into new class hierarchy; same functionality, different design. |
| `feature_addition` | Commit `19367c9c` (NVIDIA/NeMo) — three new model modes added; smell removed as side effect. |
| `style_normalization` | Commit `8e9401fb` (analyticalmindsltd/smote-variants) — dominant change is PEP 8 formatting and naming; no structural change. |
| `test_refactoring` | Commit `317a6c74` (streamlit/streamlit) — e2e test restructured with new fixtures; smell in test code removed as side effect. |

---

## Summary Table

| Column | Type | Question |
|---|---|---|
| `pure_smell_fix` | Binary | Was the smell the **only** thing changed? |
| `method_extraction` | Binary | Was existing code **moved** into a new named function? |
| `error_handling` | Binary | Were new `try/except` or `raise` statements **added**? |
| `type_hints` | Binary | Were type annotations **added** to function signatures? |
| `documentation` | Binary | Were new docstrings or `#` comments **added**? |
| `refactored_nearby` | Binary | Was unrelated code modified **in the same function/class** as the smell? |
| `defensive_checks` | Binary | Were new input validation or safety checks **added**? |
| `confound_type` | Nominal | What was the **primary driver** of the commit (besides the smell)? |
