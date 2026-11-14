# Arden Simulator

An educational simulator for the course **Automata Theory and Computability** that
converts a finite automaton (DFA, NFA or ε-NFA) into an equivalent regular expression
using **Arden's theorem**. It writes one left-linear equation per state, solves the
system by substitution and Arden's rule while recording every step, simplifies the
result, checks it against the automaton on every string up to a chosen length, and lets
you step through, compare, and export the full derivation, from a Streamlit web app or
the command line.

---

## 1. Theory

### Arden's Theorem

Let $P$ and $Q$ be regular expressions over $\Sigma$. If $\varepsilon \notin L(P)$
($P$ is **not nullable**), then the equation

$$R = Q + RP$$

has the **unique** solution

$$R = QP^{*}.$$

### Proof sketch

**$QP^*$ is a solution.** Substitute it into the right-hand side:

$$Q + (QP^{*})P = Q(\varepsilon + P^{*}P) = QP^{*},$$

because $\varepsilon + P^{*}P = P^{*}$ (a string in $P^*$ is either empty or a string of
$P^*$ followed by one more block from $P$).

**Every solution contains $QP^*$.** Let $R$ satisfy $R = Q + RP$. Then $Q \subseteq R$,
and if $QP^k \subseteq R$ then $QP^{k+1} \subseteq RP \subseteq R$. By induction
$QP^k \subseteq R$ for all $k$, so $QP^{*} \subseteq R$.

**Every solution is contained in $QP^*$ (this is where $\varepsilon \notin P$ is needed).**
Unrolling the equation $k$ times gives

$$R = Q + QP + QP^2 + \dots + QP^{k} + RP^{k+1}.$$

Take any $w \in R$ with $|w| = n$ and unroll $k = n$ times. Every string in $P$ has
length at least 1, so every string in $RP^{n+1}$ has length at least $n+1 > |w|$. Hence
$w \notin RP^{n+1}$, so $w \in QP^{i}$ for some $i \le n$, i.e. $w \in QP^{*}$.

**Why the precondition matters.** If $\varepsilon \in P$, uniqueness fails. With
$P = \varepsilon$ the equation reads $R = Q + R$, which every $R \supseteq Q$ satisfies
(e.g. $R = \Sigma^{*}$).

### The method (FA → RE)

1. **Write one equation per state.** The right-hand side of state $q_i$ sums its
   *incoming* transitions: each move $q_j \xrightarrow{a} q_i$ adds the term $q_j a$.
   The start state also gets the constant $\varepsilon$.
2. **Solve by substitution.** Eliminate the states one at a time. Whenever an equation
   has the form $q = Q + qP$, apply Arden: $q = QP^{*}$. Because the automaton has no
   ε-transitions, every coefficient $P$ denotes only non-empty strings, so the
   precondition always holds (the solver still checks it).
3. **Combine.** The regular expression is the union of the solutions of all **final**
   states; with no final states it is $\varnothing$.

Equations are kept in **left-linear form**, `q = q₁·C₁ + … + qₖ·Cₖ + K`. Substituting
`qᵢ := Σ qₖ·Cₖ + K` into a term `qᵢ·a` gives `Σ qₖ·(Cₖ·a) + K·a`, so the form is preserved.

Different elimination orders give **different-looking but equivalent** regular
expressions. The app lets you choose the order and compare all of them.

---

## 2. Worked example: strings ending in `01`

`examples/ends_with_01.json` is a 3-state DFA over `{0, 1}`. `q0` is the start state,
`q2` is the only final state, and the moves are `q0 -0-> q1`, `q0 -1-> q0`, `q1 -0-> q1`,
`q1 -1-> q2`, `q2 -0-> q1` and `q2 -1-> q0`. The output below is taken verbatim from
`python -m core.pipeline examples/ends_with_01.json --trace` (default order
`q2, q1, q0`).

**Initial equations**

```
q0 = q0·1 + q2·1 + ε
q1 = q0·0 + q1·0 + q2·0
q2 = q1·1
```

**Steps**

| # | Step | What happens | Equation afterwards |
|---|------|--------------|---------------------|
| 1 | Substitute `q2` | `q2 = q1·1` goes into `q1` and `q0` (`q2·0` becomes `q1·10`, merged with `q1·0`) | `q1 = q0·0 + q1·(0+10)`, `q0 = q0·1 + q1·11 + ε` |
| 2 | Arden on `q1` | `P = 0+10`, `Q = q0·0` ⇒ `q1 = (q0·0)·(0+10)*` | `q1 = q0·0(0+10)*` |
| 3 | Substitute `q1` | into `q0` | `q0 = q0·(0(0+10)*11+1) + ε` |
| 4 | Arden on `q0` | `P = 0(0+10)*11+1`, `Q = ε` | `q0 = (0(0+10)*11+1)*` |
| 5 | Back-substitute | `q0` into `q1` | `q1 = (0(0+10)*11+1)*0(0+10)*` |
| 6 | Back-substitute | `q1` into `q2` | `q2 = (0(0+10)*11+1)*0(0+10)*1` |
| 7 | Combine finals | only `q2` is final | |

**Result**

```
RE:    (0(0+10)*11+1)*0(0+10)*1
LaTeX: (0(0+10)^{*}11+1)^{*}0(0+10)^{*}1

PASSED: the automaton and the RE agree on all 511 strings up to length 8.
```

This is equivalent to the textbook answer `(0+1)*01`. A different elimination order gives
a different-looking expression; the **Compare orders** tab shows all six.

---

## 3. Pipeline

```
 dict / JSON text / .json file
              │
              ▼
 ┌─────────────────────────┐  pydantic schema, convention checks
 │ Validation & parsing     │  → Automaton (frozen)              core/automaton.py
 └────────────┬────────────┘
              ▼
 ┌─────────────────────────┐  ε-removal, unreachable- and
 │ Normalization            │  dead-state cleanup                core/automaton.py
 └────────────┬────────────┘
              ▼
 ┌─────────────────────────┐  one left-linear equation
 │ Equation builder         │  per state                         core/equations.py
 └────────────┬────────────┘
              ▼
 ┌─────────────────────────┐  Arden + substitution,
 │ Solver                   │  back-substitution, TraceSteps     core/solver.py
 └────────────┬────────────┘        ▲
              │                     │ simplify(), nullable()     core/regex_ast.py
              ▼                     │ (used throughout)
 ┌─────────────────────────┐
 │ Verifier                 │  FA vs RE on every string ≤ n      core/verifier.py
 └────────────┬────────────┘
              ▼
       PipelineResult  ── run_pipeline() facade + CLI ──         core/pipeline.py
         │          │
         ▼          ▼
   Streamlit UI   Markdown / LaTeX reports
     ui/            export/
```

`core/` is pure Python. It never imports Streamlit, Graphviz, `ui/` or `export/`. The UI and
the exporter reach the core only through `core/pipeline.py` and the dataclasses it
returns.

---

## 4. Project structure

```
arden-simulator/
├── core/
│   ├── __init__.py          # exception hierarchy rooted at ArdenError
│   ├── automaton.py         # FA model, pydantic schema, validation, ε-removal, cleanup, simulation
│   ├── regex_ast.py         # regex AST, smart constructors, simplifier, nullable(), printers
│   ├── equations.py         # Equation / EquationSystem, builder, substitution, printers
│   ├── solver.py            # Arden + substitution, elimination orders, TraceStep events
│   ├── verifier.py          # bounded shortlex enumeration, shortest counterexample
│   └── pipeline.py          # run_pipeline() facade, list_examples(), CLI
├── ui/
│   ├── __init__.py
│   ├── app.py               # Streamlit entry point (streamlit run ui/app.py)
│   └── components.py        # diagram, equation LaTeX, step badge, error/verification boxes
├── export/
│   ├── __init__.py
│   ├── report.py            # render_markdown() / render_latex() with Jinja2
│   └── templates/
│       ├── report.md.j2     # Markdown report template
│       └── report.tex.j2    # LaTeX report template (\BLOCK{} / \VAR{} delimiters)
├── examples/                # 10 textbook automata (DFA, NFA, ε-NFA, edge cases)
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # shared hypothesis strategy (random small NFAs)
│   ├── test_smoke.py        # imports and exception hierarchy
│   ├── test_regex_ast.py    # constructors, printers, simplifier (+ property test)
│   ├── test_automaton.py    # schema validation, simulation, preprocessing (+ property test)
│   ├── test_equations.py    # builder, substitution, printers
│   ├── test_solver.py       # Arden elimination, all orders, trace (+ property test)
│   ├── test_verifier.py     # enumeration, counterexamples, budget
│   ├── test_pipeline.py     # facade, order adjustment, errors, CLI
│   ├── test_edge_cases.py   # edge-case matrix through the whole pipeline
│   ├── test_export.py       # reports, JSON round-trip, pdflatex compile
│   └── test_ui.py           # Streamlit AppTest flows and component unit tests
├── CLAUDE.md                # project rules and build order
├── README.md
├── pyproject.toml           # project metadata + pytest config
└── requirements.txt
```

---

## 5. Setup

**Requirements:** Python **3.11+**.

```bash
# from the project root
python -m venv .venv

# activate it
.venv\Scripts\activate            # Windows (PowerShell / cmd)
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

**Graphviz (optional).** The `graphviz` Python package only builds the DOT text of the
diagrams, and Streamlit draws them in the browser, so **the app works without the
Graphviz system package**. Install it only if you want to render DOT files to images
yourself:

| OS | Command |
|----|---------|
| Windows | `winget install graphviz` (or the installer from graphviz.org; add `bin` to PATH) |
| macOS | `brew install graphviz` |
| Debian / Ubuntu | `sudo apt install graphviz` |
| Fedora | `sudo dnf install graphviz` |

**LaTeX (optional).** Compiling the exported `.tex` reports locally needs a TeX
distribution with `pdflatex` (MiKTeX, TeX Live or MacTeX). Without one, upload the file
to Overleaf.

---

## 6. Usage

### Streamlit app

```bash
streamlit run ui/app.py
```

1. **Sidebar → Automaton.** Pick an *Example*, *Upload JSON*, or *Build manually* (states,
   alphabet, start, finals, and an editable transition table; press *Load into simulator*).
2. **Elimination order.** *Default* (reverse declaration order, start state last) or
   *Custom* (pick every state in the order to eliminate them).
3. **Verification length** (4–12), then **Run simulation**.

Tabs: **Step-by-step** (First/Prev/Next/Last, slider, Play; the diagram highlights the
state being solved, and the before/after equations highlight what changed),
**Preprocessing**, **Result & verification** (closed forms, sample strings, *Try a
string*), **Compare orders** (≤ 4 states), and **Export**.

### Command line

```bash
python -m core.pipeline examples/ends_with_01.json
python -m core.pipeline examples/ends_with_01.json --order q0,q1,q2 --verify-length 10 --trace
```

| Flag | Meaning |
|------|---------|
| `file` | path to the automaton JSON (required) |
| `--order q2,q1,q0` | comma-separated elimination order (default: reverse declaration order, start last) |
| `--verify-length N` | check every string up to length `N` (default 8) |
| `--trace` | print every step's explanation and equations |

The command prints the preprocessing summary, the initial equations, the trace (with
`--trace`), the final RE in text and LaTeX, and the verification result.

**Exit codes:** `0` verification passed, `1` verification failed, `2` input error (bad
JSON, invalid automaton, invalid order).

From Python:

```python
from core.pipeline import run_pipeline

result = run_pipeline("examples/ends_with_01.json", order=None, verify_length=8)
print(result.regex_text, result.verification.passed)
```

### JSON schema

```json
{
  "name": "strings ending in 01",
  "description": "Binary strings whose last two symbols are 0 then 1.",
  "expected_language_note": "Equivalent to (0+1)*01.",
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start": "q0",
  "finals": ["q2"],
  "transitions": [
    {"from": "q0", "symbol": "0", "to": "q1"},
    {"from": "q0", "symbol": "1", "to": "q0"},
    {"from": "q1", "symbol": "0", "to": "q1"},
    {"from": "q1", "symbol": "1", "to": "q2"},
    {"from": "q2", "symbol": "0", "to": "q1"},
    {"from": "q2", "symbol": "1", "to": "q0"}
  ]
}
```

- `states`, `alphabet`, `start` are required. `finals` and `transitions` default to empty,
  `name` defaults to "Untitled automaton", and `description` / `expected_language_note`
  are optional. Any other key is rejected, which catches typos such as `"final"`.
- Alphabet symbols are **single characters**, excluding `+ * ( ) | ε ∅` and whitespace.
- An ε-transition uses the symbol `"ε"` or `"eps"`.
- Several transitions from the same state on the same symbol are allowed (NFA).

### Exporting reports

In the app's **Export** tab:

- **Download Markdown report**: the theorem, the input automaton with its transition
  table, preprocessing, the initial equations, the elimination order, every step, the
  final RE, and verification with sample strings.
- **Download LaTeX report**: the same content as a complete `article` document
  (amsmath, amssymb, booktabs; `align*` for equation systems). It compiles in Overleaf or
  with `pdflatex`.
- **Download automaton JSON**: save an automaton (e.g. one built manually) to upload it
  again later.

From Python:

```python
from pathlib import Path
from core.pipeline import run_pipeline
from export.report import render_latex, render_markdown

result = run_pipeline("examples/ends_with_01.json")
Path("reports").mkdir(exist_ok=True)          # reports/ is git-ignored
Path("reports/ends_with_01.md").write_text(render_markdown(result), encoding="utf-8")
Path("reports/ends_with_01.tex").write_text(render_latex(result), encoding="utf-8")
```

---

## 7. Testing and verification strategy

```bash
pytest                                                    # whole suite
pytest --cov=core --cov=export --cov=ui --cov-report=term-missing
ARDEN_SOLVER_EXAMPLES=1000 pytest -k random_nfas          # one-off stress run (bash syntax)
```

(In PowerShell: `$env:ARDEN_SOLVER_EXAMPLES=1000; pytest -k random_nfas`.)

- **Unit tests** for every module: the smart-constructor rules, each simplification
  rule (before/after), printers, every schema validation rule (with the message
  checked), ε-closure and preprocessing, hand-checked substitutions, trace structure, the
  shortest counterexample, report content, and the Streamlit app through
  `streamlit.testing.v1.AppTest`.
- **Property-based tests** (Hypothesis):
  - `simplify()` preserves the language of random regexes, never grows them, and is
    idempotent;
  - `normalize()` preserves the language of random ε-NFAs;
  - the whole pipeline, run on random NFAs with ε-moves, produces an RE that agrees
    with the automaton.
- **Edge-case matrix** (`tests/test_edge_cases.py`): no finals, nullable start,
  single-state automata, self-loops everywhere, unreachable and dead states, multiple
  finals, nondeterminism, ε-cycles, a 3-symbol alphabet, and a 6-state DFA.
- **Bounded verification.** Every pipeline run compares the RE with the *original*
  automaton on **every** string over its alphabet up to length *n*, in shortlex order.
  The first disagreement is therefore a shortest counterexample.

> **Limitation:** bounded enumeration shows the automaton and the RE agree on all
> strings up to length *n*. It is strong evidence, **not a proof of equivalence**: two
> languages could in principle first differ on a longer string. A full equivalence check
> would require converting the RE back to an automaton and comparing minimal DFAs,
> which is outside this project's scope.

---

## 8. Known limitations

- **Compare orders** is limited to automata with **≤ 4 states** after preprocessing
  (4! = 24 orders). Larger automata can still be solved with any custom order.
- **Expression blow-up guard.** If any coefficient grows beyond **5000 AST nodes**
  during solving, the solver stops with "expression too large; try a different
  elimination order or a smaller automaton". Substitution can grow expressions
  exponentially in the worst case.
- **Simplification is heuristic.** The simplifier applies a fixed set of
  language-preserving identities. Results are correct but not minimal: ε-NFAs in
  particular can give long expressions such as `a*(ε+((a+b)b*(b+c)+a+b+c)c*+(a+b)b*)`
  for the language `a*b*c*`.
- **Single-character symbols.** Alphabet symbols must be single characters and cannot be
  `+ * ( ) | ε ∅` or whitespace.
- **Verification budget.** At most 200,000 strings are checked. For large alphabets, the
  length is reduced automatically and the result says so.
- **Very long expressions** may run past the margin in the LaTeX report (the PDF still
  compiles).

---

## 9. Credits

- **Course:** Automata Theory and Computability, *‹university / term›*
- **Instructor:** *‹name›*
- **Author(s):** *‹name(s), student ID(s)›*
- **Repository:** *‹link›*

*‹Add any acknowledgements or references here, e.g. the textbook used in the course.›*
