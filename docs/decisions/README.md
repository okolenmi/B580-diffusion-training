# Decision records

Short records of decisions that a future reader would otherwise have to
re-derive, or -- worse -- would assume away.

Each one is four things and no more:

* **Context** -- what forced a choice.
* **Decision** -- what was chosen.
* **Consequences** -- what that costs, stated plainly.
* **What pins it** -- the test that fails if the decision is quietly
  reversed. A decision with no test is a comment that will be edited.

Written because the alternative was the same reasoning scattered across
code comments, where the *why* is separated from the *what* by whatever
distance the reader happens to look.

## Index

| # | Decision | Cost accepted |
|---|---|---|
| [0001](0001-no-authentication.md) | No authentication; Host/Origin guard instead | No protection against a non-browser client on the network |
| [0002](0002-graph-training-in-process.md) | Graph training runs in the API process | A device fault takes the server with it |
| [0003](0003-adopt-surviving-trainers.md) | Adopting trainers that outlived a restart | The supervisor watches a pid it did not spawn |
| [0004](0004-b580-only.md) | Intel Arc B580 only; `xpu` hardcoded | Nothing runs on another accelerator |

Numbers these documents assert are checked by `scripts/check_docs.py`
against the code, so they cannot drift silently.