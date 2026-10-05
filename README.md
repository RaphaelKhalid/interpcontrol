# interpcontrol

Interpretability tools as measurable add-ons to AI-control protocols.

- `em/` — main experiment: does a training-free J-lens readout of the untrusted model add safety to a weak trusted monitor? (execute → replace/audit flips; pre-emption; training-free white-box evasion round). Untrusted model: the public emergent-misalignment insecure-code organism.
- `hlens/` — H-lens pilot (second-order J-lens) on Qwen3.5-0.8B; see `hlens/RESULTS.md`.
- `dash/` — local OLED live dashboard (`python dash/server.py 8796`).
- `PROPOSAL-*.md`, `BRIEF-experiment.md`, `REVIEW-critiques.md` — earlier APPS-version design docs.

Preregistration lives in `em/prereg/` and is committed and hashed before any GPU run.
