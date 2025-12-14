CFU Playground — Quickstart

This repository contains the AAML CFU Playground project sources and utilities. The steps below show a concise quickstart for using the repository on a system where the CFU Playground environment is already installed and configured.

Prerequisites

- Git installed and configured
- A working CFU Playground environment (follow the repository's `scripts/setup` if needed)

Quickstart

1. Clone the repository and enter its directory (replace `<repo-url>` and the folder name if different):

```bash
gh repo clone masamune-prog/AAML_Final_Project
cd AAML_Final_Project
```

2. Run the repository setup script (if you haven't already set up the environment):

```bash
./scripts/setup
```

3. Use or switch to a branch. Examples:

- Checkout an existing remote branch named `simd` locally:

```bash
git fetch origin
git checkout -b simd origin/simd
```

- Or, to switch to a local branch that already exists:

```bash
git checkout simd
```

4. Fetch LFS files and build / run the project as usual.

Clone and fetch LFS files in one sequence:

```bash
cd AAML_Final_Project
git lfs install
git lfs pull
git lfs checkout
cd proj/AAML-2025-Project
make prog EXTRA_LITEX_ARGS="--cpu-variant=perf+cfu"&& make load

```
