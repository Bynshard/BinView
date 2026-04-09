# BinView

BinView is a VS Code extension for inspecting algorithm output binaries and comparing them with golden data.

Current capabilities:

- Raw binary files with manual dtype and endianness selection.
- NumPy `.npy` files with automatic dtype and shape parsing.
- PyTorch saved tensor files through the VS Code Python interpreter selected by the Python extension.
- Readable 8-values-per-line viewer with virtual scrolling for large files.
- `Ctrl+G` line jump inside the BinView viewer.
- Optional torch-side reshape and slice operations when Python and `torch` are available.
- Output-vs-golden compare mode with per-cell difference highlight.

Notes:

- Raw files are displayed as a flat 1-D sequence.
- `.npy` files using Fortran order are shown in stored linear order.
- Torch support is intentionally optional. If the Python extension or interpreter is unavailable, raw and NumPy viewing still works.

Development:

```bash
npm install
npm run compile
```

Packaging:

```bash
npm run package
```

GitHub automation:

- Push and pull request builds run on GitHub Actions.
- Every CI run uploads a packaged `.vsix` artifact.
- Pushing a `v*` tag also creates or updates a GitHub Release and attaches the `.vsix`.
