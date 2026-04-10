# BinView

BinView is a VS Code extension for inspecting algorithm output binaries and comparing them with golden data.

Current capabilities:

- Raw binary files with manual dtype and endianness selection.
- NumPy `.npy` files with automatic dtype and shape parsing.
- PyTorch saved tensor files through the VS Code Python interpreter selected by the Python extension.
- Readable 8-values-per-line viewer with virtual scrolling for large files.
- In-page jump box for direct line or element-index navigation with highlight.
- Responsive 8-cells-per-row layout that shrinks cells to fit the available width.
- In-page Python console for persistent torch/tensor interaction, plus reshape/slice helpers.
- Output-vs-golden compare mode with per-cell difference highlight.

Notes:

- Raw files are displayed as a flat 1-D sequence.
- `.npy` files using Fortran order are shown in stored linear order.
- Torch support is intentionally optional. If the Python extension or interpreter is unavailable, raw and NumPy viewing still works.

Usage:

- Right click a `.bin`, `.npy`, `.pt`, `.pth`, or `.ckpt` file in Explorer and choose `Reopen With...`, then select `BinView`.
- In the same picker, choose `Set as Default` if you want that file pattern to always open in BinView.
- You can also use `BinView: Open Binary File` from the command palette or Explorer context menu for one-off inspection.
- Use `BinView: Compare Output With Golden` to open two files in the side-by-side compare viewer.
- In the viewer toolbar, enter `#128` or an index expression such as `#64*2` to jump to an element, or `L16` to jump to display line 16.
- In the Python Console, `tensor`, `torch`, and `math` are available. Commands keep their context between runs.
- Use `Python / Torch` for free-form expressions or statements, or switch to `Reshape`, `Slice`, or `Reset` helper mode.

Recommended VS Code settings if you want these formats to open in BinView by default:

```json
"workbench.editorAssociations": {
  "*.bin": "binView.viewer",
  "*.npy": "binView.viewer",
  "*.pt": "binView.viewer",
  "*.pth": "binView.viewer",
  "*.ckpt": "binView.viewer"
}
```

Raw binary settings:

- `binView.raw.promptOnOpen`: when enabled, BinView asks for dtype and endianness every time a raw file is opened.
- `binView.raw.defaultDtype`: default dtype used for raw files when prompting is disabled.
- `binView.raw.defaultEndianness`: default byte order used for raw files when prompting is disabled.

Example:

```json
"binView.raw.promptOnOpen": false,
"binView.raw.defaultDtype": "float32",
"binView.raw.defaultEndianness": "little"
```

Development:

```bash
npm install
npm run compile
```

Generate sample NumPy data for local testing:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy
.venv/bin/python scripts/generate_numpy_samples.py
```

This writes a few `.npy` files into `samples/`, including:

- Numeric grids across `float32`, `int16`, and `uint8`
- A `float64` sample with `NaN`, `Infinity`, and `-0`
- A Fortran-order tensor
- A golden/output pair for compare mode

Packaging:

```bash
npm run package
```

GitHub automation:

- Push and pull request builds run on GitHub Actions.
- Every CI run uploads a packaged `.vsix` artifact.
- Pushing a `v*` tag also creates or updates a GitHub Release and attaches the `.vsix`.
