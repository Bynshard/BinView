import * as path from 'node:path';
import * as vscode from 'vscode';
import type { BinarySource, SourceMetadata } from './dataSource';
import type { DisplayScalar } from './dtypes';

const ELEMENTS_PER_LINE = 8;

interface ViewerLine {
  lineNumber: number;
  startIndex: number;
  left: Array<DisplayScalar | null>;
  right?: Array<DisplayScalar | null>;
  different?: boolean;
}

interface ViewerState {
  rowsPerLine: number;
  totalLines: number;
  left: SourceMetadata;
  right?: SourceMetadata;
  canTransform: boolean;
  title: string;
}

export class ViewerSession implements vscode.Disposable {
  private readonly panel: vscode.WebviewPanel;
  private leftMetadata: SourceMetadata | undefined;
  private rightMetadata: SourceMetadata | undefined;
  private disposed = false;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly leftSource: BinarySource,
    private readonly rightSource: BinarySource | undefined,
    private readonly onFocus: (session: ViewerSession | undefined, source: ViewerSession) => void
  ) {
    this.panel = vscode.window.createWebviewPanel('binView.viewer', 'BinView', vscode.ViewColumn.Active, {
      enableScripts: true,
      retainContextWhenHidden: true
    });

    this.panel.onDidDispose(() => {
      void this.dispose();
    });

    this.panel.onDidChangeViewState((event) => {
      if (event.webviewPanel.active) {
        this.onFocus(this, this);
        void this.syncContexts();
      }
    });

    this.panel.webview.onDidReceiveMessage((message) => {
      void this.handleMessage(message);
    });

    this.panel.webview.html = this.renderHtml();
    this.onFocus(this, this);
  }

  static async create(
    context: vscode.ExtensionContext,
    leftSource: BinarySource,
    rightSource: BinarySource | undefined,
    onFocus: (session: ViewerSession | undefined, source: ViewerSession) => void
  ): Promise<ViewerSession> {
    const session = new ViewerSession(context, leftSource, rightSource, onFocus);
    await session.refresh();
    return session;
  }

  async refresh(): Promise<void> {
    const [leftMetadata, rightMetadata] = await Promise.all([
      this.leftSource.getMetadata(),
      this.rightSource?.getMetadata()
    ]);

    this.leftMetadata = leftMetadata;
    this.rightMetadata = rightMetadata;
    this.panel.title = rightMetadata ? `${path.basename(leftMetadata.path)} vs ${path.basename(rightMetadata.path)}` : `BinView: ${path.basename(leftMetadata.path)}`;

    await this.syncContexts();
    this.postState();
  }

  async gotoLine(): Promise<void> {
    const totalLines = this.getTotalLines();
    const raw = await vscode.window.showInputBox({
      placeHolder: `1 - ${totalLines}`,
      prompt: 'Go to display line (8 values per line)',
      validateInput: (value) => {
        const parsed = Number.parseInt(value, 10);
        if (!Number.isFinite(parsed) || parsed < 1 || parsed > totalLines) {
          return `Enter an integer between 1 and ${totalLines}.`;
        }
        return undefined;
      }
    });

    if (!raw) {
      return;
    }

    const lineNumber = Number.parseInt(raw, 10);
    this.panel.webview.postMessage({ type: 'gotoLine', lineNumber });
  }

  async reshape(): Promise<void> {
    if (!this.canTransform()) {
      void vscode.window.showInformationMessage('The active viewer does not support torch reshape.');
      return;
    }

    const input = await vscode.window.showInputBox({
      placeHolder: 'Example: 1, 32, -1',
      prompt: 'Enter the new tensor shape. One -1 dimension is allowed.'
    });
    if (!input) {
      return;
    }

    const shape = parseShapeInput(input);
    await this.applyTransform(async (source) => {
      if (!source.reshape) {
        throw new Error('This source does not support reshape.');
      }
      return source.reshape(shape);
    });
  }

  async slice(): Promise<void> {
    if (!this.canTransform()) {
      void vscode.window.showInformationMessage('The active viewer does not support torch slicing.');
      return;
    }

    const expression = await vscode.window.showInputBox({
      placeHolder: 'Example: :, 0, :128',
      prompt: 'Enter a Python-style indexing expression used as tensor[expression]'
    });
    if (!expression) {
      return;
    }

    await this.applyTransform(async (source) => {
      if (!source.slice) {
        throw new Error('This source does not support slicing.');
      }
      return source.slice(expression);
    });
  }

  async resetTransform(): Promise<void> {
    if (!this.canTransform()) {
      return;
    }

    await this.applyTransform(async (source) => {
      if (!source.resetTransform) {
        throw new Error('This source does not support transform reset.');
      }
      return source.resetTransform();
    });
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    await Promise.allSettled([this.leftSource.dispose(), this.rightSource?.dispose()]);
    this.onFocus(undefined, this);
    await vscode.commands.executeCommand('setContext', 'binView.viewerFocused', false);
    await vscode.commands.executeCommand('setContext', 'binView.canTransform', false);
  }

  private async applyTransform(run: (source: BinarySource) => Promise<SourceMetadata>): Promise<void> {
    const sources = [this.leftSource, this.rightSource].filter((source): source is BinarySource => Boolean(source));
    try {
      await Promise.all(sources.map((source) => run(source)));
      await this.refresh();
      this.panel.webview.postMessage({ type: 'clearRows' });
    } catch (error) {
      void vscode.window.showErrorMessage(error instanceof Error ? error.message : String(error));
    }
  }

  private async handleMessage(message: { type?: string; [key: string]: unknown }): Promise<void> {
    switch (message.type) {
      case 'ready':
        this.postState();
        break;
      case 'viewport':
        if (typeof message.startLine === 'number' && typeof message.lineCount === 'number') {
          await this.sendLines(message.startLine, message.lineCount, typeof message.requestId === 'number' ? message.requestId : 0);
        }
        break;
      case 'focus':
        this.onFocus(this, this);
        await this.syncContexts();
        break;
      case 'requestCommand':
        if (message.command === 'goto') {
          await this.gotoLine();
        } else if (message.command === 'reshape') {
          await this.reshape();
        } else if (message.command === 'slice') {
          await this.slice();
        } else if (message.command === 'reset') {
          await this.resetTransform();
        } else if (message.command === 'reload') {
          await this.refresh();
          this.panel.webview.postMessage({ type: 'clearRows' });
        }
        break;
      default:
        break;
    }
  }

  private async sendLines(startLine: number, lineCount: number, requestId: number): Promise<void> {
    try {
      const lines = await this.readLines(startLine, lineCount);
      this.panel.webview.postMessage({
        type: 'rows',
        requestId,
        startLine,
        lines
      });
    } catch (error) {
      this.panel.webview.postMessage({
        type: 'error',
        message: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async readLines(startLine: number, lineCount: number): Promise<ViewerLine[]> {
    const startIndex = Math.max(0, startLine) * ELEMENTS_PER_LINE;
    const count = Math.max(0, lineCount) * ELEMENTS_PER_LINE;
    const [leftValues, rightValues] = await Promise.all([
      this.leftSource.readScalars(startIndex, count),
      this.rightSource?.readScalars(startIndex, count)
    ]);

    const lines: ViewerLine[] = [];
    for (let lineOffset = 0; lineOffset < lineCount; lineOffset += 1) {
      const rowStart = lineOffset * ELEMENTS_PER_LINE;
      const left = fillLine(leftValues.slice(rowStart, rowStart + ELEMENTS_PER_LINE));
      const right = rightValues ? fillLine(rightValues.slice(rowStart, rowStart + ELEMENTS_PER_LINE)) : undefined;
      const different = Boolean(right && left.some((value, index) => {
        const other = right[index];
        if (!value && !other) {
          return false;
        }
        if (!value || !other) {
          return true;
        }
        return value.bits !== other.bits;
      }));

      lines.push({
        lineNumber: startLine + lineOffset + 1,
        startIndex: startIndex + rowStart,
        left,
        right,
        different
      });
    }

    return lines;
  }

  private postState(): void {
    if (!this.leftMetadata) {
      return;
    }

    const state: ViewerState = {
      rowsPerLine: ELEMENTS_PER_LINE,
      totalLines: this.getTotalLines(),
      left: this.leftMetadata,
      right: this.rightMetadata,
      canTransform: this.canTransform(),
      title: this.panel.title
    };

    this.panel.webview.postMessage({
      type: 'state',
      state
    });
  }

  private canTransform(): boolean {
    return Boolean(this.leftMetadata?.canTransform && (!this.rightMetadata || this.rightMetadata.canTransform));
  }

  private getTotalLines(): number {
    const leftTotal = this.leftMetadata?.totalElements ?? 0;
    const rightTotal = this.rightMetadata?.totalElements ?? 0;
    return Math.max(1, Math.ceil(Math.max(leftTotal, rightTotal) / ELEMENTS_PER_LINE));
  }

  private async syncContexts(): Promise<void> {
    await vscode.commands.executeCommand('setContext', 'binView.viewerFocused', this.panel.active && !this.disposed);
    await vscode.commands.executeCommand('setContext', 'binView.canTransform', this.panel.active && this.canTransform());
  }

  private renderHtml(): string {
    const nonce = Math.random().toString(36).slice(2);
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BinView</title>
  <style>
    :root {
      --line-height: 112px;
      --cell-width: 164px;
      --line-gutter: 92px;
      --border: var(--vscode-panel-border);
      --muted: var(--vscode-descriptionForeground);
      --bg-subtle: color-mix(in srgb, var(--vscode-editor-background) 92%, white 8%);
      --diff: color-mix(in srgb, var(--vscode-editorError-foreground) 18%, transparent);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
    body { display: flex; flex-direction: column; font-family: var(--vscode-font-family); }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: var(--vscode-sideBar-background);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .toolbar button {
      border: 1px solid var(--border);
      background: var(--vscode-button-secondaryBackground);
      color: var(--vscode-button-secondaryForeground);
      padding: 4px 10px;
      cursor: pointer;
    }
    .toolbar button.primary {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .summary {
      display: flex;
      gap: 16px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      overflow-x: auto;
      white-space: nowrap;
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 12px;
    }
    .summary strong { color: var(--vscode-textLink-foreground); }
    .viewer {
      flex: 1;
      overflow: auto;
      position: relative;
    }
    .content {
      position: relative;
      min-width: max-content;
    }
    .row {
      position: absolute;
      left: 0;
      right: 0;
      display: flex;
      min-height: var(--line-height);
      border-bottom: 1px solid color-mix(in srgb, var(--border) 65%, transparent);
    }
    .line-number {
      width: var(--line-gutter);
      padding: 10px;
      border-right: 1px solid var(--border);
      position: sticky;
      left: 0;
      background: var(--vscode-editor-background);
      z-index: 1;
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 12px;
    }
    .line-number .index {
      color: var(--muted);
      margin-top: 4px;
    }
    .side {
      display: flex;
      gap: 0;
      padding: 6px;
      min-width: calc(var(--cell-width) * 8 + 12px);
    }
    .cell {
      width: var(--cell-width);
      min-width: var(--cell-width);
      padding: 6px 8px;
      border-right: 1px solid color-mix(in srgb, var(--border) 50%, transparent);
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 12px;
      background: transparent;
    }
    .cell.compare-gap {
      border-left: 2px solid var(--border);
    }
    .cell.diff {
      background: var(--diff);
    }
    .cell.empty {
      color: var(--muted);
      background: var(--bg-subtle);
    }
    .cell .offset {
      color: var(--muted);
      margin-bottom: 4px;
    }
    .cell .value {
      font-weight: 600;
      margin-bottom: 4px;
    }
    .cell .hex {
      color: var(--muted);
      line-height: 1.35;
    }
    .status {
      padding: 8px 12px;
      color: var(--vscode-errorForeground);
      border-top: 1px solid var(--border);
      min-height: 34px;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button class="primary" data-command="goto">Go To Line</button>
    <button data-command="reload">Reload</button>
    <button data-command="reshape" id="reshapeBtn">Reshape</button>
    <button data-command="slice" id="sliceBtn">Slice</button>
    <button data-command="reset" id="resetBtn">Reset Transform</button>
  </div>
  <div class="summary" id="summary"></div>
  <div class="viewer" id="viewer" tabindex="0">
    <div class="content" id="content"></div>
  </div>
  <div class="status" id="status"></div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const viewer = document.getElementById('viewer');
    const content = document.getElementById('content');
    const summary = document.getElementById('summary');
    const statusEl = document.getElementById('status');
    const rowHeight = 112;
    const cache = new Map();
    let state = undefined;
    let lastRequestKey = '';
    let requestId = 0;

    function updateButtons() {
      const enabled = Boolean(state && state.canTransform);
      document.getElementById('reshapeBtn').disabled = !enabled;
      document.getElementById('sliceBtn').disabled = !enabled;
      document.getElementById('resetBtn').disabled = !enabled;
    }

    function setStatus(message = '') {
      statusEl.textContent = message;
    }

    function renderSummary() {
      if (!state) {
        summary.textContent = '';
        return;
      }

      const segments = [];
      segments.push('<strong>' + escapeHtml(state.title) + '</strong>');
      segments.push('Left: ' + formatMeta(state.left));
      if (state.right) {
        segments.push('Right: ' + formatMeta(state.right));
      }
      summary.innerHTML = segments.map((item) => '<span>' + item + '</span>').join('');
    }

    function formatMeta(meta) {
      const notes = meta.notes && meta.notes.length ? ' | ' + meta.notes.join(' ') : '';
      const transform = meta.transformSummary ? ' | ' + meta.transformSummary : '';
      return [
        escapeHtml(meta.label),
        escapeHtml(meta.format),
        escapeHtml(meta.dtypeLabel),
        'shape=' + escapeHtml(JSON.stringify(meta.shape)),
        'items=' + meta.totalElements
      ].join(' | ') + escapeHtml(transform + notes);
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function requestVisibleRows() {
      if (!state) {
        return;
      }

      const start = Math.max(0, Math.floor(viewer.scrollTop / rowHeight) - 20);
      const count = Math.min(state.totalLines - start, Math.ceil(viewer.clientHeight / rowHeight) + 40);
      const key = start + ':' + count;
      if (key === lastRequestKey) {
        render();
        return;
      }

      lastRequestKey = key;
      requestId += 1;
      vscode.postMessage({ type: 'viewport', startLine: start, lineCount: count, requestId });
    }

    function render() {
      if (!state) {
        return;
      }

      const start = Math.max(0, Math.floor(viewer.scrollTop / rowHeight) - 2);
      const count = Math.ceil(viewer.clientHeight / rowHeight) + 6;
      const end = Math.min(state.totalLines, start + count);
      content.style.height = (state.totalLines * rowHeight) + 'px';
      content.innerHTML = '';

      for (let line = start; line < end; line += 1) {
        const item = cache.get(line + 1);
        if (!item) {
          continue;
        }

        const row = document.createElement('div');
        row.className = 'row';
        row.style.top = (line * rowHeight) + 'px';

        const number = document.createElement('div');
        number.className = 'line-number';
        number.innerHTML = '<div>L' + item.lineNumber + '</div><div class="index">#' + item.startIndex + '</div>';
        row.appendChild(number);

        row.appendChild(renderSide(item.left, false, item.different, item.right));
        if (item.right) {
          row.appendChild(renderSide(item.right, true, item.different, item.left));
        }
        content.appendChild(row);
      }
    }

    function renderSide(values, compare, different, baseline) {
      const side = document.createElement('div');
      side.className = 'side';

      values.forEach((value, index) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        if (compare && index === 0) {
          cell.classList.add('compare-gap');
        }

        const leftValue = baseline ? baseline[index] : undefined;
        const isDiff = baseline && ((value && leftValue && value.bits !== leftValue.bits) || (!value && leftValue) || (value && !leftValue));
        if (isDiff) {
          cell.classList.add('diff');
        }

        if (!value) {
          cell.classList.add('empty');
          cell.textContent = 'EOF';
          side.appendChild(cell);
          return;
        }

        cell.title = value.bits;
        cell.innerHTML = [
          '<div class="offset">#' + value.index + '</div>',
          '<div class="value">' + escapeHtml(value.value) + '</div>',
          '<div class="hex">' + escapeHtml(value.hex) + '</div>'
        ].join('');
        side.appendChild(cell);
      });

      return side;
    }

    window.addEventListener('message', (event) => {
      const message = event.data;
      if (message.type === 'state') {
        state = message.state;
        cache.clear();
        lastRequestKey = '';
        setStatus('');
        renderSummary();
        updateButtons();
        requestVisibleRows();
      } else if (message.type === 'rows') {
        if (message.requestId !== requestId) {
          return;
        }
        for (const line of message.lines) {
          cache.set(line.lineNumber, line);
        }
        setStatus('');
        render();
      } else if (message.type === 'gotoLine') {
        if (!state) {
          return;
        }
        const target = Math.max(1, Math.min(state.totalLines, Number(message.lineNumber))) - 1;
        viewer.scrollTop = target * rowHeight;
        requestVisibleRows();
      } else if (message.type === 'clearRows') {
        cache.clear();
        lastRequestKey = '';
        requestVisibleRows();
      } else if (message.type === 'error') {
        setStatus(message.message || 'Unknown error');
      }
    });

    viewer.addEventListener('scroll', () => {
      requestVisibleRows();
      render();
    });

    viewer.addEventListener('focus', () => vscode.postMessage({ type: 'focus' }));
    window.addEventListener('focus', () => vscode.postMessage({ type: 'focus' }));

    window.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'g') {
        event.preventDefault();
        vscode.postMessage({ type: 'requestCommand', command: 'goto' });
      }
    });

    document.querySelectorAll('[data-command]').forEach((button) => {
      button.addEventListener('click', () => {
        vscode.postMessage({ type: 'requestCommand', command: button.dataset.command });
      });
    });

    vscode.postMessage({ type: 'ready' });
  </script>
</body>
</html>`;
  }
}

function fillLine(values: DisplayScalar[]): Array<DisplayScalar | null> {
  const result: Array<DisplayScalar | null> = [...values];
  while (result.length < ELEMENTS_PER_LINE) {
    result.push(null);
  }
  return result;
}

function parseShapeInput(input: string): number[] {
  const segments = input.split(',').map((item) => item.trim()).filter(Boolean);
  if (segments.length === 0) {
    throw new Error('Shape cannot be empty.');
  }

  let negativeOneCount = 0;
  const shape = segments.map((segment) => {
    const value = Number.parseInt(segment, 10);
    if (!Number.isFinite(value) || value === 0) {
      throw new Error(`Invalid shape segment: ${segment}`);
    }
    if (value < 0) {
      if (value !== -1) {
        throw new Error('Only -1 may be used as a negative reshape dimension.');
      }
      negativeOneCount += 1;
    }
    return value;
  });

  if (negativeOneCount > 1) {
    throw new Error('Only one -1 dimension is allowed.');
  }

  return shape;
}
