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
  maxElements: number;
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
    private readonly onFocus: (session: ViewerSession | undefined, source: ViewerSession) => void,
    panel?: vscode.WebviewPanel
  ) {
    this.panel = panel ?? vscode.window.createWebviewPanel('binView.viewer', 'BinView', vscode.ViewColumn.Active, {
      enableScripts: true,
      retainContextWhenHidden: true
    });
    this.panel.webview.options = {
      enableScripts: true
    };

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

  static async createInPanel(
    context: vscode.ExtensionContext,
    leftSource: BinarySource,
    rightSource: BinarySource | undefined,
    panel: vscode.WebviewPanel,
    onFocus: (session: ViewerSession | undefined, source: ViewerSession) => void
  ): Promise<ViewerSession> {
    const session = new ViewerSession(context, leftSource, rightSource, onFocus, panel);
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
    this.panel.webview.postMessage({ type: 'focusGoto' });
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
      const message = error instanceof Error ? error.message : String(error);
      this.panel.webview.postMessage({ type: 'error', message });
      void vscode.window.showErrorMessage(message);
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
      case 'applyTransform':
        if (typeof message.mode === 'string' && typeof message.value === 'string') {
          await this.applyTransformRequest(message.mode, message.value);
        }
        break;
      case 'runConsole':
        if (typeof message.mode === 'string' && typeof message.value === 'string') {
          await this.runConsoleRequest(message.mode, message.value);
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
      maxElements: this.getMaxElements(),
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
    return Math.max(1, Math.ceil(this.getMaxElements() / ELEMENTS_PER_LINE));
  }

  private getMaxElements(): number {
    const leftTotal = this.leftMetadata?.totalElements ?? 0;
    const rightTotal = this.rightMetadata?.totalElements ?? 0;
    return Math.max(leftTotal, rightTotal);
  }

  private async syncContexts(): Promise<void> {
    await vscode.commands.executeCommand('setContext', 'binView.viewerFocused', this.panel.active && !this.disposed);
    await vscode.commands.executeCommand('setContext', 'binView.canTransform', this.panel.active && this.canTransform());
  }

  private async applyTransformRequest(mode: string, value: string): Promise<void> {
    if (!this.canTransform()) {
      this.panel.webview.postMessage({ type: 'error', message: 'The active viewer does not support torch transforms.' });
      return;
    }

    try {
      const normalizedMode = mode.trim().toLowerCase();
      if (normalizedMode === 'reset') {
        await this.resetTransform();
        return;
      }

      if (normalizedMode === 'reshape') {
        const shape = parseShapeInput(value);
        await this.applyTransform(async (source) => {
          if (!source.reshape) {
            throw new Error('This source does not support reshape.');
          }
          return source.reshape(shape);
        });
        return;
      }

      if (normalizedMode === 'slice') {
        await this.applyTransform(async (source) => {
          if (!source.slice) {
            throw new Error('This source does not support slicing.');
          }
          return source.slice(value);
        });
        return;
      }

      if (normalizedMode === 'python') {
        await this.applyTransform(async (source) => {
          if (!source.applyPython) {
            throw new Error('This source does not support Python tensor expressions.');
          }
          return source.applyPython(value);
        });
        return;
      }

      this.panel.webview.postMessage({ type: 'error', message: `Unknown transform mode: ${mode}` });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.panel.webview.postMessage({ type: 'error', message });
      void vscode.window.showErrorMessage(message);
    }
  }

  private async runConsoleRequest(mode: string, value: string): Promise<void> {
    if (!this.canTransform()) {
      this.panel.webview.postMessage({ type: 'error', message: 'The active viewer does not support torch transforms.' });
      return;
    }

    try {
      const normalizedMode = mode.trim().toLowerCase();
      if (normalizedMode === 'reset') {
        await this.resetTransform();
        this.panel.webview.postMessage({
          type: 'consoleResult',
          mode: normalizedMode,
          command: 'reset',
          entries: [
            {
              label: this.rightSource ? 'Both' : 'Tensor',
              output: 'Tensor reset to the original selection.',
              updated: true
            }
          ]
        });
        return;
      }

      const code = buildConsoleSource(normalizedMode, value);
      const sources = [this.leftSource, this.rightSource].filter((source): source is BinarySource => Boolean(source));
      const results = await Promise.all(sources.map(async (source, index) => {
        if (!source.runConsole) {
          throw new Error('This source does not support the Python console.');
        }

        const result = await source.runConsole(code);
        const label = this.rightSource ? (index === 0 ? 'Left' : 'Right') : 'Tensor';
        return {
          label,
          output: result.output,
          resultText: result.resultText,
          updated: result.updated,
          raw: result
        };
      }));

      if (results.some((entry) => entry.raw.updated)) {
        await this.refresh();
        this.panel.webview.postMessage({ type: 'clearRows' });
      }

      this.panel.webview.postMessage({
        type: 'consoleResult',
        mode: normalizedMode,
        command: normalizedMode === 'python' ? value : code,
        entries: results.map(({ raw, ...entry }) => entry)
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.panel.webview.postMessage({ type: 'error', message });
      void vscode.window.showErrorMessage(message);
    }
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
      --focus: color-mix(in srgb, var(--vscode-focusBorder) 60%, transparent);
      --focus-strong: color-mix(in srgb, var(--vscode-focusBorder) 22%, transparent);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
    body { display: flex; flex-direction: column; font-family: var(--vscode-font-family); }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: end;
      flex-wrap: wrap;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: var(--vscode-sideBar-background);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }
    .field span {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .jump-field {
      width: min(320px, 100%);
    }
    .toolbar-actions {
      display: flex;
      gap: 10px;
      margin-left: auto;
    }
    .toolbar input,
    .toolbar select,
    .toolbar button {
      border: 1px solid var(--border);
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      padding: 7px 10px;
      min-height: 34px;
      font: inherit;
    }
    .toolbar input,
    .toolbar select {
      width: 100%;
    }
    .toolbar input:focus,
    .toolbar select:focus {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: 0;
    }
    .toolbar button {
      background: var(--vscode-button-secondaryBackground);
      color: var(--vscode-button-secondaryForeground);
      cursor: pointer;
    }
    .toolbar button.primary {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .toolbar button:disabled,
    .toolbar input:disabled,
    .toolbar select:disabled {
      opacity: 0.6;
      cursor: default;
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
    .main-shell {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
    }
    .viewer-shell {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .console-dock {
      display: grid;
      grid-template-columns: 46px 0;
      min-height: 0;
      border-left: 1px solid var(--border);
      background: var(--vscode-sideBar-background);
      transition: grid-template-columns 160ms ease;
    }
    .console-dock[data-open="true"] {
      grid-template-columns: 46px minmax(320px, 420px);
    }
    .console-dock-toggle {
      border: 0;
      border-right: 1px solid var(--border);
      background:
        linear-gradient(180deg, color-mix(in srgb, var(--vscode-sideBar-background) 82%, var(--vscode-editor-background) 18%), color-mix(in srgb, var(--vscode-editor-background) 84%, black 16%));
      color: var(--vscode-sideBar-foreground);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      padding: 14px 10px;
    }
    .console-dock-toggle:focus {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: -1px;
    }
    .console-shell {
      display: none;
      min-width: 0;
      min-height: 0;
      height: 100%;
      grid-template-rows: auto auto minmax(160px, 1fr);
      gap: 12px;
      padding: 14px 12px 12px;
      background:
        radial-gradient(circle at top right, color-mix(in srgb, var(--vscode-textLink-foreground) 18%, transparent), transparent 34%),
        linear-gradient(135deg, color-mix(in srgb, var(--vscode-sideBar-background) 86%, var(--vscode-editor-background) 14%), color-mix(in srgb, var(--vscode-editor-background) 88%, black 12%));
      overflow: auto;
    }
    .console-dock[data-open="true"] .console-shell {
      display: grid;
    }
    .console-shell input,
    .console-shell select,
    .console-shell textarea,
    .console-shell button {
      border: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
      background: color-mix(in srgb, var(--vscode-input-background) 88%, var(--vscode-editor-background) 12%);
      color: var(--vscode-input-foreground);
      font: inherit;
    }
    .console-shell input:focus,
    .console-shell select:focus,
    .console-shell textarea:focus {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: 0;
    }
    .console-shell button {
      min-height: 36px;
      padding: 8px 12px;
      cursor: pointer;
      border-radius: 12px;
      background: color-mix(in srgb, var(--vscode-button-secondaryBackground) 88%, var(--vscode-editor-background) 12%);
      color: var(--vscode-button-secondaryForeground);
    }
    .console-shell button.primary {
      background: linear-gradient(135deg, var(--vscode-button-background), color-mix(in srgb, var(--vscode-button-background) 76%, black 24%));
      color: var(--vscode-button-foreground);
    }
    .console-shell button:disabled,
    .console-shell input:disabled,
    .console-shell select:disabled,
    .console-shell textarea:disabled {
      opacity: 0.58;
      cursor: default;
    }
    .console-hero {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .console-title {
      min-width: 0;
    }
    .console-kicker {
      margin-bottom: 6px;
      color: var(--vscode-textLink-foreground);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .console-title strong {
      display: block;
      font-size: 18px;
      line-height: 1.2;
    }
    .console-title p {
      margin: 8px 0 0;
      max-width: 720px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .console-head-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .console-chip {
      padding: 8px 12px;
      border: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
      border-radius: 999px;
      background: color-mix(in srgb, var(--vscode-editor-background) 84%, white 16%);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .console-stage {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      align-items: stretch;
    }
    .console-controls,
    .console-editor {
      border: 1px solid color-mix(in srgb, var(--border) 78%, transparent);
      border-radius: 18px;
      background: color-mix(in srgb, var(--vscode-editor-background) 90%, white 10%);
      box-shadow: 0 10px 28px color-mix(in srgb, black 10%, transparent);
    }
    .console-controls {
      display: grid;
      gap: 12px;
      padding: 14px;
    }
    .console-controls .field input,
    .console-controls .field select {
      width: 100%;
      min-height: 38px;
      padding: 8px 10px;
      border-radius: 12px;
    }
    .console-inline-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .console-support {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .console-editor {
      display: grid;
      gap: 12px;
      padding: 14px;
      min-width: 0;
      transition: opacity 140ms ease, transform 140ms ease;
    }
    .console-editor[data-open="false"] {
      opacity: 0.72;
    }
    .console-editor[data-open="false"] .console-input,
    .console-editor[data-open="false"] .console-editor-actions {
      display: none;
    }
    .console-editor-head {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .console-editor-title {
      font-size: 13px;
      font-weight: 600;
    }
    .console-editor-hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      max-width: 540px;
    }
    .console-input {
      width: 100%;
      min-height: 160px;
      padding: 12px;
      resize: vertical;
      border-radius: 14px;
      white-space: pre;
      font-family: var(--vscode-editor-font-family, monospace);
    }
    .console-editor-actions {
      display: flex;
      gap: 10px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .console-log {
      margin: 0;
      min-height: 140px;
      max-height: 280px;
      padding: 14px;
      overflow: auto;
      border: 1px solid color-mix(in srgb, var(--border) 78%, transparent);
      border-radius: 18px;
      background: color-mix(in srgb, var(--vscode-editor-background) 88%, black 12%);
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 12px;
      line-height: 1.45;
      box-shadow: inset 0 1px 0 color-mix(in srgb, white 8%, transparent);
    }
    .console-log.empty {
      color: var(--muted);
    }
    .console-entry + .console-entry {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
    }
    .console-entry-label {
      color: var(--muted);
      margin-bottom: 4px;
    }
    .console-entry pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .console-code {
      color: var(--vscode-textLink-foreground);
      margin-bottom: 6px;
    }
    .console-output {
      color: var(--vscode-terminal-ansiGreen, var(--vscode-editor-foreground));
      margin-top: 6px;
    }
    .console-result {
      color: var(--vscode-editor-foreground);
      margin-top: 6px;
    }
    @media (max-width: 1100px) {
      .main-shell {
        grid-template-columns: 1fr;
      }
      .console-dock {
        grid-template-columns: 1fr;
        border-left: 0;
        border-top: 1px solid var(--border);
      }
      .console-dock[data-open="true"] {
        grid-template-columns: 1fr;
      }
      .console-dock-toggle {
        writing-mode: horizontal-tb;
        transform: none;
        border-right: 0;
        border-bottom: 1px solid var(--border);
        padding: 10px 12px;
      }
    }
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
    .row.target-line {
      background: var(--focus-strong);
    }
    .row.target-line .line-number {
      background: var(--focus-strong);
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
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 0;
      padding: 6px;
      flex: 1 1 0;
      min-width: 0;
    }
    .cell {
      width: auto;
      min-width: 0;
      padding: 6px 8px;
      border-right: 1px solid color-mix(in srgb, var(--border) 50%, transparent);
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 11px;
      background: transparent;
      overflow: hidden;
    }
    .cell.compare-gap {
      border-left: 2px solid var(--border);
    }
    .cell.diff {
      background: var(--diff);
    }
    .cell.target-cell {
      background: var(--focus-strong);
      box-shadow: inset 0 0 0 1px var(--vscode-focusBorder);
    }
    .cell.empty {
      color: var(--muted);
      background: var(--bg-subtle);
    }
    .cell .offset {
      color: var(--muted);
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }
    .cell .value {
      font-weight: 600;
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }
    .cell .hex {
      color: var(--muted);
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .status {
      padding: 8px 12px;
      color: var(--vscode-errorForeground);
      border-top: 1px solid var(--border);
      min-height: 34px;
    }
    .status[data-kind="info"] {
      color: var(--vscode-descriptionForeground);
    }
    .status[data-kind="success"] {
      color: var(--vscode-testing-iconPassed);
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <label class="field jump-field">
      <span>Jump</span>
      <input id="jumpInput" type="text" spellcheck="false" placeholder="Index: #128 or #64*2, line: L16">
    </label>
    <div class="toolbar-actions">
      <button type="button" id="reloadBtn" data-command="reload">Reload</button>
    </div>
  </div>
  <div class="summary" id="summary"></div>
  <div class="main-shell">
    <div class="viewer-shell">
      <div class="viewer" id="viewer" tabindex="0">
        <div class="content" id="content"></div>
      </div>
      <div class="status" id="status"></div>
    </div>
    <aside class="console-dock" id="consoleDock" data-open="false">
      <button class="console-dock-toggle" type="button" id="toggleConsoleDockBtn" aria-expanded="false">Python</button>
      <section class="console-shell">
        <div class="console-hero">
          <div class="console-title">
            <div class="console-kicker">Torch Workspace</div>
            <strong>Python Side Panel</strong>
            <p>Python stays hidden by default. Open this right-hand panel only when you need reshape, slice, reset, or free-form torch commands.</p>
          </div>
          <div class="console-head-actions">
            <div class="console-chip" id="consoleStateChip">Panel hidden</div>
            <button class="primary" type="button" id="toggleConsoleBtn">Open Python</button>
            <button type="button" id="clearConsoleBtn">Clear Log</button>
            <button type="button" id="closeDockBtn">Hide Panel</button>
          </div>
        </div>
        <div class="console-stage">
          <div class="console-controls">
            <label class="field">
              <span>Action</span>
              <select id="transformMode">
                <option value="python">Python / Torch</option>
                <option value="reshape">Reshape Helper</option>
                <option value="slice">Slice Helper</option>
                <option value="reset">Reset</option>
              </select>
            </label>
            <label class="field" id="helperField">
              <span id="helperLabel">Helper Input</span>
              <input id="helperInput" type="text" spellcheck="false" placeholder="1, 32, -1">
            </label>
            <div class="console-inline-actions">
              <button class="primary" type="button" id="runHelperBtn">Run Helper</button>
              <button type="button" id="insertExampleBtn">Insert Example</button>
            </div>
            <div class="console-support" id="consoleSupportText">Python input is hidden by default. Open the panel and switch to <code>Python / Torch</code> when you want multi-line commands.</div>
          </div>
          <form class="console-editor" id="consoleForm" data-open="false">
            <div class="console-editor-head">
              <div class="console-editor-title">Python Editor</div>
              <div class="console-editor-hint"><code>tensor</code>, <code>torch</code>, and <code>math</code> are available. <code>Ctrl+Enter</code> runs the current block.</div>
            </div>
            <textarea id="transformInput" class="console-input" spellcheck="false" placeholder="print(tensor.shape)&#10;tensor.permute(0, 2, 1)"></textarea>
            <div class="console-editor-actions">
              <button class="primary" type="submit" id="applyBtn">Run Python</button>
              <button type="button" id="closeConsoleBtn">Close Editor</button>
            </div>
          </form>
        </div>
        <div class="console-log empty" id="consoleLog">No console output yet.</div>
      </section>
    </aside>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const viewer = document.getElementById('viewer');
    const content = document.getElementById('content');
    const summary = document.getElementById('summary');
    const statusEl = document.getElementById('status');
    const consoleDock = document.getElementById('consoleDock');
    const toggleConsoleDockBtn = document.getElementById('toggleConsoleDockBtn');
    const consoleLog = document.getElementById('consoleLog');
    const consoleForm = document.getElementById('consoleForm');
    const consoleStateChip = document.getElementById('consoleStateChip');
    const consoleSupportText = document.getElementById('consoleSupportText');
    const clearConsoleBtn = document.getElementById('clearConsoleBtn');
    const closeDockBtn = document.getElementById('closeDockBtn');
    const toggleConsoleBtn = document.getElementById('toggleConsoleBtn');
    const closeConsoleBtn = document.getElementById('closeConsoleBtn');
    const insertExampleBtn = document.getElementById('insertExampleBtn');
    const jumpInput = document.getElementById('jumpInput');
    const transformMode = document.getElementById('transformMode');
    const helperField = document.getElementById('helperField');
    const helperLabel = document.getElementById('helperLabel');
    const helperInput = document.getElementById('helperInput');
    const runHelperBtn = document.getElementById('runHelperBtn');
    const transformInput = document.getElementById('transformInput');
    const applyBtn = document.getElementById('applyBtn');
    const reloadBtn = document.getElementById('reloadBtn');
    const rowHeight = 112;
    const cache = new Map();
    let state = undefined;
    let lastRequestKey = '';
    let requestId = 0;
    let highlighted = undefined;
    let jumpTimer = undefined;
    let transformBusy = false;
    let consoleDockOpen = false;
    let consoleEditorOpen = false;
    let consoleEntries = [];
    const pythonExample = 'print(tensor.shape)\\ntensor.permute(0, 2, 1)';
    const helperPlaceholders = {
      reshape: '1, 32, -1',
      slice: ':, 0, :128',
      reset: 'Reset returns to the original tensor'
    };
    const helperExamples = {
      reshape: '1, 32, -1',
      slice: ':, 0, :128',
      reset: ''
    };
    const helperLabels = {
      reshape: 'Shape',
      slice: 'Slice Expression',
      reset: 'Reset'
    };

    function isTransformEnabled() {
      return Boolean(state && state.canTransform);
    }

    function focusPythonEditor() {
      if (transformInput.disabled || !consoleDockOpen) {
        return;
      }
      transformInput.focus();
      transformInput.setSelectionRange(transformInput.value.length, transformInput.value.length);
    }

    function openConsoleDock(options = {}) {
      const { focusPython = false, silent = false } = options;
      consoleDockOpen = true;
      updateControls({ focusPython });
      if (!silent) {
        setStatus('Python panel opened.', 'info');
      }
    }

    function closeConsoleDock() {
      consoleDockOpen = false;
      updateControls();
      if (statusEl.dataset.kind !== 'error') {
        setStatus('Python panel hidden.', 'info');
      }
    }

    function updateControls(options = {}) {
      const { focusPython = false } = options;
      const transformEnabled = isTransformEnabled();
      const mode = transformMode.value;
      const pythonMode = mode === 'python';
      const helperNeedsValue = mode === 'reshape' || mode === 'slice';
      const helperVisible = !pythonMode && mode !== 'reset';

      consoleDock.dataset.open = consoleDockOpen ? 'true' : 'false';
      toggleConsoleDockBtn.textContent = consoleDockOpen ? 'Hide Python' : 'Python';
      toggleConsoleDockBtn.setAttribute('aria-expanded', consoleDockOpen ? 'true' : 'false');
      jumpInput.disabled = !state;
      transformMode.disabled = !transformEnabled || transformBusy;
      reloadBtn.disabled = transformBusy;
      clearConsoleBtn.disabled = transformBusy || consoleEntries.length === 0;
      closeDockBtn.disabled = !consoleDockOpen;
      helperField.hidden = !helperVisible;
      helperLabel.textContent = helperLabels[mode] || 'Helper Input';
      helperInput.disabled = !transformEnabled || transformBusy || !helperVisible;
      helperInput.placeholder = helperPlaceholders[mode] || '';
      runHelperBtn.hidden = pythonMode;
      runHelperBtn.disabled = !transformEnabled || transformBusy || (helperNeedsValue && !helperInput.value.trim());
      runHelperBtn.textContent = mode === 'reset' ? 'Reset Tensor' : 'Run Helper';
      insertExampleBtn.disabled = !transformEnabled || transformBusy || (pythonMode ? false : mode === 'reset');
      insertExampleBtn.textContent = pythonMode ? 'Insert Python Example' : 'Insert Example';
      consoleForm.hidden = !pythonMode;
      consoleForm.dataset.open = consoleEditorOpen ? 'true' : 'false';
      transformInput.disabled = !transformEnabled || transformBusy || !pythonMode || !consoleEditorOpen || !consoleDockOpen;
      transformInput.placeholder = pythonExample;
      applyBtn.disabled = !transformEnabled || transformBusy || !pythonMode || !consoleEditorOpen || !consoleDockOpen || !transformInput.value.trim();
      toggleConsoleBtn.disabled = !transformEnabled || transformBusy;
      toggleConsoleBtn.textContent = pythonMode ? (consoleEditorOpen ? 'Focus Editor' : 'Open Python') : 'Switch To Python';
      closeConsoleBtn.disabled = !transformEnabled || transformBusy || !pythonMode || !consoleEditorOpen;
      if (!consoleDockOpen) {
        consoleStateChip.textContent = 'Panel hidden';
      } else if (pythonMode) {
        consoleStateChip.textContent = consoleEditorOpen ? 'Python input active' : 'Python input closed';
      } else {
        consoleStateChip.textContent = mode === 'reset' ? 'Reset helper ready' : 'Helper mode ready';
      }

      if (!transformEnabled) {
        consoleSupportText.innerHTML = 'Python control is available only when the current viewer is backed by a torch tensor.';
      } else if (pythonMode) {
        consoleSupportText.innerHTML = consoleEditorOpen
          ? 'Multi-line Python is live. Return a tensor, or assign back into <code>tensor</code>, then press <code>Ctrl+Enter</code> to run.'
          : 'Python input stays closed by default. Click <code>Open Python</code> when you want to edit and run commands.';
      } else if (mode === 'reshape') {
        consoleSupportText.innerHTML = 'Quick helper runs <code>tensor = tensor.reshape(...)</code> and refreshes the viewer in place.';
      } else if (mode === 'slice') {
        consoleSupportText.innerHTML = 'Quick helper runs <code>tensor = tensor[...]</code> using Python-style indexing.';
      } else {
        consoleSupportText.innerHTML = 'Reset restores the original tensor selection and clears the current transform chain.';
      }

      if (focusPython && pythonMode && consoleEditorOpen) {
        focusPythonEditor();
      }
    }

    function openPythonEditor(options = {}) {
      const { silent = false } = options;
      if (!isTransformEnabled() || transformBusy) {
        return;
      }
      transformMode.value = 'python';
      consoleDockOpen = true;
      consoleEditorOpen = true;
      updateControls({ focusPython: true });
      if (!silent) {
        setStatus('Python editor opened.', 'info');
      }
    }

    function closePythonEditor() {
      consoleEditorOpen = false;
      updateControls();
      if (isTransformEnabled()) {
        setStatus('Python editor closed.', 'info');
      }
    }

    function startConsoleRun(mode, value) {
      if (!isTransformEnabled()) {
        return;
      }
      transformBusy = true;
      updateControls();
      if (mode === 'python') {
        setStatus('Running Python console...', 'info');
      } else if (mode === 'reset') {
        setStatus('Resetting tensor...', 'info');
      } else {
        setStatus('Applying ' + mode + ' helper...', 'info');
      }
      vscode.postMessage({ type: 'runConsole', mode, value });
    }

    function renderConsole() {
      if (consoleEntries.length === 0) {
        consoleLog.classList.add('empty');
        consoleLog.textContent = 'No console output yet.';
        return;
      }

      consoleLog.classList.remove('empty');
      consoleLog.innerHTML = consoleEntries.map((entry, index) => {
        const rows = entry.entries.map((item) => {
          const segments = [
            '<div class="console-entry-label">' + escapeHtml(item.label) + (item.updated ? ' | view updated' : '') + '</div>'
          ];
          if (item.output) {
            segments.push('<pre class="console-output">' + escapeHtml(item.output) + '</pre>');
          }
          if (item.resultText) {
            segments.push('<pre class="console-result">' + escapeHtml(item.resultText) + '</pre>');
          }
          if (!item.output && !item.resultText) {
            segments.push('<pre class="console-output">Command completed.</pre>');
          }
          return segments.join('');
        }).join('');

        return [
          '<div class="console-entry">',
          '<div class="console-entry-label">In [' + (index + 1) + '] ' + escapeHtml(entry.mode) + '</div>',
          '<pre class="console-code">' + escapeHtml(entry.command) + '</pre>',
          rows,
          '</div>'
        ].join('');
      }).join('');
      consoleLog.scrollTop = consoleLog.scrollHeight;
    }

    function setStatus(message = '', kind = 'error') {
      statusEl.dataset.kind = message ? kind : '';
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

    function clearHighlight() {
      highlighted = undefined;
      render();
    }

    function focusJumpInput() {
      jumpInput.focus();
      jumpInput.select();
    }

    function tokenizeIndexExpression(input) {
      const tokens = [];
      let position = 0;
      while (position < input.length) {
        const remaining = input.slice(position);
        const whitespace = remaining.match(/^\\s+/);
        if (whitespace) {
          position += whitespace[0].length;
          continue;
        }

        const operator = remaining.match(/^(\\*\\*|\\/\\/|[()+\\-*/%])/);
        if (operator) {
          tokens.push(operator[0]);
          position += operator[0].length;
          continue;
        }

        const number = remaining.match(/^\\d+/);
        if (number) {
          tokens.push(number[0]);
          position += number[0].length;
          continue;
        }

        throw new Error('Only integers, parentheses, and + - * / // % ** operators are supported in index expressions.');
      }

      return tokens;
    }

    function evaluateIndexExpression(raw) {
      const value = raw.trim().replace(/^#\\s*/, '');
      if (!value) {
        throw new Error('Enter an index expression after #.');
      }

      const tokens = tokenizeIndexExpression(value);
      let index = 0;

      function peek() {
        return tokens[index];
      }

      function consume(token) {
        if (tokens[index] !== token) {
          throw new Error('Invalid index expression.');
        }
        index += 1;
      }

      function ensureFinite(valueToCheck) {
        if (!Number.isFinite(valueToCheck)) {
          throw new Error('Index expression is too large to evaluate.');
        }
        return valueToCheck;
      }

      function parsePrimary() {
        const token = peek();
        if (token === '(') {
          consume('(');
          const inner = parseExpression();
          if (peek() !== ')') {
            throw new Error('Missing closing parenthesis in index expression.');
          }
          consume(')');
          return inner;
        }

        if (token && /^\\d+$/.test(token)) {
          index += 1;
          return Number.parseInt(token, 10);
        }

        throw new Error('Invalid index expression.');
      }

      function parseUnary() {
        const token = peek();
        if (token === '+' || token === '-') {
          index += 1;
          const valueToCheck = parseUnary();
          return ensureFinite(token === '-' ? -valueToCheck : valueToCheck);
        }
        return parsePrimary();
      }

      function parsePower() {
        let left = parseUnary();
        if (peek() === '**') {
          consume('**');
          const right = parsePower();
          left = ensureFinite(left ** right);
        }
        return left;
      }

      function parseTerm() {
        let left = parsePower();
        while (true) {
          const token = peek();
          if (token !== '*' && token !== '/' && token !== '//' && token !== '%') {
            return left;
          }

          index += 1;
          const right = parsePower();
          if ((token === '/' || token === '//' || token === '%') && right === 0) {
            throw new Error('Division by zero is not allowed in index expressions.');
          }

          if (token === '*') {
            left = ensureFinite(left * right);
          } else if (token === '/') {
            left = ensureFinite(left / right);
          } else if (token === '//') {
            left = ensureFinite(Math.floor(left / right));
          } else {
            left = ensureFinite(left % right);
          }
        }
      }

      function parseExpression() {
        let left = parseTerm();
        while (true) {
          const token = peek();
          if (token !== '+' && token !== '-') {
            return left;
          }
          index += 1;
          const right = parseTerm();
          left = ensureFinite(token === '+' ? left + right : left - right);
        }
      }

      const result = parseExpression();
      if (index !== tokens.length) {
        throw new Error('Invalid index expression.');
      }
      if (!Number.isInteger(result) || result < 0) {
        throw new Error('Index expression must evaluate to a non-negative integer.');
      }
      return result;
    }

    function parseJumpTarget(raw) {
      if (!state) {
        return { error: 'Viewer is not ready yet.' };
      }

      const value = raw.trim();
      if (!value) {
        return undefined;
      }

      const lineMatch = value.match(/^l(?:ine)?\\s*[:#]?\\s*(\\d+)$/i);
      if (lineMatch) {
        const lineNumber = Number.parseInt(lineMatch[1], 10);
        if (!Number.isFinite(lineNumber) || lineNumber < 1 || lineNumber > state.totalLines) {
          return { error: 'Line must be between 1 and ' + state.totalLines + '.' };
        }
        return {
          lineNumber,
          message: 'Jumped to line L' + lineNumber + '.'
        };
      }

      const looksLikeIndex = /^#/.test(value) || /^[\\d(+-]/.test(value);
      if (looksLikeIndex) {
        if (state.maxElements <= 0) {
          return { error: 'The current view has no elements to jump to.' };
        }

        let index;
        try {
          index = evaluateIndexExpression(value);
        } catch (error) {
          return { error: error instanceof Error ? error.message : String(error) };
        }

        if (!Number.isFinite(index) || index < 0 || index >= state.maxElements) {
          return { error: 'Index must be between 0 and ' + Math.max(0, state.maxElements - 1) + '.' };
        }
        return {
          lineNumber: Math.floor(index / state.rowsPerLine) + 1,
          index,
          message: 'Jumped to index #' + index + '.'
        };
      }

      return { error: 'Use #128 or #64*2 for an index, or L16 for a display line.' };
    }

    function applyJump(raw, showErrors) {
      const target = parseJumpTarget(raw);
      if (!target) {
        setStatus('', 'info');
        clearHighlight();
        return false;
      }
      if (target.error) {
        if (showErrors) {
          setStatus(target.error, 'error');
        }
        return false;
      }

      highlighted = target;
      viewer.scrollTop = (target.lineNumber - 1) * rowHeight;
      requestVisibleRows();
      render();
      setStatus(target.message, 'info');
      return true;
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
        if (highlighted && highlighted.lineNumber === item.lineNumber) {
          row.classList.add('target-line');
        }

        const number = document.createElement('div');
        number.className = 'line-number';
        number.innerHTML = '<div>L' + item.lineNumber + '</div><div class="index">#' + item.startIndex + '</div>';
        row.appendChild(number);

        row.appendChild(renderSide(item.left, false, item.different, item.right, item.startIndex, item.lineNumber));
        if (item.right) {
          row.appendChild(renderSide(item.right, true, item.different, item.left, item.startIndex, item.lineNumber));
        }
        content.appendChild(row);
      }
    }

    function renderSide(values, compare, different, baseline, startIndex, lineNumber) {
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
        if (highlighted && highlighted.index === startIndex + index && highlighted.lineNumber === lineNumber) {
          cell.classList.add('target-cell');
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
        transformBusy = false;
        if (highlighted && highlighted.index !== undefined && highlighted.index >= state.maxElements) {
          highlighted = undefined;
        }
        setStatus('', 'info');
        renderSummary();
        updateControls();
        requestVisibleRows();
      } else if (message.type === 'rows') {
        if (message.requestId !== requestId) {
          return;
        }
        for (const line of message.lines) {
          cache.set(line.lineNumber, line);
        }
        if (!statusEl.textContent || statusEl.dataset.kind !== 'error') {
          setStatus(statusEl.textContent, statusEl.dataset.kind || 'info');
        }
        render();
      } else if (message.type === 'focusGoto') {
        focusJumpInput();
      } else if (message.type === 'clearRows') {
        cache.clear();
        lastRequestKey = '';
        requestVisibleRows();
        render();
      } else if (message.type === 'consoleResult') {
        transformBusy = false;
        updateControls();
        const mode = message.mode || transformMode.value;
        const entries = Array.isArray(message.entries) ? message.entries : [];
        consoleEntries.push({
          mode,
          command: message.command || '',
          entries
        });
        renderConsole();
        const updated = entries.some((entry) => Boolean(entry.updated));
        if (mode === 'python') {
          setStatus(updated ? 'Python console finished and updated the tensor.' : 'Python console finished.', 'success');
        } else if (mode === 'reset') {
          setStatus('Tensor reset finished.', 'success');
        } else {
          setStatus(updated ? 'Helper finished and updated the tensor.' : 'Helper finished.', 'success');
        }
      } else if (message.type === 'error') {
        transformBusy = false;
        updateControls();
        setStatus(message.message || 'Unknown error', 'error');
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
        focusJumpInput();
      }
    });

    jumpInput.addEventListener('input', () => {
      if (jumpTimer) {
        clearTimeout(jumpTimer);
      }
      if (!jumpInput.value.trim()) {
        setStatus('', 'info');
        clearHighlight();
        return;
      }
      jumpTimer = setTimeout(() => {
        applyJump(jumpInput.value, false);
      }, 180);
    });

    jumpInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        if (jumpTimer) {
          clearTimeout(jumpTimer);
        }
        applyJump(jumpInput.value, true);
      }
    });

    jumpInput.addEventListener('blur', () => {
      if (jumpInput.value.trim()) {
        applyJump(jumpInput.value, true);
      }
    });

    transformMode.addEventListener('change', () => {
      updateControls();
      if (transformMode.value === 'reset') {
        helperInput.value = '';
        transformInput.value = '';
        setStatus('Reset restores the selected tensor.', 'info');
      } else if (statusEl.dataset.kind === 'info') {
        setStatus('', 'info');
      }
    });

    helperInput.addEventListener('input', () => {
      updateControls();
    });

    helperInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        runHelperBtn.click();
      }
    });

    transformInput.addEventListener('input', () => {
      updateControls();
    });

    transformInput.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        consoleForm.requestSubmit();
      }
    });

    consoleForm.addEventListener('submit', (event) => {
      event.preventDefault();
      if (!isTransformEnabled()) {
        return;
      }

      if (transformMode.value !== 'python') {
        return;
      }

      if (!consoleEditorOpen) {
        openPythonEditor();
        return;
      }

      const value = transformInput.value.trim();
      if (!value) {
        setStatus('Enter a tensor expression first.', 'error');
        return;
      }

      startConsoleRun('python', value);
    });

    clearConsoleBtn.addEventListener('click', () => {
      consoleEntries = [];
      renderConsole();
      updateControls();
    });

    toggleConsoleDockBtn.addEventListener('click', () => {
      if (consoleDockOpen) {
        closeConsoleDock();
        return;
      }
      openConsoleDock({ focusPython: transformMode.value === 'python' && consoleEditorOpen });
    });

    closeDockBtn.addEventListener('click', () => {
      closeConsoleDock();
    });

    toggleConsoleBtn.addEventListener('click', () => {
      if (!isTransformEnabled()) {
        return;
      }
      if (transformMode.value !== 'python' || !consoleEditorOpen) {
        openPythonEditor();
        return;
      }
      if (!consoleDockOpen) {
        openConsoleDock({ focusPython: true, silent: true });
      }
      focusPythonEditor();
    });

    closeConsoleBtn.addEventListener('click', () => {
      closePythonEditor();
    });

    runHelperBtn.addEventListener('click', () => {
      if (!isTransformEnabled() || transformMode.value === 'python') {
        return;
      }
      const mode = transformMode.value;
      const value = helperInput.value.trim();
      if (mode !== 'reset' && !value) {
        setStatus(mode === 'reshape' ? 'Enter a target shape first.' : 'Enter a slice expression first.', 'error');
        return;
      }
      startConsoleRun(mode, value);
    });

    insertExampleBtn.addEventListener('click', () => {
      if (transformMode.value === 'python') {
        if (!consoleEditorOpen) {
          openPythonEditor();
        } else if (!consoleDockOpen) {
          openConsoleDock({ focusPython: true, silent: true });
        }
        transformInput.value = pythonExample;
        updateControls({ focusPython: true });
        return;
      }

      const example = helperExamples[transformMode.value] || '';
      if (!example) {
        return;
      }
      helperInput.value = example;
      updateControls();
      helperInput.focus();
      helperInput.setSelectionRange(helperInput.value.length, helperInput.value.length);
    });

    document.querySelectorAll('[data-command]').forEach((button) => {
      button.addEventListener('click', () => {
        vscode.postMessage({ type: 'requestCommand', command: button.dataset.command });
      });
    });

    renderConsole();
    updateControls();
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

function buildConsoleSource(mode: string, value: string): string {
  if (mode === 'python') {
    return value;
  }
  if (mode === 'reshape') {
    parseShapeInput(value);
    return `tensor = tensor.reshape(${value})`;
  }
  if (mode === 'slice') {
    if (!value.trim()) {
      throw new Error('Slice expression cannot be empty.');
    }
    return `tensor = tensor[${value}]`;
  }
  throw new Error(`Unsupported console mode: ${mode}`);
}
