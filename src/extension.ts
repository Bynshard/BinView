import * as path from 'node:path';
import * as vscode from 'vscode';
import { RAW_TYPE_CHOICES } from './dtypes';
import {
  createNpyBinarySource,
  createRawBinarySource,
  TorchBinarySource,
  type BinarySource,
  type RawOpenOptions
} from './dataSource';
import { resolvePythonInterpreter } from './pythonBridge';
import { ViewerSession } from './viewerSession';

class SessionManager implements vscode.Disposable {
  private sessions = new Set<ViewerSession>();
  private activeSession: ViewerSession | undefined;

  constructor(private readonly context: vscode.ExtensionContext) {}

  async openSingle(resource?: vscode.Uri): Promise<void> {
    const uri = resource ?? (await pickOneFile('Select a binary file to inspect'));
    if (!uri) {
      return;
    }

    try {
      const source = await this.createSource(uri);
      const session = await ViewerSession.create(this.context, source, undefined, (active, sourceSession) => {
        if (active) {
          this.activeSession = active;
          return;
        }
        if (this.activeSession === sourceSession) {
          this.activeSession = undefined;
        }
        this.sessions.delete(sourceSession);
      });
      this.sessions.add(session);
      this.context.subscriptions.push(session);
    } catch (error) {
      void vscode.window.showErrorMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async compare(resource?: vscode.Uri): Promise<void> {
    try {
      let leftUri: vscode.Uri | undefined;
      let rightUri: vscode.Uri | undefined;

      if (resource) {
        leftUri = resource;
        rightUri = await pickOneFile('Select the golden/reference file');
      } else {
        const picked = await vscode.window.showOpenDialog({
          canSelectFiles: true,
          canSelectFolders: false,
          canSelectMany: true,
          openLabel: 'Select output and golden files'
        });
        if (!picked || picked.length === 0) {
          return;
        }
        if (picked.length !== 2) {
          throw new Error('Select exactly two files: output and golden.');
        }
        [leftUri, rightUri] = picked;
      }

      if (!leftUri || !rightUri) {
        return;
      }

      const [leftSource, rightSource] = await Promise.all([
        this.createSource(leftUri),
        this.createSource(rightUri)
      ]);
      const session = await ViewerSession.create(this.context, leftSource, rightSource, (active, sourceSession) => {
        if (active) {
          this.activeSession = active;
          return;
        }
        if (this.activeSession === sourceSession) {
          this.activeSession = undefined;
        }
        this.sessions.delete(sourceSession);
      });
      this.sessions.add(session);
      this.context.subscriptions.push(session);
    } catch (error) {
      void vscode.window.showErrorMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async gotoLine(): Promise<void> {
    if (!this.activeSession) {
      void vscode.window.showInformationMessage('No active BinView viewer.');
      return;
    }
    await this.activeSession.gotoLine();
  }

  async reshape(): Promise<void> {
    if (!this.activeSession) {
      return;
    }
    await this.activeSession.reshape();
  }

  async slice(): Promise<void> {
    if (!this.activeSession) {
      return;
    }
    await this.activeSession.slice();
  }

  async resetTransform(): Promise<void> {
    if (!this.activeSession) {
      return;
    }
    await this.activeSession.resetTransform();
  }

  async reload(): Promise<void> {
    if (!this.activeSession) {
      return;
    }
    await this.activeSession.refresh();
  }

  dispose(): void {
    for (const session of this.sessions) {
      void session.dispose();
    }
    this.sessions.clear();
  }

  private async createSource(uri: vscode.Uri): Promise<BinarySource> {
    const filePath = uri.fsPath;
    const fileName = path.basename(filePath).toLowerCase();

    if (fileName.endsWith('.npy')) {
      return createNpyBinarySource(filePath);
    }

    if (isTorchFile(fileName)) {
      const python = await resolvePythonInterpreter(vscode.workspace.getWorkspaceFolder(uri)?.uri);
      if (python) {
        return new TorchBinarySource(filePath, vscode.workspace.getWorkspaceFolder(uri), this.context.extensionUri);
      }
    }

    const format = await this.pickFormat(fileName);
    if (format === 'npy') {
      return createNpyBinarySource(filePath);
    }
    if (format === 'torch') {
      return new TorchBinarySource(filePath, vscode.workspace.getWorkspaceFolder(uri), this.context.extensionUri);
    }

    const rawOptions = await promptRawOptions();
    return createRawBinarySource(filePath, rawOptions);
  }

  private async pickFormat(fileName: string): Promise<'raw' | 'npy' | 'torch'> {
    const pythonAvailable = Boolean(await resolvePythonInterpreter(undefined));
    const choices: Array<{ label: string; description: string; value: 'raw' | 'npy' | 'torch' }> = [
      {
        label: 'Raw binary',
        description: 'Choose dtype manually',
        value: 'raw'
      },
      {
        label: 'NumPy .npy',
        description: 'Read dtype and shape from the file header',
        value: 'npy'
      }
    ];
    if (pythonAvailable) {
      choices.push({
        label: 'PyTorch saved tensor',
        description: 'Use the active VS Code Python interpreter',
        value: 'torch'
      });
    }

    const picked = await vscode.window.showQuickPick(choices, {
      placeHolder: pythonAvailable
        ? `Select the format for ${fileName}`
        : `Select the format for ${fileName} (PyTorch is hidden until a Python interpreter is selected)`
    });

    if (!picked) {
      throw new Error('Format selection was cancelled.');
    }

    return picked.value;
  }
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const manager = new SessionManager(context);
  context.subscriptions.push(manager);

  context.subscriptions.push(
    vscode.commands.registerCommand('binView.openBinary', async (resource?: vscode.Uri) => manager.openSingle(resource)),
    vscode.commands.registerCommand('binView.compareBinary', async (resource?: vscode.Uri) => manager.compare(resource)),
    vscode.commands.registerCommand('binView.gotoLine', async () => manager.gotoLine()),
    vscode.commands.registerCommand('binView.reshape', async () => manager.reshape()),
    vscode.commands.registerCommand('binView.slice', async () => manager.slice()),
    vscode.commands.registerCommand('binView.resetTransform', async () => manager.resetTransform()),
    vscode.commands.registerCommand('binView.reload', async () => manager.reload())
  );
}

export function deactivate(): void {
  // VS Code disposes subscriptions automatically.
}

async function promptRawOptions(): Promise<RawOpenOptions> {
  const dtype = await vscode.window.showQuickPick(
    RAW_TYPE_CHOICES.map((item) => ({
      label: item.label,
      description: item.description,
      value: item.id
    })),
    {
      placeHolder: 'Select the scalar type for this raw binary file'
    }
  );

  if (!dtype) {
    throw new Error('Raw dtype selection was cancelled.');
  }

  const littleEndian = await vscode.window.showQuickPick(
    [
      { label: 'Little endian', value: true },
      { label: 'Big endian', value: false }
    ],
    {
      placeHolder: 'Select the byte order used by this raw binary file'
    }
  );

  if (!littleEndian) {
    throw new Error('Endianness selection was cancelled.');
  }

  return {
    dtypeId: dtype.value,
    littleEndian: littleEndian.value
  };
}

async function pickOneFile(placeHolder: string): Promise<vscode.Uri | undefined> {
  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: true,
    canSelectFolders: false,
    canSelectMany: false,
    openLabel: placeHolder
  });
  if (!picked || picked.length === 0) {
    return undefined;
  }
  return picked[0];
}

function isTorchFile(fileName: string): boolean {
  return fileName.endsWith('.pt') || fileName.endsWith('.pth') || fileName.endsWith('.ckpt');
}
