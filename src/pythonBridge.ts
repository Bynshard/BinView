import * as cp from 'node:child_process';
import * as readline from 'node:readline';
import * as vscode from 'vscode';
import { PythonExtension } from '@vscode/python-extension';
import type { DisplayScalar } from './dtypes';

interface BridgeResponse<T> {
  id: number;
  ok: boolean;
  result?: T;
  error?: string;
}

interface CandidateInfo {
  id: number;
  label: string;
  dtype: string;
  shape: number[];
}

interface MetadataResponse {
  dtype: string;
  shape: number[];
  totalElements: number;
  transformSummary?: string;
  notes: string[];
}

interface OpenResponse extends MetadataResponse {
  candidates: CandidateInfo[];
}

export class TorchTensorBridge {
  private process: cp.ChildProcessWithoutNullStreams | undefined;
  private requestId = 0;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  private stderr = '';

  constructor(
    private readonly pythonPath: string,
    private readonly scriptPath: string,
    private readonly workspaceUri: vscode.Uri | undefined
  ) {}

  async initialize(filePath: string): Promise<MetadataResponse> {
    const openResult = await this.send<OpenResponse>('open', { path: filePath });
    if (openResult.candidates.length > 1) {
      const picked = await vscode.window.showQuickPick(
        openResult.candidates.map((candidate) => ({
          label: candidate.label,
          description: `${candidate.dtype} ${formatShape(candidate.shape)}`,
          candidate
        })),
        {
          placeHolder: 'Select the tensor to inspect from this torch file'
        }
      );

      if (!picked) {
        throw new Error('Tensor selection was cancelled.');
      }

      return this.send<MetadataResponse>('select', { candidateId: picked.candidate.id });
    }

    if (openResult.candidates.length === 1) {
      return this.send<MetadataResponse>('select', { candidateId: openResult.candidates[0].id });
    }

    throw new Error('No tensor-like object was found in this torch file.');
  }

  fetch(startIndex: number, count: number): Promise<DisplayScalar[]> {
    return this.send<DisplayScalar[]>('fetch', { start: startIndex, count });
  }

  reshape(shape: number[]): Promise<MetadataResponse> {
    return this.send<MetadataResponse>('reshape', { shape });
  }

  slice(expression: string): Promise<MetadataResponse> {
    return this.send<MetadataResponse>('slice', { expression });
  }

  reset(): Promise<MetadataResponse> {
    return this.send<MetadataResponse>('reset', {});
  }

  async dispose(): Promise<void> {
    if (!this.process) {
      return;
    }
    try {
      await this.send('close', {});
    } catch {
      // Ignore shutdown failures and terminate below.
    }
    this.process.kill();
    this.process = undefined;
  }

  private async send<T>(command: string, payload: Record<string, unknown>): Promise<T> {
    await this.ensureProcess();

    const process = this.process;
    if (!process) {
      throw new Error('Torch bridge process is not available.');
    }

    this.requestId += 1;
    const id = this.requestId;
    const packet = JSON.stringify({ id, command, ...payload }) + '\n';

    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject
      });

      process.stdin.write(packet, 'utf8', (error) => {
        if (error) {
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  private async ensureProcess(): Promise<void> {
    if (this.process) {
      return;
    }

    const child = cp.spawn(this.pythonPath, ['-u', this.scriptPath], {
      stdio: 'pipe',
      cwd: this.workspaceUri?.fsPath
    });

    const reader = readline.createInterface({ input: child.stdout });
    reader.on('line', (line) => {
      let message: BridgeResponse<unknown>;
      try {
        message = JSON.parse(line) as BridgeResponse<unknown>;
      } catch (error) {
        this.rejectAll(new Error(`Invalid torch bridge response: ${String(error)}`));
        return;
      }

      const pending = this.pending.get(message.id);
      if (!pending) {
        return;
      }

      this.pending.delete(message.id);
      if (message.ok) {
        pending.resolve(message.result);
        return;
      }

      pending.reject(new Error(message.error ?? 'Unknown torch bridge error.'));
    });

    child.stderr.on('data', (chunk) => {
      this.stderr += chunk.toString('utf8');
      if (this.stderr.length > 10_000) {
        this.stderr = this.stderr.slice(-10_000);
      }
    });

    child.on('exit', (code, signal) => {
      const suffix = this.stderr.trim() ? `\n${this.stderr.trim()}` : '';
      this.rejectAll(new Error(`Torch bridge exited unexpectedly (code=${code}, signal=${signal}).${suffix}`));
      this.process = undefined;
    });

    this.process = child;
  }

  private rejectAll(error: Error): void {
    for (const entry of this.pending.values()) {
      entry.reject(error);
    }
    this.pending.clear();
  }
}

function formatShape(shape: number[]): string {
  return `[${shape.join(', ')}]`;
}

export async function resolvePythonInterpreter(resource: vscode.Uri | undefined): Promise<string | undefined> {
  const pythonExtension = vscode.extensions.getExtension('ms-python.python');
  if (pythonExtension) {
    try {
      const api = await PythonExtension.api();
      const environments = api.environments as {
        getActiveEnvironmentPath: (...args: unknown[]) => { path?: string } | undefined;
        resolveEnvironment: (environment: unknown) => Promise<{ executable?: { uri?: vscode.Uri } } | undefined>;
      };
      const environmentPath = environments.getActiveEnvironmentPath(resource) ?? environments.getActiveEnvironmentPath();
      if (!environmentPath) {
        return undefined;
      }

      const resolved = await environments.resolveEnvironment(environmentPath);
      const executable = resolved?.executable?.uri?.fsPath;
      if (executable) {
        return executable;
      }

      if (typeof environmentPath.path === 'string' && environmentPath.path.length > 0) {
        return environmentPath.path;
      }
    } catch {
      // Fall through to legacy settings/commands.
    }
  }

  const configPath = vscode.workspace.getConfiguration('python', resource).get<string>('defaultInterpreterPath');
  if (configPath) {
    return configPath;
  }

  try {
    const commandResult = await vscode.commands.executeCommand<string>('python.interpreterPath');
    if (commandResult) {
      return commandResult;
    }
  } catch {
    // Ignore legacy command failures.
  }

  return undefined;
}
