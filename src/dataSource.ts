import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { decodeScalar, resolveDType, type DTypeDescriptor, type DisplayScalar, type SourceFormat } from './dtypes';
import { readNpyHeader } from './npy';
import { TorchTensorBridge, resolvePythonInterpreter } from './pythonBridge';

export interface SourceMetadata {
  label: string;
  path: string;
  format: SourceFormat;
  dtypeLabel: string;
  shape: number[];
  totalElements: number;
  transformSummary?: string;
  notes: string[];
  canTransform: boolean;
}

export interface BinarySource {
  getMetadata(): Promise<SourceMetadata>;
  readScalars(startIndex: number, count: number): Promise<DisplayScalar[]>;
  reshape?(shape: number[]): Promise<SourceMetadata>;
  slice?(expression: string): Promise<SourceMetadata>;
  resetTransform?(): Promise<SourceMetadata>;
  dispose(): Promise<void>;
}

export interface RawOpenOptions {
  dtypeId: string;
  littleEndian: boolean;
}

class FileBinarySource implements BinarySource {
  private handlePromise: Promise<fs.FileHandle> | undefined;
  private metadataPromise: Promise<SourceMetadata> | undefined;

  constructor(
    private readonly filePath: string,
    private readonly format: SourceFormat,
    private readonly dtypeLoader: () => Promise<{ dtype: DTypeDescriptor; offset: number; shape: number[]; notes: string[] }>
  ) {}

  async getMetadata(): Promise<SourceMetadata> {
    if (!this.metadataPromise) {
      this.metadataPromise = this.loadMetadata();
    }
    return this.metadataPromise;
  }

  async readScalars(startIndex: number, count: number): Promise<DisplayScalar[]> {
    const meta = await this.getMetadata();
    const descriptor = await this.dtypeLoader();
    const start = Math.max(0, Math.min(startIndex, meta.totalElements));
    const actualCount = Math.max(0, Math.min(count, meta.totalElements - start));
    if (actualCount === 0) {
      return [];
    }

    const byteStart = descriptor.offset + start * descriptor.dtype.byteWidth;
    const byteCount = actualCount * descriptor.dtype.byteWidth;
    const target = Buffer.allocUnsafe(byteCount);
    const handle = await this.getHandle();
    await handle.read(target, 0, byteCount, byteStart);

    const values: DisplayScalar[] = [];
    for (let index = 0; index < actualCount; index += 1) {
      const itemStart = index * descriptor.dtype.byteWidth;
      const item = target.subarray(itemStart, itemStart + descriptor.dtype.byteWidth);
      values.push(decodeScalar(item, descriptor.dtype, start + index));
    }
    return values;
  }

  async dispose(): Promise<void> {
    if (!this.handlePromise) {
      return;
    }
    const handle = await this.handlePromise;
    await handle.close();
  }

  private async loadMetadata(): Promise<SourceMetadata> {
    const stats = await fs.stat(this.filePath);
    const descriptor = await this.dtypeLoader();
    const totalBytes = Math.max(0, stats.size - descriptor.offset);
    const totalElements = Math.floor(totalBytes / descriptor.dtype.byteWidth);
    const shape = descriptor.shape.length > 0 ? descriptor.shape : [totalElements];

    return {
      label: path.basename(this.filePath),
      path: this.filePath,
      format: this.format,
      dtypeLabel: `${descriptor.dtype.label}${descriptor.dtype.byteWidth > 1 ? descriptor.dtype.littleEndian ? ' LE' : ' BE' : ''}`,
      shape,
      totalElements,
      notes: descriptor.notes,
      canTransform: false
    };
  }

  private async getHandle(): Promise<fs.FileHandle> {
    if (!this.handlePromise) {
      this.handlePromise = fs.open(this.filePath, 'r');
    }
    return this.handlePromise;
  }
}

export class TorchBinarySource implements BinarySource {
  private bridge: TorchTensorBridge | undefined;
  private metadata: SourceMetadata | undefined;

  constructor(
    private readonly filePath: string,
    private readonly workspaceFolder: vscode.WorkspaceFolder | undefined,
    private readonly extensionUri: vscode.Uri
  ) {}

  async getMetadata(): Promise<SourceMetadata> {
    if (this.metadata) {
      return this.metadata;
    }

    const bridge = await this.getBridge();
    const selection = await bridge.initialize(this.filePath);
    this.metadata = {
      label: path.basename(this.filePath),
      path: this.filePath,
      format: 'torch',
      dtypeLabel: selection.dtype,
      shape: selection.shape,
      totalElements: selection.totalElements,
      transformSummary: selection.transformSummary,
      notes: selection.notes,
      canTransform: true
    };
    return this.metadata;
  }

  async readScalars(startIndex: number, count: number): Promise<DisplayScalar[]> {
    const bridge = await this.getBridge();
    return bridge.fetch(startIndex, count);
  }

  async reshape(shape: number[]): Promise<SourceMetadata> {
    const bridge = await this.getBridge();
    const result = await bridge.reshape(shape);
    this.metadata = {
      ...(await this.getMetadata()),
      dtypeLabel: result.dtype,
      shape: result.shape,
      totalElements: result.totalElements,
      transformSummary: result.transformSummary,
      notes: result.notes
    };
    return this.metadata;
  }

  async slice(expression: string): Promise<SourceMetadata> {
    const bridge = await this.getBridge();
    const result = await bridge.slice(expression);
    this.metadata = {
      ...(await this.getMetadata()),
      dtypeLabel: result.dtype,
      shape: result.shape,
      totalElements: result.totalElements,
      transformSummary: result.transformSummary,
      notes: result.notes
    };
    return this.metadata;
  }

  async resetTransform(): Promise<SourceMetadata> {
    const bridge = await this.getBridge();
    const result = await bridge.reset();
    this.metadata = {
      ...(await this.getMetadata()),
      dtypeLabel: result.dtype,
      shape: result.shape,
      totalElements: result.totalElements,
      transformSummary: result.transformSummary,
      notes: result.notes
    };
    return this.metadata;
  }

  async dispose(): Promise<void> {
    if (!this.bridge) {
      return;
    }
    await this.bridge.dispose();
  }

  private async getBridge(): Promise<TorchTensorBridge> {
    if (this.bridge) {
      return this.bridge;
    }

    const interpreter = await resolvePythonInterpreter(this.workspaceFolder?.uri);
    if (!interpreter) {
      throw new Error('No active Python interpreter was found. Install the VS Code Python extension and select an interpreter first.');
    }

    const helper = vscode.Uri.joinPath(this.extensionUri, 'resources', 'python', 'binview_tensor_bridge.py').fsPath;
    this.bridge = new TorchTensorBridge(interpreter, helper, this.workspaceFolder?.uri);
    return this.bridge;
  }
}

export async function createRawBinarySource(filePath: string, options: RawOpenOptions): Promise<BinarySource> {
  return new FileBinarySource(filePath, 'raw', async () => ({
    dtype: resolveDType(options.dtypeId, options.littleEndian),
    offset: 0,
    shape: [],
    notes: []
  }));
}

export async function createNpyBinarySource(filePath: string): Promise<BinarySource> {
  let headerCache: Awaited<ReturnType<typeof readNpyHeader>> | undefined;
  const getHeader = async () => {
    if (!headerCache) {
      headerCache = await readNpyHeader(filePath);
    }
    return headerCache;
  };

  return new FileBinarySource(filePath, 'npy', async () => {
    const header = await getHeader();
    return {
      dtype: header.dtype,
      offset: header.dataOffset,
      shape: header.shape,
      notes: header.fortranOrder ? ['NumPy file uses Fortran order; values are shown in the stored linear order.'] : []
    };
  });
}
