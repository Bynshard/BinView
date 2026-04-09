import * as fs from 'node:fs/promises';
import { parseNpyDescriptor, type DTypeDescriptor } from './dtypes';

export interface NpyHeader {
  dataOffset: number;
  dtype: DTypeDescriptor;
  shape: number[];
  fortranOrder: boolean;
}

function parseShape(shapeText: string): number[] {
  return shapeText
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map((part) => Number.parseInt(part, 10))
    .filter((value) => Number.isFinite(value));
}

export async function readNpyHeader(filePath: string): Promise<NpyHeader> {
  const handle = await fs.open(filePath, 'r');

  try {
    const prefix = Buffer.alloc(512);
    const { bytesRead } = await handle.read(prefix, 0, prefix.length, 0);
    const buffer = prefix.subarray(0, bytesRead);

    if (buffer.length < 12 || buffer.toString('latin1', 0, 6) !== '\u0093NUMPY') {
      throw new Error('File is not a valid .npy file.');
    }

    const major = buffer.readUInt8(6);
    const headerLengthBytes = major >= 2 ? 4 : 2;
    const headerLength = headerLengthBytes === 2 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
    const headerStart = 8 + headerLengthBytes;
    const headerEnd = headerStart + headerLength;

    if (buffer.length < headerEnd) {
      throw new Error('NumPy header is larger than the current probe buffer.');
    }

    const headerText = buffer.toString('latin1', headerStart, headerEnd);
    const descrMatch = /'descr'\s*:\s*'([^']+)'/.exec(headerText);
    const shapeMatch = /'shape'\s*:\s*\(([^)]*)\)/.exec(headerText);
    const orderMatch = /'fortran_order'\s*:\s*(True|False)/.exec(headerText);

    if (!descrMatch || !shapeMatch || !orderMatch) {
      throw new Error('Unsupported .npy header layout.');
    }

    return {
      dataOffset: headerEnd,
      dtype: parseNpyDescriptor(descrMatch[1]),
      shape: parseShape(shapeMatch[1]),
      fortranOrder: orderMatch[1] === 'True'
    };
  } finally {
    await handle.close();
  }
}
