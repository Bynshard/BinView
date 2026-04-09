export type SourceFormat = 'raw' | 'npy' | 'torch';

export interface DTypeDescriptor {
  id: string;
  label: string;
  byteWidth: number;
  littleEndian: boolean;
}

export interface DisplayScalar {
  index: number;
  value: string;
  hex: string;
  bits: string;
}

const BASE_TYPES: Record<string, Omit<DTypeDescriptor, 'littleEndian'>> = {
  bool: { id: 'bool', label: 'bool', byteWidth: 1 },
  int8: { id: 'int8', label: 'int8', byteWidth: 1 },
  uint8: { id: 'uint8', label: 'uint8', byteWidth: 1 },
  int16: { id: 'int16', label: 'int16', byteWidth: 2 },
  uint16: { id: 'uint16', label: 'uint16', byteWidth: 2 },
  int32: { id: 'int32', label: 'int32', byteWidth: 4 },
  uint32: { id: 'uint32', label: 'uint32', byteWidth: 4 },
  int64: { id: 'int64', label: 'int64', byteWidth: 8 },
  uint64: { id: 'uint64', label: 'uint64', byteWidth: 8 },
  float16: { id: 'float16', label: 'float16', byteWidth: 2 },
  float32: { id: 'float32', label: 'float32', byteWidth: 4 },
  float64: { id: 'float64', label: 'float64', byteWidth: 8 },
  bfloat16: { id: 'bfloat16', label: 'bfloat16', byteWidth: 2 }
};

export const RAW_TYPE_CHOICES: Array<{
  id: string;
  label: string;
  description: string;
}> = [
  { id: 'float32', label: 'float32', description: '4-byte IEEE 754 float' },
  { id: 'float16', label: 'float16', description: '2-byte IEEE 754 float16' },
  { id: 'float64', label: 'float64', description: '8-byte IEEE 754 float' },
  { id: 'bfloat16', label: 'bfloat16', description: '2-byte bfloat16' },
  { id: 'int8', label: 'int8', description: '1-byte signed integer' },
  { id: 'uint8', label: 'uint8', description: '1-byte unsigned integer' },
  { id: 'int16', label: 'int16', description: '2-byte signed integer' },
  { id: 'uint16', label: 'uint16', description: '2-byte unsigned integer' },
  { id: 'int32', label: 'int32', description: '4-byte signed integer' },
  { id: 'uint32', label: 'uint32', description: '4-byte unsigned integer' },
  { id: 'int64', label: 'int64', description: '8-byte signed integer' },
  { id: 'uint64', label: 'uint64', description: '8-byte unsigned integer' },
  { id: 'bool', label: 'bool', description: '1-byte boolean' }
];

export function resolveDType(id: string, littleEndian = true): DTypeDescriptor {
  const base = BASE_TYPES[id];
  if (!base) {
    throw new Error(`Unsupported dtype: ${id}`);
  }

  return { ...base, littleEndian };
}

export function parseNpyDescriptor(descr: string): DTypeDescriptor {
  const endianFlag = descr[0];
  const typeCode = descr[1];
  const size = Number.parseInt(descr.slice(2), 10);

  if (!Number.isFinite(size)) {
    throw new Error(`Unsupported NumPy descriptor: ${descr}`);
  }

  const littleEndian = endianFlag !== '>' && endianFlag !== '!';
  if (typeCode === 'b' && size === 1) {
    return resolveDType('bool', true);
  }
  if (typeCode === 'i') {
    return resolveDType(`int${size * 8}`, littleEndian);
  }
  if (typeCode === 'u') {
    return resolveDType(`uint${size * 8}`, littleEndian);
  }
  if (typeCode === 'f') {
    if (size === 2) {
      return resolveDType('float16', littleEndian);
    }
    if (size === 4) {
      return resolveDType('float32', littleEndian);
    }
    if (size === 8) {
      return resolveDType('float64', littleEndian);
    }
  }

  throw new Error(`Unsupported NumPy descriptor: ${descr}`);
}

function formatBits(buffer: Buffer): string {
  return Array.from(buffer, (byte) => byte.toString(2).padStart(8, '0')).join(' ');
}

function formatHex(buffer: Buffer): string {
  return Array.from(buffer, (byte) => byte.toString(16).padStart(2, '0')).join(' ');
}

function formatFloat(value: number): string {
  if (Number.isNaN(value)) {
    return 'NaN';
  }
  if (!Number.isFinite(value)) {
    return value > 0 ? 'Infinity' : '-Infinity';
  }
  if (Object.is(value, -0)) {
    return '-0';
  }

  const abs = Math.abs(value);
  if ((abs >= 1_000_000 || (abs > 0 && abs < 0.0001))) {
    return value.toExponential(7);
  }
  if (Number.isInteger(value)) {
    return value.toString();
  }
  return Number.parseFloat(value.toPrecision(9)).toString();
}

function decodeFloat16(word: number): number {
  const sign = (word & 0x8000) >> 15;
  const exponent = (word & 0x7c00) >> 10;
  const fraction = word & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign ? -0 : 0;
    }
    return (sign ? -1 : 1) * 2 ** -14 * (fraction / 1024);
  }

  if (exponent === 0x1f) {
    if (fraction === 0) {
      return sign ? -Infinity : Infinity;
    }
    return Number.NaN;
  }

  return (sign ? -1 : 1) * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function decodeBFloat16(word: number): number {
  const full = word << 16;
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint32(0, full, false);
  return view.getFloat32(0, false);
}

export function decodeScalar(buffer: Buffer, dtype: DTypeDescriptor, index: number): DisplayScalar {
  let value: string;
  const little = dtype.littleEndian;

  switch (dtype.id) {
    case 'bool':
      value = buffer[0] === 0 ? 'false' : 'true';
      break;
    case 'int8':
      value = buffer.readInt8(0).toString();
      break;
    case 'uint8':
      value = buffer.readUInt8(0).toString();
      break;
    case 'int16':
      value = (little ? buffer.readInt16LE(0) : buffer.readInt16BE(0)).toString();
      break;
    case 'uint16':
      value = (little ? buffer.readUInt16LE(0) : buffer.readUInt16BE(0)).toString();
      break;
    case 'int32':
      value = (little ? buffer.readInt32LE(0) : buffer.readInt32BE(0)).toString();
      break;
    case 'uint32':
      value = (little ? buffer.readUInt32LE(0) : buffer.readUInt32BE(0)).toString();
      break;
    case 'int64':
      value = (little ? buffer.readBigInt64LE(0) : buffer.readBigInt64BE(0)).toString();
      break;
    case 'uint64':
      value = (little ? buffer.readBigUInt64LE(0) : buffer.readBigUInt64BE(0)).toString();
      break;
    case 'float16': {
      const word = little ? buffer.readUInt16LE(0) : buffer.readUInt16BE(0);
      value = formatFloat(decodeFloat16(word));
      break;
    }
    case 'float32':
      value = formatFloat(little ? buffer.readFloatLE(0) : buffer.readFloatBE(0));
      break;
    case 'float64':
      value = formatFloat(little ? buffer.readDoubleLE(0) : buffer.readDoubleBE(0));
      break;
    case 'bfloat16': {
      const word = little ? buffer.readUInt16LE(0) : buffer.readUInt16BE(0);
      value = formatFloat(decodeBFloat16(word));
      break;
    }
    default:
      throw new Error(`Unsupported dtype: ${dtype.id}`);
  }

  return {
    index,
    value,
    hex: formatHex(buffer),
    bits: formatBits(buffer)
  };
}
