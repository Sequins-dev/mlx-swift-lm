// Copyright © 2026 Apple Inc.

import Foundation
import GGUFIQDequantizer
import MLX

public enum GGUFReaderError: Error, LocalizedError, Equatable {
    case invalidMagic
    case unsupportedVersion(UInt32)
    case truncated(offset: Int, length: Int)
    case invalidUTF8
    case unsupportedMetadataType(UInt32)
    case unsupportedTensorType(UInt32)
    case unsupportedQuantizedTensorType(GGUFReader.TensorType)
    case invalidTensorData(name: String, offset: Int, length: Int)
    case invalidAlignment(UInt32)

    public var errorDescription: String? {
        switch self {
        case .invalidMagic:
            return "Invalid GGUF file magic."
        case .unsupportedVersion(let version):
            return "Unsupported GGUF version \(version)."
        case .truncated(let offset, let length):
            return "GGUF file is truncated at offset \(offset) while reading \(length) bytes."
        case .invalidUTF8:
            return "GGUF file contains an invalid UTF-8 string."
        case .unsupportedMetadataType(let type):
            return "Unsupported GGUF metadata type \(type)."
        case .unsupportedTensorType(let type):
            return "Unsupported GGUF tensor type \(type)."
        case .unsupportedQuantizedTensorType(let type):
            return "GGUF tensor type \(type) is quantized and requires a quant-preserving loader."
        case .invalidTensorData(let name, let offset, let length):
            return
                "GGUF tensor \(name) has invalid data range at offset \(offset) with length \(length)."
        case .invalidAlignment(let alignment):
            return "Invalid GGUF tensor data alignment \(alignment)."
        }
    }
}

public struct GGUFReader {
    public enum MetadataValue: Equatable, Sendable {
        case uint8(UInt8)
        case int8(Int8)
        case uint16(UInt16)
        case int16(Int16)
        case uint32(UInt32)
        case int32(Int32)
        case float32(Float)
        case bool(Bool)
        case string(String)
        case uint64(UInt64)
        case int64(Int64)
        case float64(Double)
        case array([MetadataValue])

        public var stringValue: String? {
            if case .string(let value) = self { value } else { nil }
        }

        public var uint32Value: UInt32? {
            if case .uint32(let value) = self { value } else { nil }
        }
    }

    public enum TensorType: UInt32, Sendable {
        case f32 = 0
        case f16 = 1
        case q4_0 = 2
        case q4_1 = 3
        case q5_0 = 6
        case q5_1 = 7
        case q8_0 = 8
        case q8_1 = 9
        case q2K = 10
        case q3K = 11
        case q4K = 12
        case q5K = 13
        case q6K = 14
        case q8K = 15
        case iq2XS = 17
        case iq3XXS = 18
        case iq3S = 21
        case iq2S = 22
        case iq4XS = 23
        case i8 = 24
        case i16 = 25
        case i32 = 26
        case i64 = 27
        case f64 = 28
        case bf16 = 30

        var kQuantization: GGUFKQuantization? {
            switch self {
            case .q8_0: .q8_0
            case .q2K: .q2K
            case .q3K: .q3K
            case .q4K: .q4K
            case .q5K: .q5K
            case .q6K: .q6K
            case .q8K: .q8K
            default: nil
            }
        }
    }

    public struct Tensor: Equatable, Sendable {
        public var name: String
        public var shape: [Int]
        public var type: TensorType
        public var offset: UInt64
        public var absoluteDataOffset: Int
    }

    public let url: URL
    public let version: UInt32
    public let metadata: [String: MetadataValue]
    public let tensors: [Tensor]
    public let alignment: Int
    public let dataOffset: Int
    private let data: Data

    public init(url: URL) throws {
        self.url = url

        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        self.data = data
        var cursor = Cursor(data: data)

        let magic = try cursor.readBytes(count: 4)
        guard magic == Array("GGUF".utf8) else {
            throw GGUFReaderError.invalidMagic
        }

        let version = try cursor.readUInt32()
        guard version == 2 || version == 3 else {
            throw GGUFReaderError.unsupportedVersion(version)
        }
        self.version = version

        let tensorCount = try cursor.readUInt64()
        let metadataCount = try cursor.readUInt64()

        var metadata = [String: MetadataValue]()
        for _ in 0 ..< metadataCount {
            let key = try cursor.readString()
            let type = try cursor.readUInt32()
            metadata[key] = try cursor.readMetadataValue(type: type)
        }
        self.metadata = metadata

        let alignmentValue = metadata["general.alignment"]?.uint32Value ?? 32
        guard alignmentValue > 0 else {
            throw GGUFReaderError.invalidAlignment(alignmentValue)
        }
        self.alignment = Int(alignmentValue)

        struct TensorDescriptor {
            var name: String
            var shape: [Int]
            var type: TensorType
            var offset: UInt64
        }

        var descriptors = [TensorDescriptor]()
        descriptors.reserveCapacity(Int(tensorCount))
        for _ in 0 ..< tensorCount {
            let name = try cursor.readString()
            let dimensionCount = try cursor.readUInt32()
            var dimensions = [Int]()
            dimensions.reserveCapacity(Int(dimensionCount))
            for _ in 0 ..< dimensionCount {
                dimensions.append(Int(try cursor.readUInt64()))
            }
            let rawType = try cursor.readUInt32()
            guard let type = TensorType(rawValue: rawType) else {
                throw GGUFReaderError.unsupportedTensorType(rawType)
            }
            let offset = try cursor.readUInt64()
            descriptors.append(
                TensorDescriptor(
                    name: name, shape: dimensions.reversed(), type: type, offset: offset))
        }

        let dataOffset = cursor.offset.roundedUp(toMultiple: Int(alignmentValue))
        self.dataOffset = dataOffset
        self.tensors = descriptors.map {
            Tensor(
                name: $0.name, shape: $0.shape, type: $0.type, offset: $0.offset,
                absoluteDataOffset: dataOffset + Int($0.offset))
        }
    }

    public func loadArrays() throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        result.reserveCapacity(tensors.count)

        for tensor in tensors {
            if let arrays = try loadAffineQuantizedArrays(tensor) {
                result[tensor.name] = arrays.weight
                let namePrefix = tensor.name.droppingWeightSuffix
                result["\(namePrefix).scales"] = arrays.scales
                result["\(namePrefix).biases"] = arrays.biases
                continue
            }

            if let array = try loadIQDequantizedArray(tensor) {
                result[tensor.name] = array
                continue
            }

            guard let dtype = tensor.type.mlxDType else {
                throw GGUFReaderError.unsupportedQuantizedTensorType(tensor.type)
            }
            let byteCount = tensor.shape.reduce(1, *) * dtype.size
            let start = tensor.absoluteDataOffset
            let end = start + byteCount
            guard start >= 0, end <= data.count else {
                throw GGUFReaderError.invalidTensorData(
                    name: tensor.name, offset: start, length: byteCount)
            }
            let tensorData = data.subdata(in: start ..< end)
            result[tensor.name] = MLXArray(tensorData, tensor.shape, dtype: dtype)
        }

        return result
    }

    private func loadAffineQuantizedArrays(_ tensor: Tensor) throws -> GGUFAffineQuantizedArrays? {
        guard let format = tensor.type.kQuantization else {
            return nil
        }

        let byteCount = try ggufKDataByteCount(format: format, shape: tensor.shape)
        let start = tensor.absoluteDataOffset
        let end = start + byteCount
        guard start >= 0, end <= data.count else {
            throw GGUFReaderError.invalidTensorData(
                name: tensor.name, offset: start, length: byteCount)
        }
        return try data.withUnsafeBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            let tensorBytes = UnsafeBufferPointer(rebasing: bytes[start ..< end])
            return try ggufKToAffineQuantizedArrays(tensorBytes, format: format, shape: tensor.shape)
        }
    }

    private func loadIQDequantizedArray(_ tensor: Tensor) throws -> MLXArray? {
        guard let iqType = tensor.type.iqType,
            let byteCount = tensor.type.iqDataByteCount(shape: tensor.shape)
        else {
            return nil
        }

        let start = tensor.absoluteDataOffset
        let end = start + byteCount
        guard start >= 0, end <= data.count else {
            throw GGUFReaderError.invalidTensorData(
                name: tensor.name, offset: start, length: byteCount)
        }

        let valueCount = tensor.shape.reduce(1, *)
        var values = [Float](repeating: 0, count: valueCount)
        let ok = data.withUnsafeBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            return values.withUnsafeMutableBufferPointer { output in
                gguf_iq_dequantize(
                    iqType,
                    UnsafeBufferPointer(rebasing: bytes[start ..< end]).baseAddress,
                    output.baseAddress,
                    Int64(valueCount))
            }
        }
        guard ok != 0 else {
            throw GGUFReaderError.unsupportedQuantizedTensorType(tensor.type)
        }

        var float16Data = Data()
        float16Data.reserveCapacity(valueCount * MemoryLayout<Float16>.size)
        for value in values {
            float16Data.appendLittleEndian(Float16(value).bitPattern)
        }
        return MLXArray(float16Data, tensor.shape, dtype: .float16)
    }

    public func mapWeightNames(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let architecture = metadata["general.architecture"]?.stringValue
        return Dictionary(
            uniqueKeysWithValues: weights.map { key, value in
                (Self.mapTensorName(key, architecture: architecture), value)
            })
    }

    public static func mapTensorName(_ name: String) -> String {
        mapTensorName(name, architecture: nil)
    }

    public static func mapTensorName(_ name: String, architecture: String?) -> String {
        if let auxiliary = mapQuantizedAuxiliaryName(name, architecture: architecture) {
            return auxiliary
        }

        if architecture == "gemma4" {
            return mapGemma4TensorName(name)
        }

        switch name {
        case "token_embd.weight":
            return "model.embed_tokens.weight"
        case "output.weight":
            return "lm_head.weight"
        case "output_norm.weight":
            return "model.norm.weight"
        default:
            break
        }

        guard name.hasPrefix("blk.") else {
            return name
        }

        let parts = name.split(separator: ".")
        guard parts.count >= 4, let layer = Int(parts[1]) else {
            return name
        }

        let prefix = "model.layers.\(layer)"
        let suffix = parts.dropFirst(2).joined(separator: ".")
        switch suffix {
        case "attn_norm.weight":
            return "\(prefix).input_layernorm.weight"
        case "ffn_norm.weight":
            return "\(prefix).post_attention_layernorm.weight"
        case "attn_q.weight":
            return "\(prefix).self_attn.q_proj.weight"
        case "attn_k.weight":
            return "\(prefix).self_attn.k_proj.weight"
        case "attn_v.weight":
            return "\(prefix).self_attn.v_proj.weight"
        case "attn_output.weight":
            return "\(prefix).self_attn.o_proj.weight"
        case "ffn_gate.weight":
            return "\(prefix).mlp.gate_proj.weight"
        case "ffn_up.weight":
            return "\(prefix).mlp.up_proj.weight"
        case "ffn_down.weight":
            return "\(prefix).mlp.down_proj.weight"
        default:
            return name
        }
    }

    private static func mapQuantizedAuxiliaryName(_ name: String, architecture: String?) -> String?
    {
        for suffix in [".scales", ".biases"] {
            guard name.hasSuffix(suffix) else { continue }
            let base = String(name.dropLast(suffix.count))
            let weightName = mapTensorName("\(base).weight", architecture: architecture)
            guard weightName.hasSuffix(".weight") else {
                return "\(weightName)\(suffix)"
            }
            return "\(String(weightName.dropLast(".weight".count)))\(suffix)"
        }
        return nil
    }

    private static func mapGemma4TensorName(_ name: String) -> String {
        switch name {
        case "token_embd.weight":
            return "language_model.model.embed_tokens.weight"
        case "output.weight":
            return "language_model.lm_head.weight"
        case "output_norm.weight":
            return "language_model.model.norm.weight"
        case "rope_freqs.weight":
            return "language_model.model.self_attn.rotary_emb.weight"
        case "per_layer_model_proj.weight":
            return "language_model.model.per_layer_model_projection.weight"
        case "per_layer_proj_norm.weight":
            return "language_model.model.per_layer_projection_norm.weight"
        case "per_layer_token_embd.weight":
            return "language_model.model.embed_tokens_per_layer.weight"
        default:
            break
        }

        guard name.hasPrefix("blk.") else {
            return name
        }

        let parts = name.split(separator: ".")
        guard parts.count >= 4, let layer = Int(parts[1]) else {
            return name
        }

        let prefix = "language_model.model.layers.\(layer)"
        let suffix = parts.dropFirst(2).joined(separator: ".")
        switch suffix {
        case "attn_norm.weight":
            return "\(prefix).input_layernorm.weight"
        case "post_attention_norm.weight":
            return "\(prefix).post_attention_layernorm.weight"
        case "ffn_norm.weight":
            return "\(prefix).pre_feedforward_layernorm.weight"
        case "post_ffw_norm.weight":
            return "\(prefix).post_feedforward_layernorm.weight"
        case "post_norm.weight":
            return "\(prefix).post_per_layer_input_norm.weight"
        case "attn_q.weight":
            return "\(prefix).self_attn.q_proj.weight"
        case "attn_k.weight":
            return "\(prefix).self_attn.k_proj.weight"
        case "attn_v.weight":
            return "\(prefix).self_attn.v_proj.weight"
        case "attn_output.weight":
            return "\(prefix).self_attn.o_proj.weight"
        case "attn_q_norm.weight":
            return "\(prefix).self_attn.q_norm.weight"
        case "attn_k_norm.weight":
            return "\(prefix).self_attn.k_norm.weight"
        case "ffn_gate.weight":
            return "\(prefix).mlp.gate_proj.weight"
        case "ffn_up.weight":
            return "\(prefix).mlp.up_proj.weight"
        case "ffn_down.weight":
            return "\(prefix).mlp.down_proj.weight"
        case "inp_gate.weight":
            return "\(prefix).per_layer_input_gate.weight"
        case "proj.weight":
            return "\(prefix).per_layer_projection.weight"
        case "layer_output_scale.weight":
            return "\(prefix).layer_scalar"
        default:
            return name
        }
    }
}

extension String {
    fileprivate var droppingWeightSuffix: String {
        let suffix = ".weight"
        guard hasSuffix(suffix) else {
            return self
        }
        return String(dropLast(suffix.count))
    }
}

extension GGUFReader.TensorType {
    fileprivate var iqType: GGUFIQType? {
        switch self {
        case .iq2XS:
            GGUFIQTypeIQ2XS
        case .iq3XXS:
            GGUFIQTypeIQ3XXS
        case .iq3S:
            GGUFIQTypeIQ3S
        case .iq2S:
            GGUFIQTypeIQ2S
        case .iq4XS:
            GGUFIQTypeIQ4XS
        default:
            nil
        }
    }

    fileprivate var mlxDType: DType? {
        switch self {
        case .f32:
            return .float32
        case .f16:
            return .float16
        case .bf16:
            return .bfloat16
        case .i8:
            return .int8
        case .i16:
            return .int16
        case .i32:
            return .int32
        case .i64:
            return .int64
        case .f64:
            return .float64
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .q2K, .q3K, .q4K, .q5K, .q6K, .q8K,
            .iq2XS, .iq3XXS, .iq3S, .iq2S, .iq4XS:
            return nil
        }
    }

    fileprivate func iqDataByteCount(shape: [Int]) -> Int? {
        let blockSize: Int
        switch self {
        case .iq2XS:
            blockSize = Int(gguf_iq_block_size(GGUFIQTypeIQ2XS))
        case .iq3XXS:
            blockSize = Int(gguf_iq_block_size(GGUFIQTypeIQ3XXS))
        case .iq3S:
            blockSize = Int(gguf_iq_block_size(GGUFIQTypeIQ3S))
        case .iq2S:
            blockSize = Int(gguf_iq_block_size(GGUFIQTypeIQ2S))
        case .iq4XS:
            blockSize = Int(gguf_iq_block_size(GGUFIQTypeIQ4XS))
        default:
            return nil
        }
        guard blockSize > 0 else { return nil }

        guard shape.count >= 2, let columns = shape.last, columns % 256 == 0 else {
            return nil
        }
        let rows = shape.dropLast().reduce(1, *)
        return rows * (columns / 256) * blockSize
    }
}

extension Data {
    fileprivate mutating func appendLittleEndian(_ value: UInt16) {
        append(UInt8(truncatingIfNeeded: value))
        append(UInt8(truncatingIfNeeded: value >> 8))
    }
}

private struct Cursor {
    var data: Data
    var offset = 0

    mutating func readBytes(count: Int) throws -> [UInt8] {
        guard offset + count <= data.count else {
            throw GGUFReaderError.truncated(offset: offset, length: count)
        }
        defer { offset += count }
        return Array(data[offset ..< offset + count])
    }

    mutating func readUInt8() throws -> UInt8 {
        try readInteger(UInt8.self)
    }

    mutating func readInt8() throws -> Int8 {
        try readInteger(Int8.self)
    }

    mutating func readUInt16() throws -> UInt16 {
        try readInteger(UInt16.self)
    }

    mutating func readInt16() throws -> Int16 {
        try readInteger(Int16.self)
    }

    mutating func readUInt32() throws -> UInt32 {
        try readInteger(UInt32.self)
    }

    mutating func readInt32() throws -> Int32 {
        try readInteger(Int32.self)
    }

    mutating func readUInt64() throws -> UInt64 {
        try readInteger(UInt64.self)
    }

    mutating func readInt64() throws -> Int64 {
        try readInteger(Int64.self)
    }

    mutating func readFloat32() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    mutating func readFloat64() throws -> Double {
        Double(bitPattern: try readUInt64())
    }

    mutating func readString() throws -> String {
        let length = Int(try readUInt64())
        let bytes = try readBytes(count: length)
        guard let value = String(bytes: bytes, encoding: .utf8) else {
            throw GGUFReaderError.invalidUTF8
        }
        return value
    }

    mutating func readMetadataValue(type: UInt32) throws -> GGUFReader.MetadataValue {
        switch type {
        case 0: return .uint8(try readUInt8())
        case 1: return .int8(try readInt8())
        case 2: return .uint16(try readUInt16())
        case 3: return .int16(try readInt16())
        case 4: return .uint32(try readUInt32())
        case 5: return .int32(try readInt32())
        case 6: return .float32(try readFloat32())
        case 7: return .bool(try readUInt8() != 0)
        case 8: return .string(try readString())
        case 9:
            let elementType = try readUInt32()
            let count = try readUInt64()
            var values = [GGUFReader.MetadataValue]()
            values.reserveCapacity(Int(count))
            for _ in 0 ..< count {
                values.append(try readMetadataValue(type: elementType))
            }
            return .array(values)
        case 10: return .uint64(try readUInt64())
        case 11: return .int64(try readInt64())
        case 12: return .float64(try readFloat64())
        default: throw GGUFReaderError.unsupportedMetadataType(type)
        }
    }

    private mutating func readInteger<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard offset + size <= data.count else {
            throw GGUFReaderError.truncated(offset: offset, length: size)
        }
        let value = data.withUnsafeBytes { buffer in
            buffer.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
        offset += size
        return T(littleEndian: value)
    }
}

extension Int {
    fileprivate func roundedUp(toMultiple alignment: Int) -> Int {
        let remainder = self % alignment
        return remainder == 0 ? self : self + alignment - remainder
    }
}
