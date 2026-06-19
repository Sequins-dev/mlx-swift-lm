import Foundation
import MLX
import MLXLMCommon
import Testing

extension Data {
    fileprivate mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }

    fileprivate mutating func appendLE(_ value: Float) {
        appendLE(value.bitPattern)
    }

    fileprivate mutating func appendString(_ value: String) {
        let bytes = Array(value.utf8)
        appendLE(UInt64(bytes.count))
        append(contentsOf: bytes)
    }

    fileprivate mutating func pad(toMultiple alignment: Int) {
        let remainder = count % alignment
        if remainder != 0 {
            append(contentsOf: repeatElement(UInt8(0), count: alignment - remainder))
        }
    }
}

@Suite("GGUFReader")
struct GGUFReaderTests {
    @Test func parsesMetadataAndTensorDescriptors() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-reader-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = Data()
        data.append(contentsOf: Array("GGUF".utf8))
        data.appendLE(UInt32(3))
        data.appendLE(UInt64(1))  // tensor count
        data.appendLE(UInt64(3))  // metadata count

        data.appendString("general.architecture")
        data.appendLE(UInt32(8))  // string
        data.appendString("llama")

        data.appendString("llama.context_length")
        data.appendLE(UInt32(4))  // uint32
        data.appendLE(UInt32(2048))

        data.appendString("general.alignment")
        data.appendLE(UInt32(4))  // uint32
        data.appendLE(UInt32(32))

        data.appendString("token_embd.weight")
        data.appendLE(UInt32(2))  // ndim
        data.appendLE(UInt64(4))
        data.appendLE(UInt64(2))
        data.appendLE(UInt32(1))  // f16
        data.appendLE(UInt64(0))  // relative data offset

        data.pad(toMultiple: 32)
        let absoluteDataOffset = data.count
        data.append(contentsOf: repeatElement(UInt8(0), count: 16))

        try data.write(to: url)

        let reader = try GGUFReader(url: url)

        #expect(reader.version == 3)
        #expect(reader.metadata["general.architecture"]?.stringValue == "llama")
        #expect(reader.metadata["llama.context_length"]?.uint32Value == 2048)
        #expect(reader.alignment == 32)
        #expect(reader.tensors.count == 1)
        #expect(reader.tensors[0].name == "token_embd.weight")
        #expect(reader.tensors[0].shape == [2, 4])
        #expect(reader.tensors[0].type == .f16)
        #expect(reader.tensors[0].absoluteDataOffset == absoluteDataOffset)
    }

    @Test func validatesPlainTensorDataRangeBeforeLoading() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-reader-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = header(tensorType: .f32)
        data.pad(toMultiple: 32)
        data.appendLE(Float(1.25))

        try data.write(to: url)

        #expect(
            throws: GGUFReaderError.invalidTensorData(
                name: "token_embd.weight", offset: data.count - 4, length: 24)
        ) {
            _ = try GGUFReader(url: url).loadArrays()
        }
    }

    @Test func validatesQ4KTensorDataRangeBeforeLoading() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-reader-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = header(tensorType: .q4K, dimensions: [256, 1])
        data.pad(toMultiple: 32)

        try data.write(to: url)

        #expect(
            throws: GGUFReaderError.invalidTensorData(
                name: "token_embd.weight", offset: data.count, length: 144)
        ) {
            _ = try GGUFReader(url: url).loadArrays()
        }
    }

    @Test func loadsBF16TensorData() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-reader-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = header(tensorType: .bf16)
        data.pad(toMultiple: 32)
        data.append(contentsOf: repeatElement(UInt8(0), count: 12))

        try data.write(to: url)

        let arrays = try GGUFReader(url: url).loadArrays()

        #expect(arrays["token_embd.weight"]?.shape == [2, 3])
        #expect(arrays["token_embd.weight"]?.dtype == .bfloat16)
    }

    @Test func parsesIQTensorTypesAndValidatesDataRangeBeforeLoading() throws {
        let cases: [(GGUFReader.TensorType, Int)] = [
            (.iq2XS, 74),
            (.iq3XXS, 98),
            (.iq3S, 110),
            (.iq2S, 84),
            (.iq4XS, 136),
        ]

        for (tensorType, byteCount) in cases {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("gguf-reader-\(tensorType)-\(UUID().uuidString).gguf")
            defer { try? FileManager.default.removeItem(at: url) }

            var data = header(tensorType: tensorType, dimensions: [256, 1])
            data.pad(toMultiple: 32)
            data.append(contentsOf: repeatElement(UInt8(0), count: byteCount - 1))

            try data.write(to: url)

            #expect(GGUFReader.TensorType(rawValue: tensorType.rawValue) == tensorType)
            #expect(
                throws: GGUFReaderError.invalidTensorData(
                    name: "token_embd.weight", offset: data.count - byteCount + 1, length: byteCount)
            ) {
                _ = try GGUFReader(url: url).loadArrays()
            }
        }
    }

    @Test func loadsIQTensorTypesAsFloat16Arrays() throws {
        let cases: [(GGUFReader.TensorType, Int)] = [
            (.iq2XS, 74),
            (.iq3XXS, 98),
            (.iq3S, 110),
            (.iq2S, 84),
            (.iq4XS, 136),
        ]

        for (tensorType, byteCount) in cases {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("gguf-reader-\(tensorType)-\(UUID().uuidString).gguf")
            defer { try? FileManager.default.removeItem(at: url) }

            var data = header(tensorType: tensorType, dimensions: [256, 1])
            data.pad(toMultiple: 32)
            data.append(contentsOf: repeatElement(UInt8(0), count: byteCount))

            try data.write(to: url)

            let arrays = try GGUFReader(url: url).loadArrays()

            #expect(arrays["token_embd.weight"]?.shape == [1, 256])
            #expect(arrays["token_embd.weight"]?.dtype == .float16)
        }
    }

    @Test func rejectsUnsupportedQuantizedTensorsWithoutDequantizing() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-reader-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = header(tensorType: .q5_0)
        data.pad(toMultiple: 32)
        data.append(contentsOf: repeatElement(UInt8(0), count: 24))

        try data.write(to: url)

        #expect(throws: GGUFReaderError.unsupportedQuantizedTensorType(.q5_0)) {
            _ = try GGUFReader(url: url).loadArrays()
        }
    }

    @Test func loadsRemainingKQuantizedTensorsIntoAffineSidecars() throws {
        let cases: [(GGUFReader.TensorType, Int, [Int], [Int])] = [
            (.q2K, 84, [1, 16], [1, 16]),
            (.q3K, 110, [1, 24], [1, 16]),
            (.q5K, 176, [1, 40], [1, 8]),
            (.q8K, 292, [1, 64], [1, 16]),
        ]

        for (tensorType, byteCount, weightShape, sidecarShape) in cases {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("gguf-reader-\(tensorType)-\(UUID().uuidString).gguf")
            defer { try? FileManager.default.removeItem(at: url) }

            var data = header(tensorType: tensorType, dimensions: [256, 1])
            data.pad(toMultiple: 32)
            data.append(contentsOf: repeatElement(UInt8(0), count: byteCount))
            try data.write(to: url)

            let arrays = try Device.withDefaultDevice(.cpu) {
                try GGUFReader(url: url).loadArrays()
            }

            #expect(arrays["token_embd.weight"]?.shape == weightShape)
            #expect(arrays["token_embd.scales"]?.shape == sidecarShape)
            #expect(arrays["token_embd.biases"]?.shape == sidecarShape)
        }
    }

    @Test func mapsQuantizedSidecarNamesThroughWeightNameMapping() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-reader-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = header(tensorType: .q2K, dimensions: [256, 1])
        data.pad(toMultiple: 32)
        data.append(contentsOf: repeatElement(UInt8(0), count: 84))
        try data.write(to: url)

        let reader = try GGUFReader(url: url)
        let mapped = try Device.withDefaultDevice(.cpu) {
            reader.mapWeightNames(try reader.loadArrays())
        }

        #expect(mapped["model.embed_tokens.weight"] != nil)
        #expect(mapped["model.embed_tokens.scales"] != nil)
        #expect(mapped["model.embed_tokens.biases"] != nil)
    }

    @Test func mapsCommonLlamaFamilyTensorNames() {
        #expect(GGUFReader.mapTensorName("token_embd.weight") == "model.embed_tokens.weight")
        #expect(GGUFReader.mapTensorName("output.weight") == "lm_head.weight")
        #expect(GGUFReader.mapTensorName("output_norm.weight") == "model.norm.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.attn_norm.weight")
                == "model.layers.3.input_layernorm.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.ffn_norm.weight")
                == "model.layers.3.post_attention_layernorm.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.attn_q.weight")
                == "model.layers.3.self_attn.q_proj.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.attn_k.weight")
                == "model.layers.3.self_attn.k_proj.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.attn_v.weight")
                == "model.layers.3.self_attn.v_proj.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.attn_output.weight")
                == "model.layers.3.self_attn.o_proj.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.ffn_gate.weight")
                == "model.layers.3.mlp.gate_proj.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.ffn_up.weight") == "model.layers.3.mlp.up_proj.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.ffn_down.weight")
                == "model.layers.3.mlp.down_proj.weight")
        #expect(GGUFReader.mapTensorName("some.other.weight") == "some.other.weight")
    }

    @Test func mapsGemma4PerLayerTensorNames() {
        #expect(
            GGUFReader.mapTensorName("per_layer_model_proj.weight", architecture: "gemma4")
                == "language_model.model.per_layer_model_projection.weight")
        #expect(
            GGUFReader.mapTensorName("per_layer_proj_norm.weight", architecture: "gemma4")
                == "language_model.model.per_layer_projection_norm.weight")
        #expect(
            GGUFReader.mapTensorName("per_layer_token_embd.weight", architecture: "gemma4")
                == "language_model.model.embed_tokens_per_layer.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.post_norm.weight", architecture: "gemma4")
                == "language_model.model.layers.3.post_per_layer_input_norm.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.inp_gate.weight", architecture: "gemma4")
                == "language_model.model.layers.3.per_layer_input_gate.weight")
        #expect(
            GGUFReader.mapTensorName("blk.3.proj.weight", architecture: "gemma4")
                == "language_model.model.layers.3.per_layer_projection.weight")
    }

    private func header(tensorType: GGUFReader.TensorType, dimensions: [UInt64] = [3, 2]) -> Data {
        var data = Data()
        data.append(contentsOf: Array("GGUF".utf8))
        data.appendLE(UInt32(3))
        data.appendLE(UInt64(1))  // tensor count
        data.appendLE(UInt64(2))  // metadata count

        data.appendString("general.architecture")
        data.appendLE(UInt32(8))  // string
        data.appendString("llama")

        data.appendString("general.alignment")
        data.appendLE(UInt32(4))  // uint32
        data.appendLE(UInt32(32))

        data.appendString("token_embd.weight")
        data.appendLE(UInt32(2))  // ndim
        for dimension in dimensions {
            data.appendLE(dimension)
        }
        data.appendLE(tensorType.rawValue)
        data.appendLE(UInt64(0))  // relative data offset
        return data
    }
}
