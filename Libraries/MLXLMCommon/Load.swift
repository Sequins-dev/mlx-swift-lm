// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first weight file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    var safetensorURLs = [URL]()
    var ggufURLs = [URL]()
    for case let url as URL in enumerator {
        switch url.pathExtension.lowercased() {
        case "safetensors":
            safetensorURLs.append(url)
        case "gguf":
            ggufURLs.append(url)
        default:
            break
        }
    }

    if !safetensorURLs.isEmpty {
        for url in safetensorURLs.sorted(by: { $0.path < $1.path }) {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    } else if let ggufURL = selectedGGUFURL(in: modelDirectory, from: ggufURLs) {
        let reader = try GGUFReader(url: ggufURL)
        weights = try reader.mapWeightNames(reader.loadArrays())
        metadata = reader.stringMetadata
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}

private func selectedGGUFURL(in modelDirectory: URL, from urls: [URL]) -> URL? {
    if let selected = selectedGGUFFilename(in: modelDirectory),
        let url = urls.first(where: { $0.lastPathComponent == selected })
    {
        return url
    }

    return urls.sorted(by: { $0.path < $1.path }).first
}

private func selectedGGUFFilename(in modelDirectory: URL) -> String? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }

    return json["gguf_file"] as? String
}

extension GGUFReader {
    fileprivate var stringMetadata: [String: String] {
        metadata.compactMapValues(\.stringValue)
    }
}
