#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

enum class LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    Linear,
    Unknown
};

inline std::string layerTypeStr(LayerType t) {
    switch (t) {
        case LayerType::Embedding:  return "Embedding";
        case LayerType::Attention:  return "Attn (Self)";
        case LayerType::MLP:        return "MLP (SwiGLU)";
        case LayerType::LayerNorm:  return "LayerNorm";
        case LayerType::Linear:     return "Linear";
        default:                    return "Unknown";
    }
}

struct TensorMeta {
    std::vector<int64_t> shape;
    std::string          dtype;
    float sparsity = 0.0f;
    float mean     = 0.0f;
    float max_val  = 0.0f;
};

struct LayerEvent {
    uint64_t id = 0;
    std::chrono::time_point<std::chrono::system_clock> timestamp;

    std::string layer_name;
    LayerType   type   = LayerType::Unknown;
    std::string device;

    TensorMeta tensor;
    double latency_ms = 0.0;

    bool        has_anomaly  = false;
    std::string anomaly_desc;

    // populated only for attention layers
    std::vector<std::vector<float>> attn_weights;
    std::vector<std::string>        token_labels;
};

struct LayerNode {
    std::string name;
    LayerType   type           = LayerType::Unknown;
    bool        expanded       = false;
    bool        active_capture = false;
    std::vector<LayerNode> children;
};
