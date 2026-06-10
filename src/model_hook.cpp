#include "model_hook.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>

void ModelHook::record(LayerEvent ev) {
    if (!capture_enabled.load(std::memory_order_relaxed)) return;
    ev.id = counter.fetch_add(1, std::memory_order_relaxed);
    events.push(std::move(ev));
}

LayerType ModelHook::inferType(const std::string& name) {
    if (name.find("token_embd") != std::string::npos) return LayerType::Embedding;
    if (name.find("attn")       != std::string::npos) return LayerType::Attention;
    if (name.find("ffn")        != std::string::npos) return LayerType::MLP;
    if (name.find("norm")       != std::string::npos) return LayerType::LayerNorm;
    if (name.find("output")     != std::string::npos) return LayerType::Linear;
    return LayerType::Unknown;
}

std::string ModelHook::inferDevice(bool on_gpu, int gpu_id) {
    if (on_gpu) return "CUDA [GPU " + std::to_string(gpu_id) + "]";
    return "CPU (Fallback)";
}

bool ModelHook::detectAnomaly(const TensorMeta& m, std::string& out_desc) {
    if (std::isnan(m.mean) || std::isinf(m.mean)) {
        out_desc = "NaN/Inf in activation mean";
        return true;
    }
    if (m.max_val > 6.0f) {
        std::ostringstream ss;
        ss << "Outlier Feature: Max > 6.0 (got "
           << std::fixed << std::setprecision(2) << m.max_val << ")";
        out_desc = ss.str();
        return true;
    }
    return false;
}
