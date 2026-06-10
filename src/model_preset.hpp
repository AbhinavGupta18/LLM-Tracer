#pragma once

#include <string>
#include <vector>

struct ModelPreset {
    std::string name;
    std::string display_name;
    int n_layers;
    int n_heads;
    int hidden_dim;
    std::string dtype;
    std::string notes;
};

inline const std::vector<ModelPreset> PRESETS = {
    {"mistral-7b",      "Mistral 7B",        32, 32, 4096, "float16",  "Strong general model, same dims as LLaMA 8B"},
    {"llama-3.1-8b",    "LLaMA 3.1 8B",      32, 32, 4096, "float16",  "Meta's base model, widely used baseline"},
    {"llama-3.1-70b",   "LLaMA 3.1 70B",     80, 64, 8192, "float16",  "Large variant, needs high VRAM"},
    {"qwen-2.5-7b",     "Qwen 2.5 7B",       28, 28, 3584, "float16",  "Alibaba, strong on code and math"},
    {"qwen-2.5-72b",    "Qwen 2.5 72B",      80, 64, 8192, "float16",  "Qwen large, competitive with LLaMA 70B"},
    {"gemma-2-9b",      "Gemma 2 9B",        42, 16, 3584, "bfloat16", "Google, unique interleaved attention"},
    {"phi-3-mini",      "Phi-3 Mini 3.8B",   32, 32, 3072, "float16",  "Microsoft, punches above its weight"},
    {"phi-3-medium",    "Phi-3 Medium 14B",  40, 40, 5120, "float16",  "Stronger Phi variant, good on CPU"},
};
