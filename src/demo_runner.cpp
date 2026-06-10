#include "demo_runner.hpp"
#include <random>
#include <chrono>
#include <cmath>
#include <thread>

static const std::vector<std::string> DEMO_TOKENS = {
    "Google", "Developer", "Student", "Club"
};

DemoRunner::DemoRunner(ModelHook* hook, const ModelPreset& preset)
    : hook_(hook), preset_(preset) {}

DemoRunner::~DemoRunner() {
    stop();
}

void DemoRunner::start() {
    running_.store(true);
    worker_ = std::thread(&DemoRunner::loop, this);
}

void DemoRunner::stop() {
    running_.store(false);
    if (worker_.joinable()) worker_.join();
}

static std::vector<std::vector<float>> makeAttentionMatrix(std::mt19937& rng) {
    int seq = (int)DEMO_TOKENS.size();
    std::vector<std::vector<float>> W(seq, std::vector<float>(seq, 0.0f));
    std::uniform_real_distribution<float> ud(0.1f, 1.0f);

    for (int r = 0; r < seq; r++) {
        float sum = 0.0f;
        for (int c = 0; c <= r; c++) {
            float raw = (c == r) ? 1.5f + ud(rng) : ud(rng);
            W[r][c] = std::exp(raw);
            sum += W[r][c];
        }
        for (int c = 0; c <= r; c++) {
            W[r][c] /= sum;
        }
    }
    return W;
}

void DemoRunner::loop() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> latency_d(0.8f, 3.5f);
    std::uniform_real_distribution<float> sparsity_d(0.2f, 0.75f);
    std::uniform_real_distribution<float> mean_d(-0.8f, 0.8f);
    std::uniform_real_distribution<float> max_d(0.4f, 5.8f);
    std::bernoulli_distribution            anomaly_d(0.07);

    struct LayerSpec { LayerType type; std::string suffix; };
    static const LayerSpec block_layers[] = {
        {LayerType::Attention, ".attn"},
        {LayerType::MLP,       ".mlp"},
    };

    int n_layers = preset_.n_layers;
    int cpu_threshold = n_layers - 2;

    while (running_.load()) {
        for (int l = 0; l < n_layers && running_.load(); l++) {
            for (auto& spec : block_layers) {
                if (!running_.load()) break;

                LayerEvent ev;
                ev.timestamp  = std::chrono::system_clock::now();
                ev.type       = spec.type;
                ev.layer_name = "layers." + std::to_string(l) + spec.suffix;

                bool on_gpu = (l < cpu_threshold);
                ev.device = ModelHook::inferDevice(on_gpu, 0);

                ev.tensor.shape    = {1, (int64_t)preset_.n_heads,
                                      (int64_t)preset_.hidden_dim};
                ev.tensor.dtype    = preset_.dtype;
                ev.tensor.sparsity = sparsity_d(rng);
                ev.tensor.mean     = mean_d(rng);
                ev.tensor.max_val  = max_d(rng);
                ev.latency_ms      = latency_d(rng);

                if (!on_gpu) {
                    ev.has_anomaly  = true;
                    ev.anomaly_desc = "CUDA OOM Fallback: Processing " +
                                      ev.layer_name + " on CPU Host Memory.";
                }

                if (!ev.has_anomaly && anomaly_d(rng)) {
                    std::uniform_real_distribution<float> bump(6.1f, 8.5f);
                    ev.tensor.max_val = bump(rng);
                    std::string desc;
                    ModelHook::detectAnomaly(ev.tensor, desc);
                    ev.has_anomaly  = true;
                    ev.anomaly_desc = "Layer " + std::to_string(l) + " " + desc;
                }

                if (spec.type == LayerType::Attention) {
                    ev.attn_weights = makeAttentionMatrix(rng);
                    ev.token_labels = DEMO_TOKENS;
                }

                hook_->record(std::move(ev));
                std::this_thread::sleep_for(std::chrono::milliseconds(130));
            }
        }

        if (running_.load()) {
            LayerEvent norm_ev;
            norm_ev.timestamp       = std::chrono::system_clock::now();
            norm_ev.type            = LayerType::LayerNorm;
            norm_ev.layer_name      = "output_norm";
            norm_ev.device          = ModelHook::inferDevice(false, 0);
            norm_ev.tensor.shape    = {1, (int64_t)preset_.hidden_dim};
            norm_ev.tensor.dtype    = preset_.dtype;
            norm_ev.tensor.sparsity = 0.08f;
            norm_ev.tensor.mean     = 0.01f;
            norm_ev.tensor.max_val  = 1.4f;
            norm_ev.latency_ms      = 0.28;
            norm_ev.has_anomaly     = true;
            norm_ev.anomaly_desc    = "CUDA OOM Fallback: Processing LayerNorm on CPU Host Memory.";
            hook_->record(std::move(norm_ev));

            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
    }
}

std::vector<LayerNode> DemoRunner::buildTopology() const {
    LayerNode root;
    root.name     = preset_.name;
    root.type     = LayerType::Unknown;
    root.expanded = true;

    LayerNode emb;
    emb.name = "embed_tokens";
    emb.type = LayerType::Embedding;
    root.children.push_back(emb);

    LayerNode lg;
    lg.name     = "layers";
    lg.type     = LayerType::Unknown;
    lg.expanded = true;

    for (int i = 0; i < preset_.n_layers; i++) {
        LayerNode blk;
        blk.name           = "layers." + std::to_string(i);
        blk.type           = LayerType::Unknown;
        blk.expanded       = (i == 1);
        blk.active_capture = (i == 1);

        LayerNode attn;
        attn.name           = "layers." + std::to_string(i) + ".attn";
        attn.type           = LayerType::Attention;
        attn.active_capture = (i == 1);

        LayerNode mlp;
        mlp.name           = "layers." + std::to_string(i) + ".mlp";
        mlp.type           = LayerType::MLP;
        mlp.active_capture = (i == 1);

        blk.children = {attn, mlp};
        lg.children.push_back(blk);
    }

    root.children.push_back(lg);
    return {root};
}
