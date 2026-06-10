#pragma once

#include "model_hook.hpp"
#include "model_preset.hpp"
#include "layer_event.hpp"
#include <thread>
#include <atomic>
#include <vector>

class DemoRunner {
    ModelHook*         hook_;
    const ModelPreset& preset_;
    std::thread        worker_;
    std::atomic<bool>  running_{false};

public:
    DemoRunner(ModelHook* hook, const ModelPreset& preset);
    ~DemoRunner();

    void start();
    void stop();

    std::vector<LayerNode> buildTopology() const;

private:
    void loop();
};
