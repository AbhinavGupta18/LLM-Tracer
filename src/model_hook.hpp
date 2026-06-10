#pragma once

#include "ring_buffer.hpp"
#include "layer_event.hpp"
#include <atomic>
#include <string>

class ModelHook {
public:
    static constexpr size_t CAPACITY = 512;

    RingBuffer<LayerEvent, CAPACITY> events;
    std::atomic<uint64_t> counter{0};
    std::atomic<bool>     capture_enabled{true};

    void record(LayerEvent ev);

    static LayerType    inferType(const std::string& tensor_name);
    static std::string  inferDevice(bool on_gpu, int gpu_id = 0);
    static bool         detectAnomaly(const TensorMeta& m, std::string& out_desc);
};
