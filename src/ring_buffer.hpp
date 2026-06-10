#pragma once

#include <array>
#include <mutex>
#include <vector>

template<typename T, size_t N>
class RingBuffer {
    std::array<T, N> buf_;
    size_t write_pos_ = 0;
    size_t count_     = 0;
    mutable std::mutex mu_;

public:
    void push(T item) {
        std::lock_guard<std::mutex> lk(mu_);
        buf_[write_pos_] = std::move(item);
        write_pos_ = (write_pos_ + 1) % N;
        if (count_ < N) count_++;
    }

    // returns a snapshot of all current items in insertion order
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<T> out;
        out.reserve(count_);
        size_t start = (count_ < N) ? 0 : write_pos_;
        for (size_t i = 0; i < count_; i++) {
            out.push_back(buf_[(start + i) % N]);
        }
        return out;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return count_;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        write_pos_ = 0;
        count_     = 0;
    }
};
