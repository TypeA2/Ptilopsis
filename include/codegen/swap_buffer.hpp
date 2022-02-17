#pragma once

#include <memory>

#include "simd.hpp"

template <typename T>
struct buffer_deleter {
    void operator()(T* ptr) const {
        operator delete(ptr, std::align_val_t{ AVX_ALIGNMENT });
    }
};

template <typename T>
using avx_buffer = std::unique_ptr<T[], buffer_deleter<T>>;

template <typename T>
constexpr avx_buffer<T> make_buffer(size_t size) {
    return {
        static_cast<T*>(operator new[](sizeof(T) * size, std::align_val_t{ AVX_ALIGNMENT })),
        {}
    };
}

/* Utility vector to automatically swap 2 buffers */
template <typename T>
class swap_buffer {
    size_t _size;
    std::unique_ptr<T[]> vec1;
    std::unique_ptr<T[]> vec2;
    bool active = false;

    struct buffer_deleter {
        void operator()(T* ptr) const {
            operator delete(ptr, std::align_val_t{ AVX_ALIGNMENT });
        }
    };

    public:
    swap_buffer() = delete;
    explicit swap_buffer(size_t size)
        : _size{ size }
        /* Align to 64 bytes, for theoretical AVX-512 support */
        , vec1{ static_cast<T*>(operator new[](sizeof(T)* _size, std::align_val_t{ AVX_ALIGNMENT })), {} }
        , vec2{ static_cast<T*>(operator new[](sizeof(T)* _size, std::align_val_t{ AVX_ALIGNMENT })), {} } { }

    [[nodiscard]] size_t size() const {
        return _size;
    }

    void swap() {
        active = !active;
    }

    [[nodiscard]] auto& cur() {
        return active ? vec2 : vec1;
    }

    [[nodiscard]] const auto& cur() const {
        return active ? vec2 : vec1;
    }

    std::unique_ptr<T[]>* operator->() {
        return &cur();
    }

    [[nodiscard]] auto& operator[](size_t i) {
        return cur()[i];
    }

    [[nodiscard]] const auto& operator[](size_t i) const {
        return cur()[i];
    }

    [[nodiscard]] std::span<T> in() {
        return { cur().get(), _size };
    }

    [[nodiscard]] auto& in(size_t i) {
        return in()[i];
    }

    [[nodiscard]] const auto& in(size_t i) const {
        return in()[i];
    }

    [[nodiscard]] std::span<T> out() {
        return { ((&cur() == &vec1) ? vec2 : vec1).get(), _size };
    }

    [[nodiscard]] auto& out(size_t i) {
        return out()[i];
    }

    [[nodiscard]] const auto& out(size_t i) const {
        return out()[i];
    }
};
