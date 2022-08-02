#pragma once

#include <span>
#include <memory>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <algorithm>

#include "simd.hpp"

/* template <std::ranges::range R> should work... */
template <std::ranges::range R> 
using range_element_type_t = std::remove_cvref_t<decltype(*std::ranges::begin(R {}))>;

template <std::ranges::range R>
auto range_to_vec(R&& range) {
    std::vector<uint32_t> vec;
    std::ranges::copy(std::forward<R>(range), std::back_inserter(vec));

    return vec;
}

template <typename T>
class AVX_ALIGNED avx_buffer {
    public:
    using element_type = T;
    using value_type = std::remove_cvref_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;

    private:
    size_type _count{};
    T* _ptr{};

    void _resize(size_type new_count) {
        clear();
        _count = new_count;

        /* Round up to power of 2: https://stackoverflow.com/a/9194117/8662472 */
        size_t new_capacity = ((sizeof(T) * _count) + AVX_ALIGNMENT - 1) & -AVX_ALIGNMENT;

        /* Over-allocate by 1x the AVX alignment at both the start and the end */
        char* raw_ptr = static_cast<char*>(operator new[](new_capacity + (2 * AVX_ALIGNMENT), std::align_val_t { AVX_ALIGNMENT }));

        /* Zero-initialize everything so everything can be read */
        std::fill_n(raw_ptr, new_capacity + (2 * AVX_ALIGNMENT), 0);

        /* Keep a buffer the size of the AVX alignemtn before our start */
        _ptr = reinterpret_cast<T*>(raw_ptr + AVX_ALIGNMENT);
    }

    public:

    constexpr ~avx_buffer() { clear(); }

    constexpr avx_buffer() : _ptr { nullptr } { }
    constexpr explicit avx_buffer(size_t count) { _resize(count); }

    constexpr avx_buffer(const avx_buffer& other) : avx_buffer { other._count } {
        std::ranges::copy(other, _ptr);
    }

    constexpr avx_buffer& operator=(const avx_buffer& other) {
        _resize(other._count);
        std::ranges::copy(other, _ptr);

        return *this;
    }

    constexpr avx_buffer(avx_buffer&& other) noexcept
        : _count { std::exchange(other._count, 0) }
        , _ptr { std::exchange(other._ptr, nullptr) } {
    }

    constexpr avx_buffer& operator=(avx_buffer&& other) noexcept {
        clear();

        _count = std::exchange(other._count, 0);
        _ptr = std::exchange(other._ptr, nullptr);

        return *this;
    }

    template <std::ranges::sized_range R>
    constexpr explicit avx_buffer(R&& range) : avx_buffer { std::ranges::size(range) } {  // NOLINT(bugprone-forwarding-reference-overload)
        std::ranges::copy(range, _ptr);
    }

    template <std::ranges::sized_range R>
    constexpr avx_buffer& operator=(R&& range) {
        _resize(std::ranges::size(range));
        std::ranges::copy(range, _ptr);

        return *this;
    }

    template <std::ranges::range R> requires (!std::ranges::sized_range<R>)
    constexpr explicit avx_buffer(R&& range) {  // NOLINT(bugprone-forwarding-reference-overload)
        auto vec = range_to_vec(range);
        _resize(vec.size());
        std::ranges::move(vec, _ptr);
    }

    template <std::ranges::range R> requires (!std::ranges::sized_range<R>)
    constexpr avx_buffer& operator=(R&& range) {
        auto vec = range_to_vec(range);
        _resize(vec.size());
        std::ranges::move(vec, _ptr);

        return *this;
    }

    constexpr explicit avx_buffer(std::span<T> elements) : avx_buffer { elements.size() } {
        std::ranges::copy(elements, _ptr);
    }

    constexpr avx_buffer& operator=(std::span<T> elements) {
        _resize(elements.size());
        std::ranges::copy(elements, _ptr);

        return *this;
    }

    [[nodiscard]] static constexpr avx_buffer zero(size_type count) {
        return avx_buffer { count };
    }

    [[nodiscard]] static constexpr avx_buffer iota(size_type bound) {
        return avx_buffer { std::views::iota(T{ 0 }, bound) };
    }

    [[nodiscard]] static constexpr avx_buffer fill(size_type count, T val) {
        avx_buffer buf { count };

        /* Zero-initialized is the default*/
        if (val) {
            std::ranges::fill(buf, val);
        }

        return buf;
    }

    [[nodiscard]] constexpr size_type size() const {
        return _count;
    }

    [[nodiscard]] constexpr T* data() {
        return _ptr;
    }

    [[nodiscard]] constexpr const T* data() const {
        return _ptr;
    }

    [[nodiscard]] constexpr __m256i* m256i() {
        return reinterpret_cast<__m256i*>(_ptr);
    }

    [[nodiscard]] constexpr const __m256i* m256i() const {
        return reinterpret_cast<const __m256i*>(_ptr);
    }

    [[nodiscard]] constexpr __m256i* m256i(size_type i) {
        return m256i() + i;
    }

    [[nodiscard]] constexpr const __m256i* m256i(size_type i) const {
        return m256i() + i;
    }

    [[nodiscard]] constexpr size_t size_m256i() const {
        /* How many 32-byte __m256i elements this contains*/
        return (((sizeof(T) * _count) + sizeof(__m256i) - 1) & -static_cast<int64_t>(sizeof(__m256i))) / sizeof(__m256i);
    }

    [[nodiscard]] constexpr T& operator[](size_type i) {
        return _ptr[i];
    }

    [[nodiscard]] constexpr const T& operator[](size_type i) const {
        return _ptr[i];
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return _ptr;
    }

    [[nodiscard]] constexpr T* begin() {
        return _ptr;
    }

    [[nodiscard]] constexpr const T* begin() const {
        return _ptr;
    }

    [[nodiscard]] constexpr const T* cbegin() const {
        return _ptr;
    }

    [[nodiscard]] constexpr T* end() {
        return _ptr + _count;
    }

    [[nodiscard]] constexpr const T* end() const {
        return _ptr + _count;
    }

    [[nodiscard]] constexpr const T* cend() const {
        return _ptr + _count;
    }

    // ReSharper disable once CppNonExplicitConversionOperator
    [[nodiscard]] constexpr operator std::span<T>() const {
        return { _ptr, _count };
    }

    [[nodiscard]] constexpr T& front() {
        return *_ptr;
    }

    [[nodiscard]] constexpr const T& front() const {
        return *_ptr;
    }

    [[nodiscard]] constexpr T& back() {
        return _ptr[_count - 1];
    }

    [[nodiscard]] constexpr const T& back() const {
        return _ptr[_count - 1];
    }

    [[nodiscard]] constexpr std::span<T> slice(size_type start, size_type end) const {
        return { _ptr + start, _ptr + end };
    }

    template <typename U>
        requires std::convertible_to<T, U> || (
            std::is_enum_v<T> && std::integral<U> && (sizeof(U) >= sizeof(std::underlying_type_t<T>)))
    [[nodiscard]] constexpr avx_buffer<U> cast() const {
        avx_buffer<U> res { _count };
        std::ranges::transform(*this, res.begin(), [](T v) { return static_cast<U>(v);  });

        return res;
    }

    constexpr void shrink_to(size_type new_size) {
        _count = std::min(new_size, _count);
    }

    constexpr void clear() {
        if (_ptr) {
            operator delete[](reinterpret_cast<char*>(_ptr) - AVX_ALIGNMENT, std::align_val_t { AVX_ALIGNMENT });
            _ptr = nullptr;
        }

        _count = 0;
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const avx_buffer<T>& buf) {
    os << '[';

    for (typename avx_buffer<T>::size_type i = 0; i < (buf.size() - 1); ++i) {
        os << buf[i] << ", ";
    }

    os << buf.back() << ']';

    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::span<T> buf) {
    os << '[';

    for (size_t i = 0; i < (buf.size() - 1); ++i) {
        os << buf[i] << ", ";
    }

    os << buf.back() << ']';

    return os;
}

template <typename T>
bool operator==(const avx_buffer<T>& lhs, const avx_buffer<T>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }

    return true;
}
