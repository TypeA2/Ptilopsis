#pragma once

#include <ostream>
#include <concepts>

#include <immintrin.h>

#include <magic_enum.hpp>

#ifdef _MSC_VER
#   define FORCE_INLINE __forceinline
#else
#   define FORCE_INLINE inline __attribute__((__always_inline__))
#endif

/* Align for potential AVX-512 usage */
#define AVX_ALIGNMENT 64
#define AVX_ALIGNED alignas(64)

/* Pretty-printing */
template <size_t bits>
concept integer_size = (bits == 64) || (bits == 32) || (bits == 16) || (bits == 8);

template <size_t bits, bool is_signed> requires integer_size<bits>
struct avx_formatter {
    __m256i reg;
};

template <size_t bits, bool is_signed> requires integer_size<bits>
std::ostream& operator<<(std::ostream& os, const avx_formatter<bits, is_signed>& reg) {
    os << '[';


    switch (bits) {
        case 64:
            if constexpr (is_signed) {
                os << _mm256_extract_epi64(reg.reg, 0) << ", "
                    << _mm256_extract_epi64(reg.reg, 1) << ", "
                    << _mm256_extract_epi64(reg.reg, 2) << ", "
                    << _mm256_extract_epi64(reg.reg, 3);
            } else {
                os << static_cast<uint64_t>(_mm256_extract_epi64(reg.reg, 0)) << ", "
                    << static_cast<uint64_t>(_mm256_extract_epi64(reg.reg, 1)) << ", "
                    << static_cast<uint64_t>(_mm256_extract_epi64(reg.reg, 2)) << ", "
                    << static_cast<uint64_t>(_mm256_extract_epi64(reg.reg, 3));
            }
            break;

        case 32:
            if constexpr (is_signed) {
                os << _mm256_extract_epi32(reg.reg, 0) << ", "
                    << _mm256_extract_epi32(reg.reg, 1) << ", "
                    << _mm256_extract_epi32(reg.reg, 2) << ", "
                    << _mm256_extract_epi32(reg.reg, 3) << ", "
                    << _mm256_extract_epi32(reg.reg, 4) << ", "
                    << _mm256_extract_epi32(reg.reg, 5) << ", "
                    << _mm256_extract_epi32(reg.reg, 6) << ", "
                    << _mm256_extract_epi32(reg.reg, 7);
            } else {
                os << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 0)) << ", "
                    << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 1)) << ", "
                    << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 2)) << ", "
                    << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 3)) << ", "
                    << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 4)) << ", "
                    << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 5)) << ", "
                    << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 6)) << ", "
                    << static_cast<uint32_t>(_mm256_extract_epi32(reg.reg, 7));
            }
            break;

        case 16:
            if constexpr (is_signed) {
                os << _mm256_extract_epi16(reg.reg, 0) << ", "
                    << _mm256_extract_epi16(reg.reg, 1) << ", "
                    << _mm256_extract_epi16(reg.reg, 2) << ", "
                    << _mm256_extract_epi16(reg.reg, 3) << ", "
                    << _mm256_extract_epi16(reg.reg, 4) << ", "
                    << _mm256_extract_epi16(reg.reg, 5) << ", "
                    << _mm256_extract_epi16(reg.reg, 6) << ", "
                    << _mm256_extract_epi16(reg.reg, 7) << ", "
                    << _mm256_extract_epi16(reg.reg, 8) << ", "
                    << _mm256_extract_epi16(reg.reg, 9) << ", "
                    << _mm256_extract_epi16(reg.reg, 10) << ", "
                    << _mm256_extract_epi16(reg.reg, 11) << ", "
                    << _mm256_extract_epi16(reg.reg, 12) << ", "
                    << _mm256_extract_epi16(reg.reg, 13) << ", "
                    << _mm256_extract_epi16(reg.reg, 14) << ", "
                    << _mm256_extract_epi16(reg.reg, 15);
            } else {
                os << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 0)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 1)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 2)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 3)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 4)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 5)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 6)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 7)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 8)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 9)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 10)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 11)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 12)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 13)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 14)) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi16(reg.reg, 15));
            }
            break;

        case 8:
            if constexpr (is_signed) {
                os << _mm256_extract_epi8(reg.reg, 0) << ", "
                    << _mm256_extract_epi8(reg.reg, 1) << ", "
                    << _mm256_extract_epi8(reg.reg, 2) << ", "
                    << _mm256_extract_epi8(reg.reg, 3) << ", "
                    << _mm256_extract_epi8(reg.reg, 4) << ", "
                    << _mm256_extract_epi8(reg.reg, 5) << ", "
                    << _mm256_extract_epi8(reg.reg, 6) << ", "
                    << _mm256_extract_epi8(reg.reg, 7) << ", "
                    << _mm256_extract_epi8(reg.reg, 8) << ", "
                    << _mm256_extract_epi8(reg.reg, 9) << ", "
                    << _mm256_extract_epi8(reg.reg, 10) << ", "
                    << _mm256_extract_epi8(reg.reg, 11) << ", "
                    << _mm256_extract_epi8(reg.reg, 12) << ", "
                    << _mm256_extract_epi8(reg.reg, 13) << ", "
                    << _mm256_extract_epi8(reg.reg, 14) << ", "
                    << _mm256_extract_epi8(reg.reg, 15) << ", "
                    << _mm256_extract_epi8(reg.reg, 16) << ", "
                    << _mm256_extract_epi8(reg.reg, 17) << ", "
                    << _mm256_extract_epi8(reg.reg, 18) << ", "
                    << _mm256_extract_epi8(reg.reg, 19) << ", "
                    << _mm256_extract_epi8(reg.reg, 20) << ", "
                    << _mm256_extract_epi8(reg.reg, 21) << ", "
                    << _mm256_extract_epi8(reg.reg, 22) << ", "
                    << _mm256_extract_epi8(reg.reg, 23) << ", "
                    << _mm256_extract_epi8(reg.reg, 24) << ", "
                    << _mm256_extract_epi8(reg.reg, 25) << ", "
                    << _mm256_extract_epi8(reg.reg, 26) << ", "
                    << _mm256_extract_epi8(reg.reg, 27) << ", "
                    << _mm256_extract_epi8(reg.reg, 28) << ", "
                    << _mm256_extract_epi8(reg.reg, 29) << ", "
                    << _mm256_extract_epi8(reg.reg, 30) << ", "
                    << _mm256_extract_epi8(reg.reg, 31);
            } else {
                os << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 0) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 1) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 2) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 3) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 4) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 5) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 6) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 7) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 8) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 9) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 10) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 11) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 12) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 13) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 14) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 15) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 16) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 17) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 18) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 19) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 20) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 21) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 22) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 23) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 24) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 25) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 26) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 27) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 28) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 29) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 30) & 0xFF) << ", "
                    << static_cast<uint16_t>(_mm256_extract_epi8(reg.reg, 31) & 0xFF);
            }
            break;
    }

    os << ']';
    return os;
}

template <size_t bits = 32, bool is_signed = false>
std::ostream& print(std::ostream& os, __m256i reg) {
    return (os << avx_formatter<bits, is_signed>{ .reg = reg });
}

template <size_t bits = 32, bool is_signed = false>
auto format(__m256i reg) {
    return avx_formatter<bits, is_signed>{.reg = reg };
}

inline std::ostream& operator<<(std::ostream& os, __m256i reg) {
    return (os << format(reg));
}

/* _mm256_slli_si256 operates on 2 128-bit lanes. This emulates a full-width
 * 256-bit left-shift, as per: https://stackoverflow.com/a/25264853/8662472
 */
template <uint8_t bytes>
FORCE_INLINE __m256i _mm256_slli_si256_dual(__m256i x) {  // NOLINT(clang-diagnostic-reserved-identifier, bugprone-reserved-identifier)
    /* Left shift by 128 bits:
     * dest[127:0] := 0
     * dest[255:128] := src1[127:0]
     */
    __m256i shuffled = _mm256_permute2x128_si256(x, x, 0b0'000'1'000);

    if constexpr (bytes < 16) {
        /* Emulate shift
         * dest[127:0] := ((src1[127:0] << 128) | src2[127:0]) >> (imm8 * 8)
         * dest[255:128] := ((src1[255:128] << 128) | src2[255:128]) >> (imm8 * 8)
         * 
         * Where:
         *   src1 = x
         *   src2[127:0] = 0
         *   src2[255:128] = src1[127:0]
         * So effectively:
         *   dest[127:0] = (src1[127:0] << 128) >> (imm8 * 8)
         *   dest[255:127] = src1 >> (imm8 * 8)
         * 
         * TL;DR:
         *   Shift lower 128 bits as usual, shift in zeroes,
         *   For the upper 128 bits, reconstruct the original input from 2
         *     parts and shift as usual.
         */
        return _mm256_alignr_epi8(x, shuffled, 16 - bytes);
    } else if constexpr (bytes == 16) {
        /* This already represents a 128-bit shift */
        return shuffled;
    } else {
        /* Shift >16 bytes, so lower 16 bytes are definitely 0.
         * This effectively only shifts the upper bytes, but that's what we want.
         */
        return _mm256_slli_si256(shuffled, bytes - 16);
    }
}

/* Same situation as with _mm256_slli_si256, this emulates a right-shift on 1 256-bit lane. */
template <uint8_t bytes>
FORCE_INLINE __m256i _mm256_srli_si256_dual(__m256i x) {  // NOLINT(clang-diagnostic-reserved-identifier, bugprone-reserved-identifier)
    /* Right shift by 128 bits:
     * dest[127:0] := src1[255:128]
     * dest[255:128] := 0
     */
    __m256i shuffled = _mm256_permute2x128_si256(x, x, 0b1'000'0'001);

    if constexpr (bytes < 16) {
        /* Emulate shift
         * dest[127:0] := ((src1[127:0] << 128) | src2[127:0]) >> (imm8 * 8)
         * dest[255:128] := ((src1[255:128] << 128) | src2[255:128]) >> (imm8 * 8)
         * 
         * Where:
         *   src1[127:0] = x[255:128]
         *   src1[255:128] = 0
         *   src2 = x
         * Effectively:
         *   dest[127:0] = ((x[255:128] << 128) | x[127:0]) >> (imm8 * 8)
         *   dest[255:128] = (x[255:128] << 128) >> (imm8 * 8)
         * 
         * TL;DR:
         *   Shift the upper 128 bits as usual, shift in zeroes,
         *   Reconstruct entire number and shift for the lower 128 bits, shifting
         *     in parts of the original number
         */
        return _mm256_alignr_epi8(shuffled, x, bytes);
    } else if constexpr (bytes == 16) {
        /* Already shifted exactly 128 bits */
        return shuffled;
    } else {
        /* We already right shifted by 128 bits, so shift the lower lane the remaining number of bytes */
        return _mm256_srli_si256(shuffled, bytes - 16);
    }
}

namespace simd::epi32_operators {
    FORCE_INLINE __m256i operator ""_m256i(unsigned long long v) {
        return _mm256_set1_epi32(static_cast<int>(v & 0xFFFFFFFF));
    }

#ifdef _MSC_VER
    FORCE_INLINE __m256i operator>(__m256i lhs, __m256i rhs) {
        return _mm256_cmpgt_epi32(lhs, rhs);
    }

    FORCE_INLINE __m256i operator==(__m256i lhs, __m256i rhs) {
        return _mm256_cmpeq_epi32(lhs, rhs);
    }

    FORCE_INLINE __m256i operator&(__m256i lhs, __m256i rhs) {
        return _mm256_and_si256(lhs, rhs);
    }

    FORCE_INLINE __m256i operator|(__m256i lhs, __m256i rhs) {
        return _mm256_or_si256(lhs, rhs);
    }

    FORCE_INLINE __m256i operator^(__m256i lhs, __m256i rhs) {
        return _mm256_xor_si256(lhs, rhs);
    }

    FORCE_INLINE __m256i operator+(__m256i lhs, __m256i rhs) {
        return _mm256_add_epi32(lhs, rhs);
    }

    FORCE_INLINE __m256i operator-(__m256i lhs, __m256i rhs) {
        return _mm256_sub_epi32(lhs, rhs);
    }

    FORCE_INLINE __m256i operator*(__m256i lhs, __m256i rhs) {
        return _mm256_mullo_epi32(lhs, rhs);
    }

    FORCE_INLINE __m256i operator~(__m256i lhs) {
        return _mm256_xor_si256(lhs, 0xFFFFFFFF_m256i);
    }

    FORCE_INLINE __m256i operator>(__m256i lhs, int rhs) {
        return _mm256_cmpgt_epi32(lhs, _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator==(__m256i lhs, int rhs) {
        return _mm256_cmpeq_epi32(lhs, _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator&(__m256i lhs, int rhs) {
        return _mm256_and_si256(lhs, _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator|(__m256i lhs, int rhs) {
        return _mm256_or_si256(lhs, _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator^(__m256i lhs, int rhs) {
        return _mm256_xor_si256(lhs, _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator+(__m256i lhs, int rhs) {
        return _mm256_add_epi32(lhs, _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator-(__m256i lhs, int rhs) {
        return _mm256_sub_epi32(lhs, _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator*(__m256i lhs, int rhs) {
        return _mm256_mullo_epi32(lhs, _mm256_set1_epi32(rhs));
    }
#endif
}

namespace simd::epi32 {
    template <typename T> requires std::is_enum_v<T>
    FORCE_INLINE __m256i from_enum(T val) {
        return _mm256_set1_epi32(magic_enum::enum_integer(val));
    }

    FORCE_INLINE __m256i from_value(int v) {
        return _mm256_set1_epi32(v);
    }

    FORCE_INLINE __m256i from_values(int a, int b, int c, int d, int e, int f, int g, int h) {
        return _mm256_set_epi32(h, g, f, e, d, c, b, a);
    }

    FORCE_INLINE bool is_zero(__m256i v) {
        return _mm256_testz_si256(v, v);
    }

    template <typename T>
    FORCE_INLINE __m256i load(const T* ptr) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    template <typename T>
    FORCE_INLINE __m256i loadu(const T* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    template <typename T>
    FORCE_INLINE __m256i maskload(const T* ptr, __m256i mask) {
        return _mm256_maskload_epi32(reinterpret_cast<const int*>(ptr), mask);
    }

    template <typename T>
    FORCE_INLINE void store(T* ptr, __m256i a) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), a);
    }

    template <typename T>
    FORCE_INLINE void storeu(T* ptr, __m256i a) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), a);
    }

    template <typename T>
    FORCE_INLINE void maskstore(T* ptr, __m256i mask, __m256i src) {
        _mm256_maskstore_epi32(reinterpret_cast<int*>(ptr), mask, src);
    }

    FORCE_INLINE __m256i max(__m256i a, __m256i b) {
        return _mm256_max_epi32(a, b);
    }

    template <typename T>
    FORCE_INLINE __m256i gather(const T* base, __m256i vindex) {
        return _mm256_i32gather_epi32(reinterpret_cast<const int*>(base), vindex, 4);
    }

    template <typename T>
    FORCE_INLINE __m256i maskgather(__m256i src, const T* base, __m256i vindex, __m256i mask) {
        return _mm256_mask_i32gather_epi32(src, reinterpret_cast<const int*>(base), vindex, mask, 4);
    }

    template <typename T>
    FORCE_INLINE __m256i maskgatherz(const T* base, __m256i vindex, __m256i mask) {
        return _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), reinterpret_cast<const int*>(base), vindex, mask, 4);
    }

    FORCE_INLINE __m256i blendv(__m256i a, __m256i b, __m256i mask) {
        return _mm256_blendv_epi8(a, b, mask);
    }

    FORCE_INLINE __m256i zero() {
        return _mm256_setzero_si256();
    }

    template <typename T = int> requires (sizeof(T) == 4)
    FORCE_INLINE [[nodiscard]] std::array<T, 8> extract(__m256i a) {
        AVX_ALIGNED std::array<T, 8> res;
        epi32::store(res.data(), a);
        return res;
    }
}
