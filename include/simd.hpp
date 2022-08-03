#pragma once

#include <ostream>
#include <concepts>

#include <immintrin.h>

#include <magic_enum.hpp>

/* Align for potential AVX-512 usage */
#define AVX_ALIGNMENT 64
#define AVX_ALIGNED alignas(64)

#ifdef _MSC_VER
#   define FORCE_INLINE __forceinline
#else
#   define FORCE_INLINE inline __attribute__((__always_inline__))
#endif

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

    FORCE_INLINE __m256i operator~(__m256i lhs) {
        return _mm256_xor_si256(lhs, 0xFFFFFFFF_m256i);
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


    FORCE_INLINE __m256i operator>(__m256i lhs, __m256i rhs) {
        return _mm256_cmpgt_epi32(lhs, rhs);
    }

    FORCE_INLINE __m256i operator==(__m256i lhs, __m256i rhs) {
        return _mm256_cmpeq_epi32(lhs, rhs);
    }

    FORCE_INLINE __m256i operator!=(__m256i lhs, __m256i rhs) {
        return ~(lhs == rhs);
    }

    FORCE_INLINE __m256i operator<(__m256i lhs, __m256i rhs) {
        return ~((lhs == rhs) | (lhs > rhs));
    }

    FORCE_INLINE __m256i operator<=(__m256i lhs, __m256i rhs) {
        return ~(lhs > rhs);
    }

    FORCE_INLINE __m256i operator>=(__m256i lhs, __m256i rhs) {
        return (lhs == rhs) | (lhs > rhs);
    }


    /* Integer converting operators */
    FORCE_INLINE __m256i operator&(__m256i lhs, int rhs) {
        return lhs & _mm256_set1_epi32(rhs);
    }

    FORCE_INLINE __m256i operator|(__m256i lhs, int rhs) {
        return lhs | _mm256_set1_epi32(rhs);
    }

    FORCE_INLINE __m256i operator^(__m256i lhs, int rhs) {
        return lhs ^ _mm256_set1_epi32(rhs);
    }


    FORCE_INLINE __m256i operator+(__m256i lhs, int rhs) {
        return lhs + _mm256_set1_epi32(rhs);
    }

    FORCE_INLINE __m256i operator-(__m256i lhs, int rhs) {
        return lhs - _mm256_set1_epi32(rhs);
    }

    FORCE_INLINE __m256i operator*(__m256i lhs, int rhs) {
        return lhs * _mm256_set1_epi32(rhs);
    }

    
    FORCE_INLINE __m256i operator>(__m256i lhs, int rhs) {
        return lhs == _mm256_set1_epi32(rhs);
    }

    FORCE_INLINE __m256i operator==(__m256i lhs, int rhs) {
        return lhs == _mm256_set1_epi32(rhs);
    }

    FORCE_INLINE __m256i operator!=(__m256i lhs, int rhs) {
        return ~(lhs == rhs);
    }

    FORCE_INLINE __m256i operator<(__m256i lhs, int rhs) {
        const __m256i val = _mm256_set1_epi32(rhs);
        return ~((lhs == val) | (lhs > val));
    }

    FORCE_INLINE __m256i operator<=(__m256i lhs, int rhs) {
        return ~(lhs > _mm256_set1_epi32(rhs));
    }

    FORCE_INLINE __m256i operator>=(__m256i lhs, int rhs) {
        const __m256i val = _mm256_set1_epi32(rhs);
        return (lhs == val) | (lhs > val);
    }
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

    FORCE_INLINE __m256i hi_to_lo(__m256i a) {
        return _mm256_permute2x128_si256(a, a, 0b1000'0001);
    }

    FORCE_INLINE __m256i pack64_32(__m256i a, __m256i b) {
        /* Pack 2 4x64-bit masks into 1 8x32-bit mask
         *   https://stackoverflow.com/a/69408295/8662472
         *
         * First shuffle so we get:
         *   0 -> lo[0]
         *   1 -> lo[1]
         *   2 -> hi[0]
         *   3 -> hi[1]
         *   4 -> lo[2]
         *   5 -> lo[3]
         *   6 -> hi[2]
         *   7 -> hi[3]
         *
         * Then shuffle in 64-bit pairs to get the correct result mask
         */
        const __m256 res = _mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), 0b10'00'10'00);

        return _mm256_permute4x64_epi64(_mm256_castps_si256(res), 0b11'01'10'00);
    }

    FORCE_INLINE __m256i expand32_64_lo(__m256i a) {
        return _mm256_cvtepi32_epi64(_mm256_castsi256_si128(a));
    }

    FORCE_INLINE __m256i expand32_64_hi(__m256i a) {
        return _mm256_cvtepi32_epi64(_mm256_castsi256_si128(hi_to_lo(a)));
    }
}

namespace simd::epi64 {
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
        return _mm256_maskload_epi64(reinterpret_cast<const __int64*>(ptr), mask);
    }

    template <typename T>
    FORCE_INLINE void store(T* ptr, __m256i a) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), a);
    }

    template <typename T>
    FORCE_INLINE void maskstore(T* ptr, __m256i mask, __m256i src) {
        _mm256_maskstore_epi64(reinterpret_cast<__int64*>(ptr), mask, src);
    }

    FORCE_INLINE __m256i from_value(__int64 v) {
        return _mm256_set1_epi64x(v);
    }

    FORCE_INLINE __m256i from_values(__int64 a, __int64 b, __int64 c, __int64 d) {
        return _mm256_set_epi64x(d, c, b, a);
    }

    template <typename T>
    FORCE_INLINE __m256i gather32(const T* base, __m256i vindex) {
        return _mm256_i32gather_epi64(reinterpret_cast<const __int64*>(base), vindex, 8);
    }

    template <typename T>
    FORCE_INLINE __m256i gather64(const T* base, __m256i vindex) {
        return _mm256_i64gather_epi64(reinterpret_cast<const __int64*>(base), vindex, 8);
    }

    template <typename T>
    FORCE_INLINE __m256i maskgather32(__m256i src, const T* base, __m256i vindex, __m256i mask) {
        return _mm256_mask_i32gather_epi64(src, reinterpret_cast<const __int64*>(base), vindex, mask, 8);
    }

    template <typename T>
    FORCE_INLINE __m256i maskgatherz32(const T* base, __m256i vindex, __m256i mask) {
        return _mm256_mask_i32gather_epi64(_mm256_setzero_si256(), reinterpret_cast<const __int64*>(base), vindex, mask, 8);
    }

    FORCE_INLINE __m256i and256(__m256i a, __int64 val) {
        return _mm256_and_si256(a, from_value(val));
    }

    FORCE_INLINE __m128i to_128i(__m256i a) {
        return _mm256_castsi256_si128(a);
    }

    FORCE_INLINE __m256i cmpeq(__m256i a, __m256i b) {
        return _mm256_cmpeq_epi64(a, b);
    }

#if 0
    namespace detail {
        constexpr auto generate_popcnt_lookup() {
            /* Pad a bit so we can safely read a few bytes past the end */
            std::array<uint8_t, 264> res {};
            res[0ull << 0] = 8; // 0b00000000
            res[1ull << 0] = 0; // 0b00000001
            res[1ull << 1] = 1; // 0b00000010
            res[1ull << 2] = 2; // 0b00000100
            res[1ull << 3] = 3; // 0b00001000
            res[1ull << 4] = 4; // 0b00010000
            res[1ull << 5] = 5; // 0b00100000
            res[1ull << 6] = 6; // 0b01000000
            res[1ull << 7] = 7; // 0b10000000

            return res;
        }

        constexpr auto popcnt_lookup = generate_popcnt_lookup();
    }

    FORCE_INLINE __m256i ffs(__m256i v) {
        // TODO broken on res > 32...
        // TODO not used currently 
        /* Find the index of the first bit set, or 64. No integer division, so use an 8-part lookup */
        __m256i res = _mm256_setzero_si256();
        __m256i done_mask = _mm256_cmpeq_epi64(_mm256_and_si256(v, _mm256_set1_epi64x(1)), _mm256_set1_epi64x(1));
        /* Isolate the lowest bit of each element using 2's complement */
        v = _mm256_and_si256(v, _mm256_sub_epi64(_mm256_setzero_si256(), v));
        for (uint64_t i = 0; i < 8; ++i) {
            /* Extract the current byte being looked at */
            const __m256i masked = _mm256_and_si256(_mm256_srli_epi64(v, i * 8), _mm256_set1_epi64x(0xFF));

            /* Perform actual lookup */
            __m256i val = _mm256_i64gather_epi64(reinterpret_cast<const __int64*>(detail::popcnt_lookup.data()), masked, 1);
            val = _mm256_and_si256(val, _mm256_set1_epi64x(0xFF));
            //return val;
            /* Only add ones that are not done */
            res = _mm256_add_epi64(res, _mm256_andnot_si256(done_mask, val));

            /* Mark finished ones as done */
            const __m256i is_zero_mask = _mm256_cmpeq_epi64(val, _mm256_set1_epi64x(8));
            done_mask = _mm256_or_si256(done_mask, _mm256_xor_si256(is_zero_mask, _mm256_set1_epi64x(-1ll)));
        }

        return res;
    }
#endif

    template <typename T = __int64> requires (sizeof(T) == 8)
        FORCE_INLINE [[nodiscard]] std::array<T, 4> extract(__m256i a) {
        AVX_ALIGNED std::array<T, 4> res;
        epi32::store(res.data(), a);
        return res;
    }
}
