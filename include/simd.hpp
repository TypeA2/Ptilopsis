#pragma once

#include <ostream>

#include <immintrin.h>

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
                for (size_t i = 0; i < 3; ++i) {
                    os << reg.reg.m256i_i64[i] << ", ";
                }
                os << reg.reg.m256i_i64[3];
            } else {
                for (size_t i = 0; i < 3; ++i) {
                    os << reg.reg.m256i_u64[i] << ", ";
                }
                os << reg.reg.m256i_u64[3];
            }
            break;

        case 32:
            if constexpr (is_signed) {
                for (size_t i = 0; i < 7; ++i) {
                    os << reg.reg.m256i_i32[i] << ", ";
                }
                os << reg.reg.m256i_i32[7];
            } else {
                for (size_t i = 0; i < 7; ++i) {
                    os << reg.reg.m256i_u32[i] << ", ";
                }
                os << reg.reg.m256i_u32[7];
            }
            break;

        case 16:
            if constexpr (is_signed) {
                for (size_t i = 0; i < 15; ++i) {
                    os << reg.reg.m256i_i16[i] << ", ";
                }
                os << reg.reg.m256i_i16[15];
            } else {
                for (size_t i = 0; i < 15; ++i) {
                    os << reg.reg.m256i_u16[i] << ", ";
                }
                os << reg.reg.m256i_u16[15];
            }
            break;

        case 8:
            if constexpr (is_signed) {
                for (size_t i = 0; i < 31; ++i) {
                    os << static_cast<int>(reg.reg.m256i_i8[i]) << ", ";
                }
                os << static_cast<int>(reg.reg.m256i_i8[31]);
            } else {
                for (size_t i = 0; i < 31; ++i) {
                    os << static_cast<int>(reg.reg.m256i_u8[i]) << ", ";
                }
                os << static_cast<int>(reg.reg.m256i_u8[31]);
            }
            break;
    }

    os << ']';
    return os;
}

template <size_t bits = 32, bool is_signed = false>
std::ostream& print(std::ostream& os, __m256i reg) {
    os << avx_formatter<bits, is_signed>{ .reg = reg };
}

template <size_t bits = 32, bool is_signed = false>
auto format(__m256i reg) {
    return avx_formatter<bits, is_signed>{.reg = reg };
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
