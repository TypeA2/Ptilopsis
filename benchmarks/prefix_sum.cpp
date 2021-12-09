#include <iostream>
#include <random>
#include <chrono>
#include <numeric>
#include <array>
#include <iomanip>
#include <span>
#include <concepts>

#include <immintrin.h>

constexpr uint64_t seed = 0xC0FFEEA4DBEEF;
constexpr size_t input_size = 8;
constexpr size_t iterations = 1e8;

using dataset = std::array<uint32_t, input_size>;
using std::chrono::steady_clock;

template <std::unsigned_integral Int>
inline std::ostream& operator<<(std::ostream& os, const std::array<Int, input_size>& vec) {
    for (const auto& v : vec) {
        os << std::setw(3) << static_cast<uint32_t>(v) << ' ';
    }

    return os;
}

inline std::ostream& operator<<(std::ostream& os, std::chrono::nanoseconds ns) {
    auto count = ns.count();
    if (count < 1'000) {
        os << count << " ns";
    } else if (count < 1'000'000) {
        os << (count / 1e3) << " us";
    } else if (count < 1'000'000'000) {
        os << (count / 1e6) << " ms";
    } else {
        os << (count / 1e9) << " s";
    }

    return os;
}

template <uint8_t bytes>
inline __m256i __attribute__((__always_inline__)) _mm256_slli_si256_dual(__m256i x) {
    /* As per:
     *   https://stackoverflow.com/a/25264853/8662472
     */

    /* Left shift by 128 bits:
     * dest[127:0] := 0; 
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

void test_shifts(const dataset& input) {
    auto test = [&]<uint32_t shift>(){
        alignas(64) dataset avx{};

        __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data()));
        
        x = _mm256_slli_si256_dual<shift>(x);

        _mm256_store_si256(reinterpret_cast<__m256i*>(avx.data()), x);

        std::cerr << std::setw(2) << shift << ": ";
        for (uint8_t i = 0; i < 8; ++i) {
            std::cerr << std::setw(3) << avx[i] << ' ';
        }
        std::cerr << '\n';
    };

    test.template operator()<0>();
    test.template operator()<4>();
    test.template operator()<8>();
    test.template operator()<12>();
    test.template operator()<16>();
    test.template operator()<20>();
    test.template operator()<24>();
    test.template operator()<28>();
    test.template operator()<32>();
}

int main() {
    alignas(64) dataset input;
    std::mt19937_64 rnd{ seed };
    std::uniform_int_distribution<uint8_t> gen{ 0, 255 };
    std::generate_n(input.begin(), input_size, [&]{ return gen(rnd); });

    std::cerr << std::setfill(' ') << "Input:  " << input << "\n\n";

    steady_clock::duration stl_time;
    dataset stl;

    {
        auto begin = steady_clock::now();
        for (volatile size_t i = 0; i < iterations;) {
            std::inclusive_scan(input.begin(), input.end(), stl.begin());

            i = i + 1;
        }
        stl_time = steady_clock::now() - begin;
    }

    std::cerr << "STL took " << stl_time << "\nResult: " << stl << "\n\n";

    steady_clock::duration avx2_time;
    alignas(64) dataset avx2{};
    {
        auto begin = steady_clock::now();
        
        for (volatile size_t i = 0; i < iterations;) {
            __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data()));
            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<4>(x));
            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<8>(x));
            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<16>(x));
            _mm256_store_si256(reinterpret_cast<__m256i*>(avx2.data()), x);

            i = i + 1;
        }
        avx2_time = steady_clock::now() - begin;
    }

    std::cerr << "AVX2 took " << avx2_time << "\nResult: " << avx2 << "\n\n";

    steady_clock::duration avx_time;
    alignas(64) dataset avx{};
    {
        auto begin = steady_clock::now();
        for (volatile size_t i = 0; i < iterations;) {
            asm("");
            __m128i x = _mm_load_si128(reinterpret_cast<const __m128i*>(input.data()));
            x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
            

            // TODO: swapping these seems to be marginally faster?
            _mm_store_si128(reinterpret_cast<__m128i*>(avx.data()), x);
            __m128i y = _mm_load_si128(reinterpret_cast<const __m128i*>(input.data()) + 1);

            y = _mm_add_epi32(y, _mm_bsrli_si128(x, 12));
            y = _mm_add_epi32(y, _mm_slli_si128(y, 4));
            y = _mm_add_epi32(y, _mm_slli_si128(y, 8));
            
            _mm_store_si128(reinterpret_cast<__m128i*>(avx.data()) + 1, y);

            i = i + 1;
        }
        avx_time = steady_clock::now() - begin;
    }

    std::cerr << "AVX took " << avx_time << "\nResult: " << avx << "\n\n";

    return 1;
}
