#include <iostream>
#include <random>
#include <chrono>
#include <numeric>
#include <array>
#include <iomanip>
#include <span>
#include <concepts>

#include "simd.hpp"
#include "utils.hpp"

constexpr uint64_t seed = 0xC0FFEEA4DBEEF;
constexpr size_t input_size = 768;
constexpr size_t iterations = 1e6;

using dataset = std::array<uint32_t, input_size>;
using std::chrono::steady_clock;

template <std::unsigned_integral Int, size_t Count>
inline std::ostream& operator<<(std::ostream& os, const std::array<Int, Count>& vec) {
    for (const auto& v : vec) {
        os << std::setw(3) << static_cast<uint32_t>(v) << ' ';
    }

    return os;
}

int main() {
    AVX_ALIGNED dataset input;
    std::mt19937_64 rnd{ seed };
    std::uniform_int_distribution<uint8_t> gen{ 0, 255 };
    std::generate_n(input.begin(), input_size, [&]{ return gen(rnd); });

    dataset stl;
    {
        std::cerr << "Benchmarking: STL...  ";
        steady_clock::duration stl_time;
        
        auto begin = steady_clock::now();
        
        for (volatile size_t i = 0; i < iterations;) {
            stl[0] = input[0];
            for (size_t i = 1; i < input_size; ++i) {
                stl[i] = stl[i - 1] + input[i];
            }

            i = i + 1;
        }
        stl_time = steady_clock::now() - begin;
        std::cerr << stl_time << '\n';
    }

    {
        std::cerr << "Benchmarking: AVX2... ";

        steady_clock::duration avx2_time;
        AVX_ALIGNED dataset avx2{};
        auto begin = steady_clock::now();
        
        for (volatile size_t i = 0; i < iterations;) {
            __m256i prev = _mm256_setzero_si256();
            for (size_t j = 0; j < input_size; j += 8) {
                // No C++23 for now
                //size_t remaining = std::min(input_size - j, size_t{8});

                __m256i x;

                //if (remaining == 8) {
                    /* Load data offset by index */
                    x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data() + j));
                // } else {

                //     /* Construct mask by setting the highest bit of all elements to load to 1 */
                //     AVX_ALIGNED std::array<uint32_t, 8> mask_template{};
                //     for (size_t k = 0; k < remaining; ++k) {
                //         mask_template[k] = 1u << 31;
                //     }

                //     /* Load mask and load using mask, leaving all other elements at 0 */
                //     __m256i mask = _mm256_load_si256(reinterpret_cast<__m256i*>(mask_template.data()));
                //     x = _mm256_maskload_epi32(reinterpret_cast<const int*>(input.data()) + j, mask);
                // }

                /* Add previous last element to first */
                x = _mm256_add_epi32(x, prev);
                x = _mm256_add_epi32(x, _mm256_slli_si256_dual<4>(x));
                x = _mm256_add_epi32(x, _mm256_slli_si256_dual<8>(x));
                x = _mm256_add_epi32(x, _mm256_slli_si256_dual<16>(x));
                prev = _mm256_srli_si256_dual<28>(x);

                //if (remaining == 8) {
                    /* Store at offset */
                    _mm256_store_si256(reinterpret_cast<__m256i*>(avx2.data() + j), x);

                    /* Last element into first, zero the rest */
                    
                // } else {
                //     /* Store using mask again */
                //     AVX_ALIGNED std::array<uint32_t, 8> mask_template{};
                //     for (size_t k = 0; k < remaining; ++k) {
                //         mask_template[k] = 1u << 31;
                //     }

                //     /* Load mask and store using mask */
                //     __m256i mask = _mm256_load_si256(reinterpret_cast<__m256i*>(mask_template.data()));
                //     _mm256_maskstore_epi32(reinterpret_cast<int*>(avx2.data()) + j, mask, x);
                // }
            }

            i = i + 1;
        }
        avx2_time = steady_clock::now() - begin;

        if (avx2 != stl) {
            std::cerr << "INCORRECT, took ";
        }

        std::cerr << avx2_time << '\n';
    }

    {
        std::cerr << "Benchmarking: AVX...  ";

        steady_clock::duration avx_time;
        AVX_ALIGNED dataset avx{};
        auto begin = steady_clock::now();
        for (volatile size_t i = 0; i < iterations;) {

            __m128i prev = _mm_setzero_si128();
            for (size_t j = 0; j < input_size; j += 4) {
                //size_t remaining = std::min(input_size - j, size_t{4});

                __m128i x;
                //if (remaining == 4) {
                    x = _mm_load_si128(reinterpret_cast<const __m128i*>(input.data() + j));
                // } else {
                //     AVX_ALIGNED std::array<uint32_t, 4> mask_template{};
                //     for (size_t k = 0; k < remaining; ++k) {
                //         mask_template[k] = 1u << 31;
                //     }

                //     __m128i mask = _mm_load_si128(reinterpret_cast<__m128i*>(mask_template.data()));
                //     x = _mm_maskload_epi32(reinterpret_cast<const int*>(input.data()) + j, mask);
                // }

                x = _mm_add_epi32(x, prev);
                x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
                x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
                prev = _mm_srli_si128(x, 12);

                //if (remaining == 4) {
                    _mm_store_si128(reinterpret_cast<__m128i*>(avx.data() + j), x);

                    
                // } else {
                //     AVX_ALIGNED std::array<uint32_t, 4> mask_template{};
                //     for (size_t k = 0; k < remaining; ++k) {
                //         mask_template[k] = 1u << 31;
                //     }

                //     __m128i mask = _mm_load_si128(reinterpret_cast<__m128i*>(mask_template.data()));
                //     _mm_maskstore_epi32(reinterpret_cast<int*>(avx.data()) + j, mask, x);
                // }
            }

            i = i + 1;
        }
        avx_time = steady_clock::now() - begin;

        if (avx != stl) {
            std::cerr << "INCORRECT, took ";
        }

        std::cerr << avx_time << '\n';
    }

    return 1;
}
