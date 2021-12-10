#include <iostream>
#include <random>
#include <chrono>
#include <numeric>
#include <array>
#include <iomanip>
#include <span>
#include <concepts>
#include <thread>
#include <vector>
#include <atomic>

#include "simd.hpp"
#include "utils.hpp"

constexpr uint64_t seed = 0xC0FFEEA4DBEEF;
constexpr size_t input_size = 131072;
constexpr size_t iterations = 1e5;

using dataset = std::array<uint32_t, input_size>;
using std::chrono::steady_clock;

template <size_t Count>
inline std::ostream& operator<<(std::ostream& os, const std::array<uint32_t, Count>& vec) {
    for (const auto& v : vec) {
        os << std::setw(3) << v << ' ';
    }

    return os;
}

int main() {
    AVX_ALIGNED dataset input;
    std::mt19937_64 rnd{ seed };
    std::uniform_int_distribution<uint8_t> gen{ 0, 255 };
    std::generate_n(input.begin(), input_size, [&]{ return gen(rnd); });

    dataset stl{};
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
        std::cerr << "Benchmarking: SSE...  ";

        steady_clock::duration sse_time;
        AVX_ALIGNED dataset sse{};
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
                    _mm_store_si128(reinterpret_cast<__m128i*>(sse.data() + j), x);

                    
                // } else {
                //     AVX_ALIGNED std::array<uint32_t, 4> mask_template{};
                //     for (size_t k = 0; k < remaining; ++k) {
                //         mask_template[k] = 1u << 31;
                //     }

                //     __m128i mask = _mm_load_si128(reinterpret_cast<__m128i*>(mask_template.data()));
                //     _mm_maskstore_epi32(reinterpret_cast<int*>(sse.data()) + j, mask, x);
                // }
            }

            i = i + 1;
        }
        sse_time = steady_clock::now() - begin;

        if (sse != stl) {
            std::cerr << "INCORRECT, took ";
        }

        std::cerr << sse_time << '\n';
    }
    
    {
        std::cerr << "Benchmarking: SSE alternative 1... ";

        steady_clock::duration sse_alt_time;
        AVX_ALIGNED dataset sse_alt{};
        auto begin = steady_clock::now();
        for (volatile size_t i = 0; i < iterations;) {
            // https://stackoverflow.com/a/19519287/8662472
            __m128i offset = _mm_setzero_si128();
            for (size_t j = 0; j < input_size; j += 4) {
                __m128i x = _mm_load_si128(reinterpret_cast<const __m128i*>(input.data() + j));
                x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
                x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
                x = _mm_add_epi32(x, offset);
                offset = _mm_shuffle_epi32(x, 0b11'11'11'11);
                _mm_store_si128(reinterpret_cast<__m128i*>(sse_alt.data() + j), x);
            }
            i = i + 1;
        }
        sse_alt_time = steady_clock::now() - begin;

        if (sse_alt != stl) {
            std::cerr << "INCORRECT, took ";
        }

        std::cerr << sse_alt_time << '\n';
    }

    {
        std::cerr << "Benchmarking: AVX alternative 1... ";

        steady_clock::duration avx_alt_time;
        AVX_ALIGNED dataset avx_alt{};
        auto begin = steady_clock::now();
        for (volatile size_t i = 0; i < iterations;) {
            // https://stackoverflow.com/a/19519287/8662472
            __m256i offset = _mm256_setzero_si256();
            for (size_t j = 0; j < input_size; j += 8) {
                __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data() + j));
                x = _mm256_add_epi32(x, _mm256_slli_si256_dual<4>(x));
                x = _mm256_add_epi32(x, _mm256_slli_si256_dual<8>(x));
                x = _mm256_add_epi32(x, _mm256_slli_si256_dual<16>(x));
                x = _mm256_add_epi32(x, offset);
                /* Copy upper half to lower half */
                __m256i tmp = _mm256_permute2x128_si256(x, x, 0b0'001'0'001);
                offset = _mm256_shuffle_epi32(tmp, 0b11'11'11'11);
                _mm256_store_si256(reinterpret_cast<__m256i*>(avx_alt.data() + j), x);
            }
            i = i + 1;
        }
        avx_alt_time = steady_clock::now() - begin;

        if (avx_alt != stl) {
            std::cerr << "INCORRECT, took ";
        }

        std::cerr << avx_alt_time << '\n';
    }

    {
        std::cerr << "Benchmarking: Multithreaded scalar 1... ";
        size_t thread_count = 23;//std::min<size_t>(std::thread::hardware_concurrency(), input_size);
        size_t partition_size = ((input_size + thread_count) / thread_count);

        std::vector<std::thread> threads;
        threads.reserve(thread_count);

        steady_clock::duration time;
        AVX_ALIGNED dataset result{};

        std::atomic_size_t idle = 0;
        std::atomic_flag wait;
        std::atomic_size_t done = 0;

        std::atomic_bool end = false;
        std::atomic_size_t sync1 = 0;
        std::vector<uint32_t> sums;

        sums.resize(thread_count);
        for (size_t i = 0; i < thread_count; ++i) {
            threads.emplace_back([&, i=i] {
                size_t start_offset = i * partition_size;
                size_t past_the_end = std::min<size_t>(start_offset + partition_size, input_size);

                idle += 1;

                while (end == false) {
                    // Wait for owner to reset
                    while (done.load(std::memory_order_relaxed) != 0) {
                        __builtin_ia32_pause();
                    }
                    wait.wait(false);

                    /* First pass */
                    result[start_offset] = input[start_offset];
                    for (size_t j = start_offset + 1; j < past_the_end; ++j) {
                        result[j] = result[j - 1] + input[j];
                    }

                    sums[i] = result[past_the_end - 1];

                    sync1 += 1;
                    while (sync1.load(std::memory_order_relaxed) < thread_count) {
                        __builtin_ia32_pause();
                    }

                    uint32_t our_sum = 0;
                    for (size_t j = 0; j < i; ++j) {
                        our_sum += sums[j];
                    }

                    for (size_t j = start_offset; j < past_the_end; ++j) {
                        result[j] += our_sum;
                    }

                    done += 1;
                }
            });
        }

        /* Wait for all threads to start */
        while (idle.load(std::memory_order_relaxed) != thread_count) {
            __builtin_ia32_pause();
        }

        auto begin = steady_clock::now();
        for (volatile size_t i = 0; i < iterations;) {
            if (i == (iterations - 1)) {
                end = true;
            }
            
            // Start all threads
            wait.test_and_set();
            wait.notify_all();

            // All are done and waiting again
            while (done.load(std::memory_order_relaxed) < thread_count) {
                __builtin_ia32_pause();
            }
            
            wait.clear();
            done = 0;
            sync1 = 0;
            sums = {};

            i = i + 1;
        }
        
        time = steady_clock::now() - begin;

        for (auto& th : threads) {
            th.join();
        }
        threads.clear();
        
        if (result != stl) {
            std::cerr << "INCORRECT, took ";
        }

        std::cerr << time << '\n';
    }

    return 1;
}
