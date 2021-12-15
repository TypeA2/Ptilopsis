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
#include <algorithm>
#include <ranges>
#include <iterator>
#include <cstring>

#include "simd.hpp"
#include "threading.hpp"
#include "utils.hpp"

constexpr uint64_t seed = 0xC0FFEEA4DBEEF;
constexpr size_t input_size = 1024*1024;
constexpr size_t repeats = 1e4;

/* Maximum value so that we don't overflow after all our additions */
constexpr size_t value_limit = std::numeric_limits<uint32_t>::max() / input_size;

/* Every thread handles at least 1 element.
 * 1 thread remains free to prevent starvation from the main thread
 */
static const size_t thread_count = std::min<size_t>(std::thread::hardware_concurrency() / 2, input_size);

/* Amount of elements processed by each thread.  */
static const size_t partition_size = ((input_size + thread_count) / thread_count);

static_assert((input_size & 0xF) == 0, "input size must be a multiple of 16 (AVX-512-support)");

using dataset = std::unique_ptr<uint32_t[]>;
using std::chrono::steady_clock;

template <size_t Count>
inline std::ostream& operator<<(std::ostream& os, const std::array<uint32_t, Count>& vec) {
    for (const auto& v : vec) {
        os << std::setw(3) << v << ' ';
    }

    return os;
}

std::unique_ptr<uint32_t[]> make_dataset() {
    return std::unique_ptr<uint32_t[]>{ new (std::align_val_t{AVX_ALIGNMENT}) uint32_t[input_size] };
}

class tester {
    bool plot;
    std::unique_ptr<uint32_t[]> input;
    std::unique_ptr<uint32_t[]> reference;

    public:
    tester() = delete;
    tester(bool plot, std::unique_ptr<uint32_t[]> input, std::unique_ptr<uint32_t[]>reference)
        : plot{ plot }, input{ std::move(input) }, reference{ std::move(reference) } { }

    template <std::invocable<std::span<const uint32_t>, std::span<uint32_t>> Func>
    void operator()(std::string_view name, Func&& f) {
        auto output = make_dataset();

        std::cerr << std::setw(32) << std::left << name << ' ';

        std::vector<std::chrono::nanoseconds> times(repeats);

        for (size_t i = 0; i < repeats; ++i) {
            
            /* Touch all memory
             * Doing so seems to remove a weird temporary drop in performance
             * after processing ~400 million elements in the scalar implementation.
             */
            /*
            for (volatile size_t j = 0; j < input_size;) {
                input[j] = input[j];
                j = j + 1;
            }

            std::memset(output.get(), 0, input_size * sizeof(uint32_t));
            */

            auto begin = steady_clock::now();

            f({ input.get(), input_size }, { output.get(), input_size });

            times[i] = steady_clock::now() - begin;

        }

        if (!std::equal(reference.get(), reference.get() + input_size, output.get())) {
            std::cerr << "INCORRECT ";
        }

        if (plot) {
            /* Output to stdout in a simple line-based format:
             * <name>,<comma-separated elements, in integer nanoseconds>
             */
            std::cout << name << ',';
            std::ranges::for_each(times, [](auto ns) {
                std::cout << ns.count() << ',';
            });
            std::cout << '\n';
        }

        std::ranges::sort(times);

        auto total = std::accumulate(times.begin(), times.end(), std::chrono::nanoseconds{});

        long double mean = static_cast<long double>(total.count()) / repeats;

        long double temp_sum = 0.L;
        for (size_t i = 0; i < repeats; ++i) {
            long double distance = std::abs(times[i].count() - mean);
            temp_sum += distance * distance;
        }

        std::chrono::nanoseconds stdev{ std::llround(std::sqrt(temp_sum / repeats)) };

        std::cerr << (total / repeats) << ", " << total << ", "
                  << times.front() << ", " << times.back()
                  << ", " << times[repeats / 2] << ", " << stdev << '\n';
    }
};

static tester setup(bool plot) {
    std::cerr << "Elements:      " << input_size << "\n"
                 "Repeats:       " << repeats << "\n"
                 "Value ranges:  [0, " << value_limit << "]\n"
                 "Random seed:   " << seed << "\n"
                 "Thread count:  " << std::thread::hardware_concurrency()
                    << " (" << thread_count << " used by operations)\n"
                 "Partition size: " << partition_size << '\n';

    std::cerr << "\nGenerating values...    ";
    auto input = make_dataset();

    {
        auto begin = steady_clock::now();
        std::mt19937_64 rnd{ seed };
        std::uniform_int_distribution<uint32_t> gen{ 0, value_limit };
        std::generate_n(input.get(), input_size, [&]{ return gen(rnd); });
        std::cerr << (steady_clock::now() - begin) << '\n';
    }

    std::cerr << "Generating reference... ";
    auto reference = make_dataset();
    
    {
        auto begin = steady_clock::now();
        std::inclusive_scan(input.get(), input.get() + input_size, reference.get());
        std::cerr << (steady_clock::now() - begin) << '\n';
    }

    std::cerr << "\n                                 avg, total, min, max, median, stdev\n";

    return { plot, std::move(input), std::move(reference) };
}

int main(int argc, char** argv) {
    bool plot = (argc > 1 && argv[1] == std::string_view{ "-p" });
    tester test = setup(plot);

    /* Basic scalar code */
    test("Scalar", [](std::span<const uint32_t> input, std::span<uint32_t> output) {
        output[0] = input[0];
        for (size_t i = 1; i < input.size(); ++i) {
            output[i] = output[i - 1] + input[i];
        }
    });

    /* SSE 1 and AVX 1 partially from [ADMS20_05] and https://stackoverflow.com/a/19519287/8662472 */
    
    /* SSE using shuffle to retrieve an offset */
    test("SSE 1", [](std::span<const uint32_t> input, std::span<uint32_t> output) {
        __m128i offset = _mm_setzero_si128();
        
        /* Process 4 integers at a time */
        for (size_t i = 0; i < input.size(); i += 4) {
            /* Offset by # of processed integers */
            __m128i x = _mm_load_si128(reinterpret_cast<const __m128i*>(input.data() + i));

            /* Perform ordinary [ADMS20_05] prefix sum */
            x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi32(x, _mm_slli_si128(x, 8));

            /* Add offset (last value of previous iteration) as an offsett */
            x = _mm_add_epi32(x, offset);

            /* Copy the highest value over to all other elements to c reate an offset */
            offset = _mm_shuffle_epi32(x, 0b11'11'11'11);

            /* Store result, offset again */
            _mm_store_si128(reinterpret_cast<__m128i*>(output.data() + i), x);
        }
    });

    /* AVX using the same principle as SSE 1, but with emulated shift */
    test("AVX 1", [](std::span<const uint32_t> input, std::span<uint32_t> output) {
        __m256i offset = _mm256_setzero_si256();
        
        /* 8 integers at a time, works because we have a guaranteed multiple of 16 */
        for (size_t i = 0; i < input.size(); i += 8) {
            __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data() + i));

            /* Prefix sum */
            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<4>(x));
            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<8>(x));
            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<16>(x));

            /* Offset */
            x = _mm256_add_epi32(x, offset);

            /* Copy uppper half to lower half */
            offset = _mm256_permute2x128_si256(x, x, 0b0'001'0'001);

            /* Of each half, copper last value to all other positions */
            offset = _mm256_shuffle_epi32(offset, 0b11'11'11'11);

            _mm256_store_si256(reinterpret_cast<__m256i*>(output.data() + i), x);
        }
    });

    /* Multi-threaded scalar without accumulate */
    {
        std::vector<std::thread> threads;
        threads.reserve(thread_count);

        std::vector<uint32_t> sums(thread_count);

        /* Used to signal the start of an iteration for testing */
        atomic_barrier wait_start = true;

        /* Used to synchronize startup of all threads */
        counting_barrier start_sync = 0;

        /* Tells all threads to stop */
        std::atomic_bool end = false;

        /* Synchronizes all threads between the 2 passes */
        counting_barrier sync = 0;

        /* Used to wait for all threads to finish 1 iteration */
        counting_barrier done = 0;

        /* Pointers for communicating in- and output */
        std::span<const uint32_t>* input_ptr = nullptr;
        std::span<uint32_t>* output_ptr = nullptr;

        for (size_t i = 0; i < thread_count; ++i) {
            threads.emplace_back([&, i=i] {
                size_t start_offset = i * partition_size;

                /* Past-the-end index of the range of elements this thread will process */
                size_t end_offset = std::min<size_t>(start_offset + partition_size, input_size);
                
                for (;;) {
                    /* Wait for start to be signaled */
                    wait_start.wait();
                    
                    /* Possibly exit */
                    if (end.load(std::memory_order_relaxed) == true) {
                        return;
                    }

                    /* Wait for all threads to have received the startup signal */
                    start_sync += 1;
                    start_sync.wait_for(thread_count);

                    /* No thread may be waiting for wait_start anymore since:
                     *  - All threads must have passed the start_sync barrier
                     *  - No threads may have passed the sync barrier (since the thread performing
                     *       this unlock is also one of the thread required to pass it)
                     */
                    if (i == 0) {
                        wait_start.lock();
                    }

                    auto input = *input_ptr;
                    auto output = *output_ptr;

                    /* First compute the local prefix sum */
                    output[start_offset] = input[start_offset];
                    for (size_t j = start_offset + 1; j < end_offset; ++j) {
                        output[j] = output[j - 1] + input[j];
                    }

                    sums[i] = output[end_offset - 1];

                    sync += 1;
                    sync.wait_for(thread_count);

                    /* Compute local offset based on previous offsets */
                    uint32_t local_sum = 0;
                    for (size_t j = 0; j < i; ++j) {
                        local_sum += sums[j];
                    }

                    /* Apply local offset */
                    for (size_t j = start_offset; j < end_offset; ++j) {
                        output[j] += local_sum;
                    }

                    done += 1;
                }
            });
        }

        test("Multi Scalar 1 (prefix + offset)", [&](std::span<const uint32_t> input, std::span<uint32_t> output) {
            input_ptr = &input;
            output_ptr = &output;

            /* Start processing */
            wait_start.unlock();

            /* Wait for all threads to finish */
            done.wait_for(thread_count);

            /* Reset state */
            input_ptr = nullptr;
            output_ptr = nullptr;
            
            /* wait_start was already reset by thread 0 */
            start_sync = 0;
            sync = 0;
            done = 0;
        });

        /* Signal end for all threads */
        end.store(true, std::memory_order_relaxed);
        wait_start.unlock();

        for (auto& th : threads) {
            th.join();
        }
    }

    /* Multi-threaded scalar accumulate  */
    {
        std::vector<std::thread> threads;
        threads.reserve(thread_count);

        std::vector<uint32_t> sums(thread_count);

        /* Used to signal the start of an iteration for testing */
        atomic_barrier wait_start = true;

        /* Used to synchronize startup of all threads */
        counting_barrier start_sync = 0;

        /* Tells all threads to stop */
        std::atomic_bool end = false;

        /* Synchronizes all threads between the 2 passes */
        counting_barrier sync = 0;

        /* Used to wait for all threads to finish 1 iteration */
        counting_barrier done = 0;

        /* Pointers for communicating in- and output */
        std::span<const uint32_t>* input_ptr = nullptr;
        std::span<uint32_t>* output_ptr = nullptr;

        for (size_t i = 0; i < thread_count; ++i) {
            threads.emplace_back([&, i=i] {
                size_t start_offset = i * partition_size;

                /* Past-the-end index of the range of elements this thread will process */
                size_t end_offset = std::min<size_t>(start_offset + partition_size, input_size);
                
                for (;;) {
                    /* Wait for start to be signaled */
                    wait_start.wait();
                    
                    /* Possibly exit */
                    if (end.load(std::memory_order_relaxed) == true) {
                        return;
                    }

                    /* Wait for all threads to have received the startup signal */
                    start_sync += 1;
                    start_sync.wait_for(thread_count);

                    /* No thread may be waiting for wait_start anymore since:
                     *  - All threads must have passed the start_sync barrier
                     *  - No threads may have passed the sync barrier (since the thread performing
                     *       this unlock is also one of the thread required to pass it)
                     */
                    if (i == 0) {
                        wait_start.lock();
                    }

                    auto input = *input_ptr;
                    auto output = *output_ptr;

                    /* First compute the local accumulated value */
                    uint32_t sum = 0;
                    for (size_t j = start_offset; j < end_offset; ++j) {
                        sum += input[j];
                    }

                    sums[i] = sum;

                    sync += 1;
                    sync.wait_for(thread_count);

                    /* Compute local offset based on previous offsets */
                    uint32_t local_sum = 0;
                    for (size_t j = 0; j < i; ++j) {
                        local_sum += sums[j];
                    }

                    /* Calculate prefix sum with offset */
                    output[start_offset] = input[start_offset] + local_sum;
                    for (size_t j = start_offset + 1; j < end_offset; ++j) {
                        output[j] = output[j - 1] + input[j];
                    }

                    done += 1;
                }
            });
        }

        test("Multi Scalar 2 (acc + prefix)", [&](std::span<const uint32_t> input, std::span<uint32_t> output) {
            input_ptr = &input;
            output_ptr = &output;

            /* Start processing */
            wait_start.unlock();

            /* Wait for all threads to finish */
            done.wait_for(thread_count);

            /* Reset state */
            input_ptr = nullptr;
            output_ptr = nullptr;
            
            /* wait_start was already reset by thread 0 */
            start_sync = 0;
            sync = 0;
            done = 0;
        });

        /* Signal end for all threads */
        end.store(true, std::memory_order_relaxed);
        wait_start.unlock();

        for (auto& th : threads) {
            th.join();
        }
    }

    return 1;
}
