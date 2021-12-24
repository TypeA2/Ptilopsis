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
#include <bitset>
#include <functional>
#include <mutex>

#include <cpuid.h>

#include "simd.hpp"
#include "threading.hpp"
#include "utils.hpp"

constexpr size_t cache_line_size = 64;

constexpr uint64_t seed = 0xC0FFEEA4DBEEF;
constexpr size_t input_size = 1024*1024;
constexpr size_t repeats = 1e4;

/* Maximum value so that we don't overflow after all our additions */
constexpr size_t value_limit = std::numeric_limits<uint32_t>::max() / input_size;

/* Every thread handles at least 1 element.
 * 1 thread remains free to prevent starvation from the main thread
 */
static size_t thread_count = std::min<size_t>(std::thread::hardware_concurrency() - 1, input_size);

/* Amount of elements processed by each thread.  */
static size_t partition_size = ((input_size + thread_count) / thread_count);
static size_t partition_size_add1 = ((input_size + (thread_count + 1)) / (thread_count + 1));

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

template <typename T>
concept test_func = std::invocable<T, std::span<const uint32_t>, std::span<uint32_t>>;

template <typename T, typename U>
concept complex_test_func = std::invocable<T, U&, std::span<const uint32_t>, std::span<uint32_t>>;

class tester {
    bool plot;
    std::unique_ptr<uint32_t[]> input;
    std::unique_ptr<uint32_t[]> reference;

    public:
    tester() = delete;
    tester(bool plot, std::unique_ptr<uint32_t[]> input, std::unique_ptr<uint32_t[]>reference)
        : plot{ plot }, input{ std::move(input) }, reference{ std::move(reference) } { }

    /* Test simple algorithms */
    void operator()(std::string_view name, test_func auto&& f) const {
        /* Algorithms will write to this */
        auto output = make_dataset();

        std::cerr << std::setw(32) << std::left << name << ' ';

        std::vector<std::chrono::nanoseconds> times(repeats);

        for (size_t i = 0; i < repeats; ++i) {
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

    /* Test complex algirhtms with additional state */
    template <typename T>
    void test_complex(std::string_view name, T& state, complex_test_func<T> auto&& f) const {
        this->operator()(name, [&](std::span<const uint32_t> input, std::span<uint32_t> output){
            f(state, input, output);
        });
    }
};

static tester setup(int argc, char** argv) {
    int c;
    bool plot = false;
    while ((c = getopt(argc, argv, "pt:")) != -1) {
        switch(c) {
            case 'p':
                plot = true;
                break;

            case 't': {
                auto val = std::stol(optarg);
                if (val > 0) {
                    thread_count = std::min<size_t>(val, input_size);
                    partition_size = ((input_size + thread_count) / thread_count);
                    partition_size_add1 = ((input_size + (thread_count + 1)) / (thread_count + 1));
                }
                break;
            }
        }
    }

    std::cerr << "Elements:           " << input_size << "\n"
                 "Repeats:            " << repeats << "\n"
                 "Value ranges:       [0, " << value_limit << "]\n"
                 "Random seed:        " << seed << "\n"
                 "Thread count:       " << std::thread::hardware_concurrency()
                    << " (" << thread_count << " used by operations)\n"
                 "Partition size:     " << partition_size << "\n"
                 "Partition size alt: " << partition_size_add1 << "\n";

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

void scalar(tester& test) {
     /* Basic scalar code */
    test("Scalar", [](std::span<const uint32_t> input, std::span<uint32_t> output) {
        output[0] = input[0];
        for (size_t i = 1; i < input.size(); ++i) {
            output[i] = output[i - 1] + input[i];
        }
    });
}

void basic_sse(tester& test) {
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
}

void basic_avx(tester& test) {
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
}

void basic_multithreaded_no_acc(tester& test) {
    /* Multi-threaded scalar without accumulate */
    struct mt_state {
        std::vector<std::thread> threads;
        std::vector<uint32_t> sums;

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

        void operator()(std::span<const uint32_t> input, std::span<uint32_t> output) {
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
        }

        ~mt_state() {
            /* Signal end for all threads */
            end.store(true, std::memory_order_relaxed);
            wait_start.unlock();

            for (auto& th : threads) {
                th.join();
            }
        }

        mt_state() {
            threads.reserve(thread_count);
            sums.resize(thread_count);
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
        }
    } state;

    test("Multi Scalar 1 (prefix + offset)", state);
}

void basic_multithreaded_acc(tester& test) {
    /* Multi-threaded scalar accumulate  */
    struct mt_state {
        std::vector<std::thread> threads;
        

        std::vector<uint32_t> sums;

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

        void operator()(std::span<const uint32_t> input, std::span<uint32_t> output) {
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
        };

        ~mt_state() {
            /* Signal end for all threads */
            end.store(true, std::memory_order_relaxed);
            wait_start.unlock();

            for (auto& th : threads) {
                th.join();
            }
        }

        mt_state() {
            threads.reserve(thread_count);
            sums.resize(thread_count);

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
        }
    } state;

    test("Multi Scalar 2 (acc + prefix)", state);
}

void optimized_multithreaded_no_acc(tester& test) {
    /* More efficient multithreaded scalar without accumulate */
    struct mt_state {
        std::vector<std::thread> threads;
        std::vector<uint32_t> sums;

        /* Used to signal the start of an iteration for testing */
        atomic_barrier wait_start { true };

        /* Used to synchronize startup of all threads */
        counting_barrier start_sync { 0 };

        /* Tells all threads to stop */
        std::atomic_bool end = false;

        /* Synchronizes all threads between the 2 passes */
        counting_barrier sync = 0;

        /* Used to wait for all threads to finish 1 iteration */
        counting_barrier done = 0;

        /* Pointers for communicating in- and output */
        std::span<const uint32_t>* input_ptr = nullptr;
        std::span<uint32_t>* output_ptr = nullptr;

        std::mutex io_mutex;

        void operator()(std::span<const uint32_t> input, std::span<uint32_t> output) {
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
        }

        ~mt_state() {
            /* Signal end for all threads */
            end.store(true, std::memory_order_relaxed);
            wait_start.unlock();

            for (auto& th : threads) {
                th.join();
            }
        }

        mt_state() {
            threads.reserve(thread_count);
            sums.resize(thread_count);

            for (size_t i = 0; i < thread_count; ++i) {
                threads.emplace_back([&, i=i] {
                    size_t start_offset = i * partition_size_add1;

                    /* Past-the-end index of the range of elements this thread will0 process */
                    size_t end_offset = std::min<size_t>(start_offset + partition_size_add1, input_size);
                    
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

                        if (i == 0) {
                            start_offset = thread_count * partition_size_add1;
                            /* First thread computes prefix sum with offset, all others just add offset */
                            uint32_t local_sum = 0;
                            for (size_t j = 0; j < thread_count; ++j) {
                                local_sum += sums[j];
                            }

                            output[start_offset] = input[start_offset] + local_sum;
                            for (size_t j = start_offset + 1; j < input_size; ++j) {
                                output[j] = output[j - 1] + input[j];
                            }


                        } else {
                            /* Compute local offset based on previous offsets */
                            uint32_t local_sum = 0;
                            for (size_t j = 0; j < i; ++j) {
                                local_sum += sums[j];
                            }

                            /* Apply local offset */
                            for (size_t j = start_offset; j < end_offset; ++j) {
                                output[j] += local_sum;
                            }
                        }

                        done += 1;
                    }
                });
            }
        }
    } state;

    test("Multi Scalar 3 (shift; p+o)", state);
}

void optimized_multithreaded_acc(tester& test) {
    /* Multi-threaded scalar accumulate  */
    struct mt_state {
        std::vector<std::thread> threads;
        

        std::vector<uint32_t> sums;

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

        void operator()(std::span<const uint32_t> input, std::span<uint32_t> output) {
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
        };

        ~mt_state() {
            /* Signal end for all threads */
            end.store(true, std::memory_order_relaxed);
            wait_start.unlock();

            for (auto& th : threads) {
                th.join();
            }
        }

        mt_state() {
            threads.reserve(thread_count);
            sums.resize(thread_count);

            for (size_t i = 0; i < thread_count; ++i) {
                threads.emplace_back([&, i=i] {
                    size_t start_offset = i * partition_size_add1;

                    /* Past-the-end index of the range of elements this thread will process */
                    size_t end_offset = std::min<size_t>(start_offset + partition_size_add1, input_size);
                    
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

                        /* Thread 0 only calculates a prefix sum, the rest a total sum */
                        if (i == 0) {
                            output[0] = input[0];
                            for (size_t j = 1; j < partition_size_add1; ++j) {
                                output[j] = output[j - 1] + input[j];
                            }

                            sums[0] = output[end_offset - 1];
                        } else {
                            /* Compute the local accumulated value */
                            uint32_t sum = 0;
                            for (size_t j = start_offset; j < end_offset; ++j) {
                                sum += input[j];
                            }

                            sums[i] = sum;
                        }

                        sync += 1;
                        sync.wait_for(thread_count);

                        
                        if (i == 0) {
                            /* Turn this into the last thread */
                            size_t new_start = thread_count * partition_size_add1;

                            uint32_t local_sum = 0;
                            for (size_t j = 0; j < thread_count; ++j) {
                                local_sum += sums[j];
                            }

                            /* Calculate prefix sum with offset */
                            output[new_start] = input[new_start] + local_sum;
                            for (size_t j = new_start + 1; j < input_size; ++j) {
                                output[j] = output[j - 1] + input[j];
                            }
                        } else {
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
                        }

                        done += 1;
                    }
                });
            }
        }
    } state;

    test("Multi Scalar 4 (shift; a+p)", state);
}


static void(*tests[])(tester&) {
    scalar,
    basic_sse,
    basic_avx,
    basic_multithreaded_no_acc,
    basic_multithreaded_acc,
    optimized_multithreaded_no_acc,
    optimized_multithreaded_acc,
};

void cpuid_dump() {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    int valid = __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    if (valid == 1) {
        std::cerr << "eax: " << std::bitset<32>(eax) << "\n"
                     "ebx: " << std::bitset<32>(ebx) << "\n"
                     "ecx: " << std::bitset<32>(ecx) << "\n"
                     "edx: " << std::bitset<32>(edx) << "\n";

        std::vector<std::pair<std::string_view, bool>> vals {
            { "fpu:   ", edx & (1 <<  0) },
            { "cx8:   ", edx & (1 <<  8) },
            { "mmx:   ", edx & (1 << 23) },
            { "sse:   ", edx & (1 << 25) },
            { "sse2:  ", edx & (1 << 26) },
            { "htt:   ", edx & (1 << 28) },
            { "tm:    ", edx & (1 << 29) },
            { "ia64:  ", edx & (1 << 30) },
            { "sse3:  ", ecx & (1 <<  0) },
            { "tm2:   ", ecx & (1 <<  8) },
            { "ssse3: ", ecx & (1 <<  9) },
            { "fma:   ", ecx & (1 << 12) },
            { "cx16:  ", ecx & (1 << 13) },
            { "sse4.1:", ecx & (1 << 19) },
            { "sse4.2:", ecx & (1 << 20) },
            { "movbe: ", ecx & (1 << 22) },
            { "popcnt:", ecx & (1 << 23) },
            { "aes:   ", ecx & (1 << 25) },
            { "avx:   ", ecx & (1 << 28) },
            { "f16c:  ", ecx & (1 << 29) },
            { "rdrnd: ", ecx & (1 << 30) },
            { "hyperv:", ecx & (1 << 31) },
        };

        for (auto& [k, v] : vals) {
            std::cerr << "  " << k << ' ' << v << '\n';
        }
    } else {
        std::cerr << "invalid cpuid\n";
    }
}

int main(int argc, char** argv) {
    tester test = setup(argc, argv);

    /* SSE 1 and AVX 1 partially from [ADMS20_05] and https://stackoverflow.com/a/19519287/8662472 */

    for (auto func : tests) {
        func(test);
    }

    return 1;
}
