#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <concepts>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#   include <intrin.h>
#else
/* CPUID is an intrinsic on MSVC */
#   include <cpuid.h>
#endif

#include <cxxopts.hpp>

#include "simd.hpp"
#include "threading.hpp"
#include "utils.hpp"

[[maybe_unused]] constexpr size_t cache_line_size = 64;

constexpr uint64_t seed = 0xC0FFEEA4DBEEF;
constexpr size_t input_size = 1024ull * 1024ull;
constexpr size_t repeats = 10'000;

/* Maximum value so that we don't overflow after all our additions */
constexpr size_t value_limit = std::numeric_limits<uint32_t>::max() / input_size;

/* Every thread handles at least 1 element.
 * 1 thread remains free to prevent starvation from the main thread
 */
static size_t thread_count = std::min<size_t>(std::thread::hardware_concurrency() - 1, input_size);

constexpr auto divide_and_round_up(std::integral auto n, std::integral auto d) {
    return (n + d - 1) / d;
}

constexpr auto round_up_to_multiple(std::integral auto n, std::integral auto m) {
    return ((n + m - 1) / m) * m;
}

/* Amount of elements processed by each thread.  */
static size_t partition_size = divide_and_round_up(input_size, thread_count);
static size_t partition_size_add1 = divide_and_round_up(input_size, thread_count + 1);
static size_t partition_size_align4 = round_up_to_multiple(partition_size, 4);
static size_t partition_size_align8 = round_up_to_multiple(partition_size, 8);

static_assert((input_size & 0xF) == 0, "input size must be a multiple of 16 (AVX-512-support)");

struct dataset_deleter {
    void operator()(uint32_t* ptr) const {
        operator delete(ptr, std::align_val_t{ AVX_ALIGNMENT });
    }
};

using dataset = std::unique_ptr<uint32_t[], dataset_deleter>;
using std::chrono::steady_clock;

template <size_t Count>
std::ostream& operator<<(std::ostream& os, const std::array<uint32_t, Count>& vec) {
    for (const auto& v : vec) {
        os << std::setw(3) << v << ' ';
    }

    return os;
}

dataset make_dataset() {
    /* Require manual aligned allocation on MSVC:
     * https://developercommunity.visualstudio.com/t/using-c17-new-stdalign-val-tn-syntax-results-in-er/528320
     */
    void* ptr = operator new[](sizeof(uint32_t) * input_size, std::align_val_t{ AVX_ALIGNMENT });
    return dataset{ static_cast<uint32_t*>(ptr), {}};
}

template <typename T>
concept test_func = std::invocable<T, std::span<const uint32_t>, std::span<uint32_t>>;

template <typename T, typename U>
concept complex_test_func = std::invocable<T, U&, std::span<const uint32_t>, std::span<uint32_t>>;

class tester {
    bool plot;
    dataset input;
    dataset reference;

    public:
    tester() = delete;
    tester(bool plot, dataset input, dataset reference)
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
            long double distance = std::abs(static_cast<long double>(times[i].count()) - mean);
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
    cxxopts::Options options("prefix_sum", "prefix sum benchmark");
    options.add_options()
        ("h,help", "Print usage")
        ("p,plot", "Output for plotting", cxxopts::value<bool>()->default_value("false"))
        ("t,threads", "Number of threads", cxxopts::value<int64_t>()->default_value(std::to_string(std::thread::hardware_concurrency())));

    options.custom_help("[-p] [-t <threadcount>]");
    bool plot = false;
    try {
        auto res = options.parse(argc, argv);

        if (res.count("help")) {
            std::cout << options.help() << '\n';
            std::exit(EXIT_SUCCESS);
        }

        plot = res["plot"].as<bool>();
        auto val = res["threads"].as<int64_t>();
        if (val > 0) {
            thread_count = std::min<size_t>(val, input_size);
            partition_size = divide_and_round_up(input_size, thread_count);
            partition_size_add1 = divide_and_round_up(input_size, thread_count + 1);
            partition_size_align4 = round_up_to_multiple(partition_size, 4);
            partition_size_align8 = round_up_to_multiple(partition_size, 8);
        }
    } catch(const cxxopts::OptionException& e) {
        std::cerr << e.what() << '\n';
        std::cerr << options.help() << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::cerr << "Elements:           " << input_size << "\n"
                 "Repeats:            " << repeats << "\n"
                 "Value ranges:       [0, " << value_limit << "]\n"
                 "Random seed:        " << seed << "\n"
                 "Thread count:       " << std::thread::hardware_concurrency()
                    << " (" << thread_count << " used by operations)\n"
                 "Partition size:     " << partition_size << "\n"
                 "Partition size alt: " << partition_size_add1 << "\n"
                 "Partition size (4): " << partition_size_align4 << "\n"
                 "Partition size (8): " << partition_size_align8 << "\n";

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

void scalar(const tester& test) {
     /* Basic scalar code */
    test("Scalar", [](std::span<const uint32_t> input, std::span<uint32_t> output) {
        output[0] = input[0];
        for (size_t i = 1; i < input.size(); ++i) {
            output[i] = output[i - 1] + input[i];
        }
    });
}

void basic_sse(const tester& test) {
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

            /* Add offset (last value of previous iteration) as an offset */
            x = _mm_add_epi32(x, offset);

            /* Copy the highest value over to all other elements to c reate an offset */
            offset = _mm_shuffle_epi32(x, 0b11'11'11'11);

            /* Store result, offset again */
            _mm_store_si128(reinterpret_cast<__m128i*>(output.data() + i), x);
        }
    });
}

void basic_avx(const tester& test) {
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

void basic_multithreaded_no_acc(const tester& test) {
    /* Multi-threaded scalar without accumulate */
    struct mt_state {
        mt_state(const mt_state&) = delete;
        mt_state& operator=(const mt_state&) = delete;
        mt_state(mt_state&&) noexcept = delete;
        mt_state& operator=(mt_state&&) noexcept = delete;

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

void basic_multithreaded_acc(const tester& test) {
    /* Multi-threaded scalar accumulate  */
    struct mt_state {
        mt_state(const mt_state&) = delete;
        mt_state& operator=(const mt_state&) = delete;
        mt_state(mt_state&&) noexcept = delete;
        mt_state& operator=(mt_state&&) noexcept = delete;

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

void optimized_multithreaded_no_acc(const tester& test) {
    /* More efficient multithreaded scalar without accumulate */
    struct mt_state {
        mt_state(const mt_state&) = delete;
        mt_state& operator=(const mt_state&) = delete;
        mt_state(mt_state&&) noexcept = delete;
        mt_state& operator=(mt_state&&) noexcept = delete;

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

void optimized_multithreaded_acc(const tester& test) {
    /* Multi-threaded scalar accumulate  */
    struct mt_state {
        mt_state(const mt_state&) = delete;
        mt_state& operator=(const mt_state&) = delete;
        mt_state(mt_state&&) noexcept = delete;
        mt_state& operator=(mt_state&&) noexcept = delete;

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

void optimized_multithreaded_sse(const tester& test) {
    /* Multithreaded SSE prefix+offset */
    struct mt_state {
        mt_state(const mt_state&) = delete;
        mt_state& operator=(const mt_state&) = delete;
        mt_state(mt_state&&) noexcept = delete;
        mt_state& operator=(mt_state&&) noexcept = delete;

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
                    size_t start_offset = i * partition_size_align4;

                    /* Past-the-end index of the range of elements this thread will process */
                    size_t end_offset = std::min<size_t>(start_offset + partition_size_align4, input_size);

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

                        __m128i offset = _mm_setzero_si128();
                        /* 4 integers at a time */
                        for (size_t j = start_offset; j < end_offset; j += 4) {
                            __m128i x = _mm_load_si128(reinterpret_cast<const __m128i*>(input.data() + j));

                            /* Calculate the local prefix sum */
                            x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
                            x = _mm_add_epi32(x, _mm_slli_si128(x, 8));

                            /* Add offset from previous iteration */
                            x = _mm_add_epi32(x, offset);

                            /* Re-calculate offset for next iteration */
                            offset = _mm_shuffle_epi32(x, 0b11'11'11'11);

                            /* Store in destination */
                            _mm_store_si128(reinterpret_cast<__m128i*>(output.data() + j), x);
                        }

                        /* The offset contains the last value of the last iteration. This is our total sum */
                        sums[i] = _mm_extract_epi32(offset, 0);

                        sync += 1;
                        sync.wait_for(thread_count);

                        /* Calculate local sum */
                        uint32_t local_sum = 0;
                        for (size_t j = 0; j < i; ++j) {
                            local_sum += sums[j];
                        }

                        offset = _mm_set1_epi32(local_sum);

                        /* Add local sum to every element */
                        for (size_t j = start_offset; j < end_offset; j += 4) {
                            __m128i x = _mm_load_si128(reinterpret_cast<const __m128i*>(output.data() + j));
                            x = _mm_add_epi32(x, offset);
                            _mm_store_si128(reinterpret_cast<__m128i*>(output.data() + j), x);
                        }

                        done += 1;
                    }
                });
            }
        }
    } state;

    test("Multi SSE 1", state);
}

void optimized_multithreaded_avx(const tester& test) {
    /* Multithreaded AVX prefix+offset */
    struct mt_state {
        mt_state(const mt_state&) = delete;
        mt_state& operator=(const mt_state&) = delete;
        mt_state(mt_state&&) noexcept = delete;
        mt_state& operator=(mt_state&&) noexcept = delete;

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
                    size_t start_offset = i * partition_size_align8;

                    /* Past-the-end index of the range of elements this thread will process */
                    size_t end_offset = std::min<size_t>(start_offset + partition_size_align8, input_size);

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

                        __m256i offset = _mm256_setzero_si256();
                        /* 4 integers at a time */
                        for (size_t j = start_offset; j < end_offset; j += 8) {
                            __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data() + j));

                            /* Calculate the local prefix sum */
                            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<4>(x));
                            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<8>(x));
                            x = _mm256_add_epi32(x, _mm256_slli_si256_dual<16>(x));

                            /* Add offset from previous iteration */
                            x = _mm256_add_epi32(x, offset);

                            /* Copy uppper half to lower half */
                            offset = _mm256_permute2x128_si256(x, x, 0b0'001'0'001);

                            /* Of each half, copper last value to all other positions */
                            offset = _mm256_shuffle_epi32(offset, 0b11'11'11'11);

                            /* Store in destination */
                            _mm256_store_si256(reinterpret_cast<__m256i*>(output.data() + j), x);
                        }

                        /* The offset contains the last value of the last iteration. This is our total sum */
                        sums[i] = _mm256_extract_epi32(offset, 0);

                        sync += 1;
                        sync.wait_for(thread_count);

                        /* Calculate local sum */
                        uint32_t local_sum = 0;
                        for (size_t j = 0; j < i; ++j) {
                            local_sum += sums[j];
                        }

                        offset = _mm256_set1_epi32(local_sum);

                        /* Add local sum to every element */
                        for (size_t j = start_offset; j < end_offset; j += 8) {
                            __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(output.data() + j));
                            x = _mm256_add_epi32(x, offset);
                            _mm256_store_si256(reinterpret_cast<__m256i*>(output.data() + j), x);
                        }

                        done += 1;
                    }
                });
            }
        }
    } state;

    test("Multi AVX 1", state);
}

static void(*tests[])(const tester&) {
    scalar,
    basic_sse,
    basic_avx,
    basic_multithreaded_no_acc,
    basic_multithreaded_acc,
    optimized_multithreaded_no_acc,
    optimized_multithreaded_acc,
    optimized_multithreaded_sse,
    optimized_multithreaded_avx,
};

void cpuid_dump() {
    
#ifdef _MSC_VER
    int res[4];
    int valid = 1;
    __cpuid(res, 1);
    auto [eax, ebx, ecx, edx] = res;
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    int valid = __get_cpuid(1, &eax, &ebx, &ecx, &edx);
#endif

    if (valid == 1) {
        std::cerr << "eax: " << std::bitset<32>(eax) << "\n"
                     "ebx: " << std::bitset<32>(ebx) << "\n"
                     "ecx: " << std::bitset<32>(ecx) << "\n"
                     "edx: " << std::bitset<32>(edx) << "\n";

        std::vector<std::pair<std::string_view, bool>> vals {
            { "fpu:   ", edx & (1u <<  0) },
            { "cx8:   ", edx & (1u <<  8) },
            { "mmx:   ", edx & (1u << 23) },
            { "sse:   ", edx & (1u << 25) },
            { "sse2:  ", edx & (1u << 26) },
            { "htt:   ", edx & (1u << 28) },
            { "tm:    ", edx & (1u << 29) },
            { "ia64:  ", edx & (1u << 30) },
            { "sse3:  ", ecx & (1u <<  0) },
            { "tm2:   ", ecx & (1u <<  8) },
            { "ssse3: ", ecx & (1u <<  9) },
            { "fma:   ", ecx & (1u << 12) },
            { "cx16:  ", ecx & (1u << 13) },
            { "sse4.1:", ecx & (1u << 19) },
            { "sse4.2:", ecx & (1u << 20) },
            { "movbe: ", ecx & (1u << 22) },
            { "popcnt:", ecx & (1u << 23) },
            { "aes:   ", ecx & (1u << 25) },
            { "avx:   ", ecx & (1u << 28) },
            { "f16c:  ", ecx & (1u << 29) },
            { "rdrnd: ", ecx & (1u << 30) },
            { "hyperv:", ecx & (1u << 31) },
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
