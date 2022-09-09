#pragma once

#include <array>
#include <thread>
#include <barrier>
#include <functional>
#include <chrono>
using steady_clock = std::chrono::steady_clock;

#include "codegen/rv_generator.hpp"
#include "utils.hpp"
#include "threading.hpp"
#include "disassembler.hpp"

#ifndef _MSC_VER
#   include <tbb/task_arena.h>
#endif

#if defined(DEBUG) && !defined(NOTRACE)
#   define TRACEPOINT(name) this->_trace(name)
#else
#   define TRACEPOINT(name)
#endif

/* SIMD-based generator */
class rv_generator_avx : public rv_generator_st {
    decltype(steady_clock::now()) start {};
    decltype(steady_clock::now()) prev {};

    void _trace(std::string_view name) {
        auto now = steady_clock::now();
        std::chrono::nanoseconds total = now - start;
        std::chrono::nanoseconds elapsed = now - prev;

        std::cerr << name;
        
        for (size_t i = 0; i < (36 - name.size()); ++i) {
            std::cerr << " ";
        }

        std::stringstream ss;
        ss << elapsed;

        std::string elapsed_str = ss.str();



        std::cerr << rvdisasm::color::imm << elapsed << rvdisasm::color::white;

        for (size_t i = 0; i < (12 - elapsed_str.size()); ++i) {
            std::cerr << " ";
        }
        std::cerr << " total = " << rvdisasm::color::imm << total << rvdisasm::color::white << "\n";
        prev = steady_clock::now();
    }

    public:
    rv_generator_avx(const DepthTree& tree, int concurrency, int sync, bool altblock, uint32_t mininstr);

    ~rv_generator_avx() override;

    [[nodiscard]] std::span<uint32_t> get_instructions() const override;

    private:
    avx_buffer<uint32_t> instr;
    avx_buffer<uint32_t> rd_avx;
    avx_buffer<uint32_t> rs1_avx;
    avx_buffer<uint32_t> rs2_avx;
    avx_buffer<uint32_t> jt_avx;

    avx_buffer<uint32_t> used_instrs_avx;

    uint32_t threads;
    std::vector<std::thread> pool;
    std::vector<std::function<void()>> tasks;

#ifndef _MSC_VER
    tbb::task_arena limited_arena;
#endif


    int sync_mode;
    std::barrier<> sync;

    bool altblock;
    uint32_t mininstr;

    counting_barrier start_sync = 0;
    counting_barrier end_sync = 0;
    counting_barrier final_sync = 0;

    std::atomic_bool done = false;

    void (rv_generator_avx::* run_func)();

    void dump_instrs() override;

    void process(bool profile) override;

    void preprocess() override;
    void isn_cnt() override;
    void isn_gen() override;
    void optimize() override;
    void regalloc() override;
    void fix_func_tab_avx(std::span<uint32_t> instr_offsets);
    void fix_jumps() override;
    void postprocess() override;

    void scatter_regalloc_fixup(const std::span<int, 8> valid_mask, const m256i indices, const std::span<int, 8> opwords);

    void thread_func_barrier(size_t idx);
    void run_barrier();

    void thread_func_cv(size_t idx);
    void run_cv();

    template <typename Func>
    FORCE_INLINE void run_basic(Func&& body, size_t elements, size_t min_elements) {
        /* Number of elements per thread, minimum of min_elements */
        size_t per_thread = std::max(min_elements, (elements + threads - 1) / threads);

        /* Total number of threads used for this */
        size_t threads_used = (elements + per_thread - 1) / per_thread;

        for (size_t i = 0; i < threads; ++i) {
            if (i >= threads_used) {
                tasks[i] = [] {};
                continue;
            }

            tasks[i] = [this, &body, elements, per_thread, threads_used, thread_idx = i] {
                if (altblock) {
                    size_t end = std::min(elements, ((thread_idx + 1) * per_thread));
                    for (size_t i = (thread_idx * per_thread); i < end; ++i) {
                        body(i);
                    }
                } else {
                    for (size_t i = thread_idx; i < elements; i += threads_used) {
                        body(i);
                    }
                }
            };
        }

        (this->*run_func)();
    }
};
