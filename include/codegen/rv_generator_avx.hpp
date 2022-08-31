#pragma once

#include <array>
#include <thread>
#include <barrier>
#include <functional>

#include "codegen/rv_generator.hpp"
#include "threading.hpp"

/* SIMD-based generator */
class rv_generator_avx : public rv_generator_st {
    public:
    rv_generator_avx(const DepthTree& tree, int concurrency, int sync, bool altblock);

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

    int sync_mode;
    bool altblock;

    std::barrier<> sync;

    counting_barrier start_sync = 0;
    counting_barrier end_sync = 0;
    counting_barrier final_sync = 0;

    std::atomic_bool done = false;

    void (rv_generator_avx::* run_func)();

    void dump_instrs() override;

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
};
