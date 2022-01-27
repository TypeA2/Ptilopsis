#ifndef PTILOPSIS_THREADING_HPP
#define PTILOPSIS_THREADING_HPP

#include <atomic>
#include <functional>

#ifdef _MSC_VER
#   include <Windows.h>
#endif

#define USE_PAUSE

#ifdef USE_PAUSE
#   ifdef _MSC_VER
#       define MAYBE_PAUSE() YieldProcessor();
#   else
#       define MAYBE_PAUSE() __builtin_ia32_pause()
#   endif
#else
#   warning "Not using PAUSE instruction"
#   define MAYBE_PAUSE()
#endif

/* What should be a more efficient spinlock than a naive implementation
 * From: https://rigtorp.se/spinlock/
 */
class spinlock {
    std::atomic_bool flag = false;

    public:
    void lock() noexcept {
        for (;;) {
            /* Assume the lock is free, perform an expensive exchange */
            if (!flag.exchange(true, std::memory_order_acquire)) {
                /* Locked */
                return;
            }

            /* Cache-friendly wait, since multiple threads can read without cache sync */
            while (flag.load(std::memory_order_relaxed)) {
                /* Hint to hardware we're in a spinlock */
                MAYBE_PAUSE();
            }
        }
    }

    void unlock() noexcept {
        flag.store(false, std::memory_order_release);
    }
};

/* Atomic barrier based on previous spinlock */
class atomic_barrier {
    std::atomic_bool should_wait;

    public:
    atomic_barrier(bool should_wait = false) : should_wait{ should_wait } { }
    
    void wait() noexcept {
        for (;;) {
            if (!should_wait.load(std::memory_order_acquire)) {
                return;
            }

            while (should_wait.load(std::memory_order_relaxed)) {
                MAYBE_PAUSE();
            }
        }
    }

    void lock() noexcept {
        should_wait.store(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        should_wait.store(false, std::memory_order_release);
    }
};


/* Counting barrier spinlock */
class counting_barrier {
    std::atomic_size_t count;

    public:
    counting_barrier() = delete;
    counting_barrier(size_t count) : count { count } { }

    template <typename Comp = std::equal_to<>>
    void wait_for(size_t val, Comp comp = {}) const noexcept {
        for (;;) {
            if (comp(count.load(std::memory_order_acquire), val)) {
                return;
            }

            while (!comp(count.load(std::memory_order_relaxed), val)) {
                MAYBE_PAUSE();
            }
        }
    }

    counting_barrier& operator=(size_t v) noexcept {
        count = v;
        return *this;
    }

    counting_barrier& operator+=(size_t v) noexcept {
        count += v;
        return *this;
    }

    counting_barrier& operator-=(size_t v) noexcept {
        count -= v;
        return *this;
    }

    size_t value() const noexcept {
        return count.load(std::memory_order_relaxed);
    }
};

#endif /* PTILOPSIS_THREADING_HPP */
