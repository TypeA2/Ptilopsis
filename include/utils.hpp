#pragma once

#include <type_traits>

inline std::ostream& operator<<(std::ostream& os, std::chrono::nanoseconds ns) {
    auto count = static_cast<long double>(ns.count());


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

namespace ptilopsis {
    /* Find the first zero bit and return the index, or 64 */
    inline uint32_t ffz(uint64_t val) {
        /* The first 1 in the NOT'ed mask means the first zero in the original */
        unsigned long r = 0;
        bool nonzero = _BitScanForward64(&r, ~val);
        /* ~fixed_regs == 0 means all registers are taken*/
        if (!nonzero) {
            return 64;
        }

        return r;
    }
}