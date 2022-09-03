#pragma once

#include <type_traits>
#include <execution>

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
        /* _BitScanForward is an Intel intrinsic but GCC doesn't have it*/
#ifdef _MSC_VER
        /* The first 1 in the NOT'ed mask means the first zero in the original */
        unsigned long r = 0;
        bool nonzero = _BitScanForward64(&r, ~val);
        /* ~val == 0 means all registers are taken*/
        if (!nonzero) {
            return 64;
        }

        return r;
#else
        int res = __builtin_ffs(~val);
        /* Not found */
        if (res == 0) {
            return 64;
        }

        return res - 1;
#endif
    }
}