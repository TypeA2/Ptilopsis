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
