#ifndef PTILOPSIS_HELPER_HPP
#define PTILOPSIS_HELPER_HPP

#include <type_traits>

namespace ptilopsis {
    template <typename E>
    constexpr auto to_integral(E e) {
        return static_cast<std::underlying_type_t<E>>(e);
    }

}

inline std::ostream& operator<<(std::ostream& os, std::chrono::nanoseconds ns) {
    auto count = ns.count();
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

#endif /* PTILOPSIS_HELPER_HPP */
