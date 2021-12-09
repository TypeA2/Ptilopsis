#ifndef PTILOPSIS_HELPER_HPP
#define PTILOPSIS_HELPER_HPP

#include <type_traits>

namespace ptilopsis {
    template <typename E>
    constexpr auto to_integral(E e) {
        return static_cast<std::underlying_type_t<E>>(e);
    }

}

#endif /* PTILOPSIS_HELPER_HPP */
