#include <array>
#include <iostream>
#include <iomanip>
#include <cstdint>

#include "simd.hpp"

using dataset = std::array<uint32_t, 8>;

int main() {
    AVX_ALIGNED dataset input{ 11, 22, 33, 44, 55, 66, 77, 88 };

    std::cerr << std::setfill(' ') << "_mm256_slli_si256_dual:\n";

    auto testleft = [&]<uint32_t shift>(){
        AVX_ALIGNED dataset avx{};

        __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data()));
        
        x = _mm256_slli_si256_dual<shift>(x);

        _mm256_store_si256(reinterpret_cast<__m256i*>(avx.data()), x);

        std::cerr << std::setw(2) << shift << ": ";
        for (uint8_t i = 0; i < 8; ++i) {
            std::cerr << std::setw(2) << avx[i] << ' ';
        }
        std::cerr << '\n';
    };

    testleft.template operator()<0>();
    testleft.template operator()<4>();
    testleft.template operator()<8>();
    testleft.template operator()<12>();
    testleft.template operator()<16>();
    testleft.template operator()<20>();
    testleft.template operator()<24>();
    testleft.template operator()<28>();
    testleft.template operator()<32>();

    std::cerr << "_mm256_srli_si256_dual:\n";;

    auto testright = [&]<uint32_t shift>(){
        AVX_ALIGNED dataset avx{};
        
        __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(input.data()));
        
        x = _mm256_srli_si256_dual<shift>(x);

        _mm256_store_si256(reinterpret_cast<__m256i*>(avx.data()), x);

        std::cerr << std::setw(2) << shift << ": ";
        for (uint8_t i = 0; i < 8; ++i) {
            std::cerr << std::setw(2) << avx[i] << ' ';
        }
        std::cerr << '\n';
    };

    testright.template operator()<0>();
    testright.template operator()<4>();
    testright.template operator()<8>();
    testright.template operator()<12>();
    testright.template operator()<16>();
    testright.template operator()<20>();
    testright.template operator()<24>();
    testright.template operator()<28>();
    testright.template operator()<32>();
}
