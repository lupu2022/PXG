#ifndef _PXG_COMMON_HPP_
#define _PXG_COMMON_HPP_

#include <stdlib.h>
#include <iostream>

// gloal help functions
#define pxg_assert(Expr, Msg) \
    pxg__M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define pxg_panic(Msg) \
    pxg__M_Panic(__FILE__, __LINE__, Msg)

inline void pxg__M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

inline void pxg__M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "Assert failed:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
}

#endif
