// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_NO_CONSTRUCT_ALLOCATOR_HH
#define LAPACK_NO_CONSTRUCT_ALLOCATOR_HH

#include <cstddef>  // std::size_t
#include <limits>   // std::numeric_limits
#include <new>      // std::bad_alloc, std::bad_array_new_length
#include <vector>   // std::vector
#if defined( _WIN32 ) || defined( _WIN64 )
#   include <malloc.h>  // _aligned_malloc, _aligned_free
#else
#   include <stdlib.h>  // posix_memalign, free
#endif

namespace lapack {

// No-construct allocator type which allocates / deallocates.
template <typename T>
struct NoConstructAllocator
{
    using value_type = T;

    NoConstructAllocator() = default;

    // Construction given an allocated pointer is a null-op.
    //
    // @tparam Args Parameter pack which handles all possible calling
    // signatures of construct outlined in the Allocator concept.
    //
    template <typename... Args>
    void construct( T* ptr, Args&& ... args ) { }

    // Destruction of an object in allocated memory is a null-op
    void destroy( T* ptr ) { }

    T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        void* memPtr = nullptr;
        #if defined( _WIN32 ) || defined( _WIN64 )
            memPtr = _aligned_malloc( n*sizeof(T), 64 );
            if (memPtr != nullptr) {
                auto p = static_cast<T*>(memPtr);
                return p;
            }
        #else
            int err = posix_memalign( &memPtr, 64, n*sizeof(T) );
            if (err == 0) {
                auto p = static_cast<T*>(memPtr);
                return p;
            }
        #endif

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t n) noexcept
    {
        #if defined( _WIN32 ) || defined( _WIN64 )
            _aligned_free( p );
        #else
            free( p );
        #endif
    }
};

template <class T, class U>
bool operator == ( NoConstructAllocator<T> const& a,
                   NoConstructAllocator<U> const& b )
{
    return true;
}

template <class T, class U>
bool operator != ( NoConstructAllocator<T> const& a,
                   NoConstructAllocator<U> const& b)
{
    return false;
}

template <typename T>
using vector = std::vector< T, NoConstructAllocator<T> >;

}  // namespace lapack

#endif  // LAPACK_NO_CONSTRUCT_ALLOCATOR_HH
