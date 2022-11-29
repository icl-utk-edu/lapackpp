// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef PRINT_HH
#define PRINT_HH

#include <assert.h>
#include <complex>

// -----------------------------------------------------------------------------
template< typename T >
void print_matrix( int64_t m, int64_t n, T *A, int64_t lda,
                   const char* format="%9.4f",
                   const char* format_int="%9.0f" )
{
    #define A(i_, j_) A[ (i_) + size_t(lda)*(j_) ]

    require( m >= 0 );
    require( n >= 0 );
    require( lda >= m );
    char format_[32], format_int_[32];
    snprintf( format_,     sizeof(format_),     " %s", format     );
    snprintf( format_int_, sizeof(format_int_), " %s", format_int );

    printf( "[\n" );
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            T re = A(i, j);
            if (re == int(re))
                printf( format_int_, re );
            else
                printf( format_, re );
        }
        printf( "\n" );
    }
    printf( "];\n" );

    #undef A
}

// -----------------------------------------------------------------------------
/// Overload for complex.
template< typename T >
void print_matrix( int64_t m, int64_t n, std::complex<T>* A, int64_t lda,
                   const char* format="%9.4f",
                   const char* format_int="%9.0f" )
{
    #define A(i_, j_) A[ (i_) + size_t(lda)*(j_) ]

    require( m >= 0 );
    require( n >= 0 );
    require( lda >= m );
    char format_[32], format_i_[32], format_int_[32], format_int_i_[32];
    snprintf( format_,       sizeof(format_),       " %s",    format     );
    snprintf( format_int_,   sizeof(format_int_),   " %s",    format_int );
    snprintf( format_i_,     sizeof(format_i_),     " + %si", format     );
    snprintf( format_int_i_, sizeof(format_int_i_), " + %si", format_int );

    printf( "[\n" );
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            T re = std::real( A(i, j) );
            if (re == int(re))
                printf( format_int_, re );
            else
                printf( format_, re );

            T im = std::imag( A(i, j) );
            if (im == int(im))
                printf( format_int_i_, im );
            else
                printf( format_i_, im );
        }
        printf( "\n" );
    }
    printf( "];\n" );

    #undef A
}

// -----------------------------------------------------------------------------
/// Overload with name.
template< typename T >
void print_matrix( const char* name,
                   int64_t m, int64_t n, T *A, int64_t lda,
                   const char* format="%9.4f",
                   const char* format_int="%9.0f" )
{
    printf( "%s = ", name );
    print_matrix( m, n, A, lda, format, format_int );
}

// -----------------------------------------------------------------------------
template< typename T >
void print_vector( int64_t n, T *x, int64_t incx,
                   const char* format="%9.4f",
                   const char* format_int="%9.0f" )
{
    require( n >= 0 );
    require( incx != 0 );
    char format_[32], format_int_[32];
    snprintf( format_,     sizeof(format_),     " %s", format     );
    snprintf( format_int_, sizeof(format_int_), " %s", format_int );

    printf( "[" );
    int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    for (int64_t i = 0; i < n; ++i) {
        T re = x[ix];
        if (re == int(re))
            printf( format_int_, re );
        else
            printf( format_, re );

        ix += incx;
    }
    printf( " ]';\n" );
}

// -----------------------------------------------------------------------------
/// Overload for complex.
template< typename T >
void print_vector( int64_t n, std::complex<T>* x, int64_t incx,
                   const char* format="%9.4f",
                   const char* format_int="%9.0f" )
{
    require( n >= 0 );
    require( incx != 0 );
    char format_[32], format_i_[32], format_int_[32], format_int_i_[32];
    snprintf( format_,       sizeof(format_),       " %s",    format     );
    snprintf( format_int_,   sizeof(format_int_),   " %s",    format_int );
    snprintf( format_i_,     sizeof(format_i_),     " + %si", format     );
    snprintf( format_int_i_, sizeof(format_int_i_), " + %si", format_int );

    printf( "[" );
    int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    for (int64_t i = 0; i < n; ++i) {
        T re = std::real( x[ix] );
        if (re == int(re))
            printf( format_int_, re );
        else
            printf( format_, re );

        T im = std::imag( x[ix] );
        if (im == int(im))
            printf( format_int_i_, im );
        else
            printf( format_i_, im );

        ix += incx;
    }
    printf( " ]';\n" );
}

// -----------------------------------------------------------------------------
/// Overload with name.
template< typename T >
void print_vector( const char* name,
                   int64_t n, T *x, int64_t incx,
                   const char* format="%9.4f",
                   const char* format_int="%9.0f" )
{
    printf( "%s = ", name );
    print_vector( n, x, incx, format, format_int );
}

#endif        //  #ifndef PRINT_HH
