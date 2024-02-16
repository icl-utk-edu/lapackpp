// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_UTIL_HH
#define LAPACK_UTIL_HH

#include <exception>
#include <complex>
#include <ctype.h>

#include <assert.h>

#include "blas.hh"

// from config, we need only lapack_logical for callback functions lapack_*_select*
#include "lapack/config.h"

namespace lapack {

// -----------------------------------------------------------------------------
/// Exception class for LAPACK errors.
class Error: public std::exception {
public:
    /// Constructs LAPACK error
    Error():
        std::exception()
    {}

    /// Constructs BLAS error with message
    Error( std::string const& msg ):
        std::exception(),
        msg_( msg )
    {}

    /// Constructs LAPACK error with message: "func: msg"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// Returns LAPACK error message
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
};

// =============================================================================
namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by blas_error_if macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
        throw Error( condstr, func );
    }
}

#if defined(_MSC_VER)
    #define LAPACKPP_ATTR_FORMAT(I, F)
#else
    #define LAPACKPP_ATTR_FORMAT(I, F) __attribute__((format( printf, I, F )))
#endif

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by lapack_error_if_msg macro
// condstr is ignored, but differentiates this from other version.
inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
    LAPACKPP_ATTR_FORMAT(4, 5);

inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );
        throw Error( buf, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; aborts if cond is true
// uses printf-style format for error message
// called by lapack_error_if_msg macro
inline void abort_if( bool cond, const char* func,  const char* format, ... )
    LAPACKPP_ATTR_FORMAT(3, 4);

inline void abort_if( bool cond, const char* func,  const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );

        fprintf( stderr, "Error: %s, in function %s\n", buf, func );
        abort();
    }
}

#undef LAPACKPP_ATTR_FORMAT

} // namespace internal

// -----------------------------------------------------------------------------
// internal macros to handle error checks
#if defined(LAPACK_ERROR_NDEBUG) || (defined(LAPACK_ERROR_ASSERT) && defined(NDEBUG))

    // lapackpp does no error checking;
    // lower level LAPACK may still handle errors via xerbla
    #define lapack_error_if( cond ) \
        ((void)0)

    #define lapack_error_if_msg( cond, ... ) \
        ((void)0)

#elif defined(LAPACK_ERROR_ASSERT)

    // lapackpp aborts on error
    #define lapack_error_if( cond ) \
        lapack::internal::abort_if( cond, __func__, "%s", #cond )

    #define lapack_error_if_msg( cond, ... ) \
        lapack::internal::abort_if( cond, __func__, __VA_ARGS__ )

#else

    // lapackpp throws errors (default)
    // internal macro to get string #cond; throws Error if cond is true
    // ex: lapack_error_if( a < b );
    #define lapack_error_if( cond ) \
        lapack::internal::throw_if( cond, #cond, __func__ )

    // internal macro takes cond and printf-style format for error message.
    // throws Error if cond is true.
    // ex: lapack_error_if_msg( a < b, "a %d < b %d", a, b );
    #define lapack_error_if_msg( cond, ... ) \
        lapack::internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

#endif

// =============================================================================
// Callback logical functions of one, two, or three arguments are used
// to select eigenvalues to sort to the top left of the Schur form in gees and gges.
// The value is selected if function returns TRUE (non-zero).

typedef lapack_logical (*lapack_s_select2) ( float const* omega_real, float const* omega_imag );
typedef lapack_logical (*lapack_s_select3) ( float const* alpha_real, float const* alpha_imag, float const* beta );

typedef lapack_logical (*lapack_d_select2) ( double const* omega_real, double const* omega_imag );
typedef lapack_logical (*lapack_d_select3) ( double const* alpha_real, double const* alpha_imag, double const* beta );

typedef lapack_logical (*lapack_c_select1) ( std::complex<float> const* omega );
typedef lapack_logical (*lapack_c_select2) ( std::complex<float> const* alpha, std::complex<float> const* beta );

typedef lapack_logical (*lapack_z_select1) ( std::complex<double> const* omega );
typedef lapack_logical (*lapack_z_select2) ( std::complex<double> const* alpha, std::complex<double> const* beta );

// =============================================================================
using blas::Layout;
using blas::Op;
using blas::Uplo;
using blas::Diag;
using blas::Side;

// -----------------------------------------------------------------------------
// like blas::side, but adds Both for trevc
enum class Sides : char {
    Left  = 'L',  L = 'L',
    Right = 'R',  R = 'R',
    Both  = 'B',  B = 'B',
};

extern const char* Sides_help;

inline char to_char( Sides value )
{
    return char( value );
}

inline const char* to_c_string( Sides value )
{
    switch (value) {
        case Sides::Left:  return "left";
        case Sides::Right: return "right";
        case Sides::Both:  return "both";
    }
    return "?";
}

inline std::string to_string( Sides value )
{
    return to_c_string( value );
}

inline Sides from_string( std::string const& str, Sides dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "l" || str_ == "left")
        return Sides::Left;
    else if (str_ == "r" || str_ == "right")
        return Sides::Right;
    else if (str_ == "b" || str_ == "both")
        return Sides::Both;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char sides2char( Sides value )
{
    return char( value );
}

[[deprecated("use to_string or to_c_string. To be removed 2025-02.")]]
inline const char* sides2str( Sides value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Sides char2sides( char ch )
{
    return from_string( std::string( 1, ch ), Sides() );
}

// -----------------------------------------------------------------------------
enum class Norm : char {
    One = '1',  // or 'O'
    Two = '2',
    Inf = 'I',
    Fro = 'F',  // or 'E'
    Max = 'M',
};

extern const char* Norm_help;

inline char to_char( Norm value )
{
    return char( value );
}

inline const char* to_c_string( Norm value )
{
    switch (value) {
        case Norm::One: return "1";
        case Norm::Two: return "2";
        case Norm::Inf: return "inf";
        case Norm::Fro: return "fro";
        case Norm::Max: return "max";
    }
    return "?";
}

inline std::string to_string( Norm value )
{
    return to_c_string( value );
}

inline Norm from_string( std::string const& str, Norm dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "1" || str_ == "o" || str_ == "one")
        return Norm::One;
    else if (str_ == "2" || str_ == "two")
        return Norm::Two;
    else if (str_ == "i" || str_ == "inf")
        return Norm::Inf;
    else if (str_ == "f" || str_ == "fro")
        return Norm::Fro;
    else if (str_ == "m" || str_ == "max")
        return Norm::Max;
    else
        throw Error( "unknown Norm: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char norm2char( Norm value )
{
    return char( value );
}

[[deprecated("use to_string or to_c_string. To be removed 2025-02.")]]
inline const char* norm2str( Norm value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Norm char2norm( char ch )
{
    return from_string( std::string( 1, ch ), Norm() );
}

// -----------------------------------------------------------------------------
// Job for computing eigenvectors and singular vectors
// # needs custom map
enum class Job : char {
    NoVec        = 'N',
    Vec          = 'V',  // geev, syev, ...
    UpdateVec    = 'U',  // gghrd#, hbtrd, hgeqz#, hseqr#, ... (many compq or compz)

    AllVec       = 'A',  // gesvd, gesdd, gejsv#
    SomeVec      = 'S',  // gesvd, gesdd, gejsv#, gesvj#
    OverwriteVec = 'O',  // gesvd, gesdd

    CompactVec   = 'P',  // bdsdc
    SomeVecTol   = 'C',  // gesvj
    VecJacobi    = 'J',  // gejsv
    Workspace    = 'W',  // gejsv
};

extern const char* Job_help;
extern const char* Job_eig_help;
extern const char* Job_eig_left_help;
extern const char* Job_eig_right_help;
extern const char* Job_svd_left_help;
extern const char* Job_svd_right_help;

inline char to_char( Job value )
{
    return char( value );
}

// custom maps
// bbcsd, orcsd2by1
inline char to_char_csd( Job value )
{
    switch (value) {
        case Job::Vec:          return 'Y';  // orcsd
        case Job::UpdateVec:    return 'Y';  // bbcsd
        default: return char( value );
    }
}

// bdsdc, gghrd, hgeqz, hseqr, pteqr, stedc, steqr, tgsja, trexc, trsen
inline char to_char_comp( Job value )
{
    switch (value) {
        case Job::Vec:          return 'I';
        case Job::UpdateVec:    return 'V';
        default: return char( value );
    }
}

// tgsja
inline char to_char_compu( Job value )
{
    switch (value) {
        case Job::Vec:          return 'I';
        case Job::UpdateVec:    return 'U';
        default: return char( value );
    }
}

// tgsja
inline char to_char_compq( Job value )
{
    switch (value) {
        case Job::Vec:          return 'I';
        case Job::UpdateVec:    return 'Q';
        default: return char( value );
    }
}

// ggsvd3, ggsvp3
inline char to_char_jobu( Job value )
{
    switch (value) {
        case Job::Vec:          return 'U';
        default: return char( value );
    }
}

// ggsvd3, ggsvp3
inline char to_char_jobq( Job value )
{
    switch (value) {
        case Job::Vec:          return 'Q';
        default: return char( value );
    }
}

// gejsv
inline char to_char_gejsv( Job value )
{
    switch (value) {
        case Job::SomeVec:      return 'U';
        case Job::AllVec:       return 'F';
        default: return char( value );
    }
}

// gesvj
inline char to_char_gesvj( Job value )
{
    switch (value) {
        case Job::SomeVec:      return 'U';  // jobu
        case Job::SomeVecTol:   return 'C';  // jobu
        case Job::UpdateVec:    return 'U';  // jobv
        default: return char( value );
    }
}

inline const char* to_c_string( Job value )
{
    switch (value) {
        case Job::NoVec:        return "novec";
        case Job::Vec:          return "vec";
        case Job::UpdateVec:    return "update";

        case Job::AllVec:       return "all";
        case Job::SomeVec:      return "some";
        case Job::OverwriteVec: return "overwrite";

        case Job::CompactVec:   return "compact";
        case Job::SomeVecTol:   return "sometol";
        case Job::VecJacobi:    return "jacobi";
        case Job::Workspace:    return "work";
    }
    return "?";
}

inline std::string to_string( Job value )
{
    return to_c_string( value );
}

inline Job from_string( std::string const& str, Job dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "n" || str_ == "novec")
        return Job::NoVec;
    else if (str_ == "v" || str_ == "vec")
        return Job::Vec;
    else if (str_ == "u" || str_ == "update" || str_ == "updatevec")
        return Job::UpdateVec;

    else if (str_ == "a" || str_ == "all" || str_ == "allvec")
        return Job::AllVec;
    else if (str_ == "s" || str_ == "some" || str_ == "somevec")
        return Job::SomeVec;
    else if (str_ == "o" || str_ == "overwrite" || str_ == "overwritevec")
        return Job::OverwriteVec;

    else if (str_ == "p" || str_ == "compact" || str_ == "compactvec")
        return Job::CompactVec;
    else if (str_ == "c" || str_ == "somevectol")
        return Job::SomeVecTol;
    else if (str_ == "j" || str_ == "jacobi")
        return Job::VecJacobi;
    else if (str_ == "w" || str_ == "workspace")
        return Job::Workspace;
    else
        throw Error( "unknown Job: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char job2char       ( Job value ) { return to_char      ( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char job_csd2char   ( Job value ) { return to_char_csd  ( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char job_comp2char  ( Job value ) { return to_char_comp ( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char job_compu2char ( Job value ) { return to_char_compu( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char job_compq2char ( Job value ) { return to_char_compq( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char jobu2char      ( Job value ) { return to_char_jobu ( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char jobq2char      ( Job value ) { return to_char_jobq ( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char jobu_gejsv2char( Job value ) { return to_char_gejsv( value ); }

[[deprecated("use to_char. To be removed 2025-02.")]]
inline char job_gesvj2char ( Job value ) { return to_char_gesvj( value ); }

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* job2str( Job value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Job char2job( char ch )
{
    return from_string( std::string( 1, ch ), Job() );
}

// -----------------------------------------------------------------------------
// hseqr
enum class JobSchur : char {
    Eigenvalues  = 'E',
    Schur        = 'S',
};

extern const char* JobSchur_help;

inline char to_char( JobSchur value )
{
    return char( value );
}

inline const char* to_c_string( JobSchur value )
{
    switch (value) {
        case JobSchur::Eigenvalues: return "eigval";
        case JobSchur::Schur:       return "schur";
    }
    return "?";
}

inline std::string to_string( JobSchur value )
{
    return to_c_string( value );
}

inline JobSchur from_string( std::string const& str, JobSchur dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "e" || str_ == "eigval")
        return JobSchur::Eigenvalues;
    else if (str_ == "s" || str_ == "schur")
        return JobSchur::Schur;
    else
        throw Error( "unknown JobSchur: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char jobschur2char( JobSchur value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* jobschur2str( JobSchur value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline JobSchur char2jobschur( char ch )
{
    return from_string( std::string( 1, ch ), JobSchur() );
}

// -----------------------------------------------------------------------------
// gees
// todo: generic yes/no
enum class Sort : char {
    NotSorted   = 'N',
    Sorted      = 'S',
};

extern const char* Sort_help;

//--------------------
inline char to_char( Sort value )
{
    return char( value );
}

inline const char* to_c_string( Sort value )
{
    switch (value) {
        case Sort::NotSorted: return "not-sorted";
        case Sort::Sorted:    return "sorted";
    }
    return "?";
}

inline std::string to_string( Sort value )
{
    return to_c_string( value );
}

inline Sort from_string( std::string const& str, Sort dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "n" || str_ == "not-sorted")
        return Sort::NotSorted;
    else if (str_ == "s" || str_ == "sorted")
        return Sort::Sorted;
    else
        throw Error( "unknown Sort: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char sort2char( Sort sort )
{
    return char( sort );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* sort2str( Sort value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Sort char2sort( char ch )
{
    return from_string( std::string( 1, ch ), Sort() );
}

// -----------------------------------------------------------------------------
// syevx
enum class Range : char {
    All         = 'A',
    Value       = 'V',
    Index       = 'I',
};

extern const char* Range_help;

//--------------------
inline char to_char( Range value )
{
    return char( value );
}

inline const char* to_c_string( Range value )
{
    switch (value) {
        case Range::All:   return "all";
        case Range::Value: return "value";
        case Range::Index: return "index";
    }
    return "?";
}

inline std::string to_string( Range value )
{
    return to_c_string( value );
}

inline Range from_string( std::string const& str, Range dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "a" || str_ == "all")
        return Range::All;
    else if (str_ == "v" || str_ == "value")
        return Range::Value;
    else if (str_ == "i" || str_ == "index")
        return Range::Index;
    else
        throw Error( "unknown Range: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char range2char( Range range )
{
    return char( range );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* range2str( Range value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Range char2range( char ch )
{
    return from_string( std::string( 1, ch ), Range() );
}

// -----------------------------------------------------------------------------
enum class Vect : char {
    Q           = 'Q',  // orgbr, ormbr
    P           = 'P',  // orgbr, ormbr
    None        = 'N',  // orgbr, ormbr, gbbrd
    Both        = 'B',  // orgbr, ormbr, gbbrd
};

extern const char* Vect_help;

//--------------------
inline char to_char( Vect value )
{
    return char( value );
}

inline const char* to_c_string( Vect value )
{
    switch (value) {
        case Vect::P:    return "p";
        case Vect::Q:    return "q";
        case Vect::None: return "none";
        case Vect::Both: return "both";
    }
    return "?";
}

inline std::string to_string( Vect value )
{
    return to_c_string( value );
}

inline Vect from_string( std::string const& str, Vect dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "q")
        return Vect::Q;
    else if (str_ == "p")
        return Vect::P;
    else if (str_ == "n" || str_ == "none")
        return Vect::None;
    else if (str_ == "b" || str_ == "both")
        return Vect::Both;
    else
        throw Error( "unknown Vect: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char vect2char( Vect value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* vect2str( Vect value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Vect char2vect( char ch )
{
    return from_string( std::string( 1, ch ), Vect() );
}

// -----------------------------------------------------------------------------
// larfb
enum class Direction : char {
    Forward     = 'F',
    Backward    = 'B',
};

extern const char* Direction_help;

//--------------------
inline char to_char( Direction value )
{
    return char( value );
}

inline const char* to_c_string( Direction value )
{
    switch (value) {
        case Direction::Forward:  return "forward";
        case Direction::Backward: return "backward";
    }
    return "?";
}

inline std::string to_string( Direction value )
{
    return to_c_string( value );
}

inline Direction from_string( std::string const& str, Direction dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "f" || str_ == "forward")
        return Direction::Forward;
    else if (str_ == "b" || str_ == "backward")
        return Direction::Backward;
    else
        throw Error( "unknown Direction: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char direction2char( Direction value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* direction2str( Direction value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Direction char2direction( char ch )
{
    return from_string( std::string( 1, ch ), Direction() );
}

// -----------------------------------------------------------------------------
// larfb
enum class StoreV : char {
    Columnwise  = 'C',
    Rowwise     = 'R',
};

extern const char* StoreV_help;

//--------------------
inline char to_char( StoreV value )
{
    return char( value );
}

inline const char* to_c_string( StoreV value )
{
    switch (value) {
        case StoreV::Columnwise: return "colwise";
        case StoreV::Rowwise:    return "rowwise";
    }
    return "?";
}

inline std::string to_string( StoreV value )
{
    return to_c_string( value );
}

inline StoreV from_string( std::string const& str, StoreV dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "c" || str_ == "col" || str_ == "columnwise")
        return StoreV::Columnwise;
    else if (str_ == "r" || str_ == "row" || str_ == "rowwise")
        return StoreV::Rowwise;
    else
        throw Error( "unknown StoreV: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char storev2char( StoreV value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* storev2str( StoreV value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline StoreV char2storev( char ch )
{
    return from_string( std::string( 1, ch ), StoreV() );
}

// -----------------------------------------------------------------------------
// lascl, laset
enum class MatrixType : char {
    General     = 'G',
    Lower       = 'L',
    Upper       = 'U',
    Hessenberg  = 'H',
    LowerBand   = 'B',
    UpperBand   = 'Q',
    Band        = 'Z',
};

extern const char* MatrixType_help;

//--------------------
inline char to_char( MatrixType value )
{
    return char( value );
}

inline const char* to_c_string( MatrixType value )
{
    switch (value) {
        case MatrixType::General:    return "general";
        case MatrixType::Lower:      return "lower";
        case MatrixType::Upper:      return "upper";
        case MatrixType::Hessenberg: return "hessenberg";
        case MatrixType::LowerBand:  return "band-lower";
        case MatrixType::UpperBand:  return "q-band-upper";
        case MatrixType::Band:       return "z-band";
    }
    return "?";
}

inline std::string to_string( MatrixType value )
{
    return to_c_string( value );
}

inline MatrixType from_string( std::string const& str, MatrixType dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "g" || str_ == "general")
        return MatrixType::General;
    else if (str_ == "l" || str_ == "lower")
        return MatrixType::Lower;
    else if (str_ == "u" || str_ == "upper")
        return MatrixType::Upper;
    else if (str_ == "h" || str_ == "hessenberg")
        return MatrixType::Hessenberg;
    else if (str_ == "b" || str_ == "band-lower")
        return MatrixType::LowerBand;
    else if (str_ == "q" || str_ == "q-band-upper")
        return MatrixType::UpperBand;
    else if (str_ == "z" || str_ == "z-band")
        return MatrixType::Band;
    else
        throw Error( "unknown MatrixType: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char matrixtype2char( MatrixType value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* matrixtype2str( MatrixType value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline MatrixType char2matrixtype( char ch )
{
    return from_string( std::string( 1, ch ), MatrixType() );
}

// -----------------------------------------------------------------------------
// trevc
enum class HowMany : char {
    All           = 'A',
    Backtransform = 'B',
    Select        = 'S',
};

extern const char* HowMany_help;

//--------------------
inline char to_char( HowMany value )
{
    return char( value );
}

inline const char* to_c_string( HowMany value )
{
    switch (value) {
        case HowMany::All:           return "all";
        case HowMany::Backtransform: return "backtransform";
        case HowMany::Select:        return "select";
    }
    return "?";
}

inline std::string to_string( HowMany value )
{
    return to_c_string( value );
}

inline HowMany from_string( std::string const& str, HowMany dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "a" || str_ == "all")
        return HowMany::All;
    else if (str_ == "b" || str_ == "backtransform")
        return HowMany::Backtransform;
    else if (str_ == "s" || str_ == "select")
        return HowMany::Select;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char howmany2char( HowMany value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* howmany2str( HowMany value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline HowMany char2howmany( char ch )
{
    return from_string( std::string( 1, ch ), HowMany() );
}

// -----------------------------------------------------------------------------
// *svx, *rfsx
enum class Equed : char {
    None        = 'N',
    Row         = 'R',
    Col         = 'C',
    Both        = 'B',
    Yes         = 'Y',  // porfsx
};

extern const char* Equed_help;

//--------------------
inline char to_char( Equed value )
{
    return char( value );
}

inline const char* to_c_string( Equed value )
{
    switch (value) {
        case Equed::None: return "none";
        case Equed::Row:  return "row";
        case Equed::Col:  return "col";
        case Equed::Both: return "both";
        case Equed::Yes:  return "yes";
    }
    return "?";
}

inline std::string to_string( Equed value )
{
    return to_c_string( value );
}

inline Equed from_string( std::string const& str, Equed dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "n" || str_ == "none")
        return Equed::None;
    else if (str_ == "r" || str_ == "row")
        return Equed::Row;
    else if (str_ == "c" || str_ == "col")
        return Equed::Col;
    else if (str_ == "b" || str_ == "both")
        return Equed::Both;
    else if (str_ == "y" || str_ == "yes")
        return Equed::Yes;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char equed2char( Equed value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* equed2str( Equed value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Equed char2equed( char ch )
{
    return from_string( std::string( 1, ch ), Equed() );
}

// -----------------------------------------------------------------------------
// *svx
// todo: what's good name for this?
enum class Factored : char {
    Factored    = 'F',
    NotFactored = 'N',
    Equilibrate = 'E',
};

extern const char* Factored_help;

//--------------------
inline char to_char( Factored value )
{
    return char( value );
}

inline const char* to_c_string( Factored value )
{
    switch (value) {
        case Factored::Factored:    return "factored";
        case Factored::NotFactored: return "notfactored";
        case Factored::Equilibrate: return "equilibrate";
    }
    return "?";
}

inline std::string to_string( Factored value )
{
    return to_c_string( value );
}

inline Factored from_string( std::string const& str, Factored dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "f" || str_ == "factored")
        return Factored::Factored;
    else if (str_ == "n" || str_ == "notfactored")
        return Factored::NotFactored;
    else if (str_ == "e" || str_ == "equilibrate")
        return Factored::Equilibrate;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char factored2char( Factored value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* factored2str( Factored value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Factored char2factored( char ch )
{
    return from_string( std::string( 1, ch ), Factored() );
}

// -----------------------------------------------------------------------------
// geesx, trsen
enum class Sense : char {
    None        = 'N',
    Eigenvalues = 'E',
    Subspace    = 'V',
    Both        = 'B',
};

extern const char* Sense_help;

//--------------------
inline char to_char( Sense value )
{
    return char( value );
}

inline const char* to_c_string( Sense value )
{
    switch (value) {
        case Sense::None:        return "none";
        case Sense::Eigenvalues: return "eigval";
        case Sense::Subspace:    return "subspace";
        case Sense::Both:        return "both";
    }
    return "?";
}

inline std::string to_string( Sense value )
{
    return to_c_string( value );
}

inline Sense from_string( std::string const& str, Sense dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "n" || str_ == "none")
        return Sense::None;
    else if (str_ == "e" || str_ == "eigval")
        return Sense::Eigenvalues;
    else if (str_ == "s" || str_ == "subspace")
        return Sense::Subspace;
    else if (str_ == "b" || str_ == "both")
        return Sense::Both;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char sense2char( Sense value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* sense2str( Sense value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Sense char2sense( char ch )
{
    return from_string( std::string( 1, ch ), Sense() );
}

// -----------------------------------------------------------------------------
// disna
enum class JobCond : char {
    EigenVec         = 'E',
    LeftSingularVec  = 'L',
    RightSingularVec = 'R',
};

extern const char* JobCond_help;

//--------------------
inline char to_char( JobCond value )
{
    return char( value );
}

inline const char* to_c_string( JobCond value )
{
    switch (value) {
        case JobCond::EigenVec:         return "eigvec";
        case JobCond::LeftSingularVec:  return "left";
        case JobCond::RightSingularVec: return "right";
    }
    return "?";
}

inline std::string to_string( JobCond value )
{
    return to_c_string( value );
}

inline JobCond from_string( std::string const& str, JobCond dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "e" || str_ == "eigvec")
        return JobCond::EigenVec;
    else if (str_ == "l" || str_ == "left")
        return JobCond::LeftSingularVec;
    else if (str_ == "r" || str_ == "right")
        return JobCond::RightSingularVec;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char jobcond2char( JobCond value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* jobcond2str( JobCond value )
{
    return to_c_string( value );
}

inline JobCond char2jobcond( char ch )
{
    return from_string( std::string( 1, ch ), JobCond() );
}

// -----------------------------------------------------------------------------
// {ge,gg}{bak,bal}
enum class Balance : char {
    None        = 'N',
    Permute     = 'P',
    Scale       = 'S',
    Both        = 'B',
};

extern const char* Balance_help;

//--------------------
inline char to_char( Balance value )
{
    return char( value );
}

inline const char* to_c_string( Balance value )
{
    switch (value) {
        case Balance::None:    return "none";
        case Balance::Permute: return "permute";
        case Balance::Scale:   return "scale";
        case Balance::Both:    return "both";
    }
    return "?";
}

inline std::string to_string( Balance value )
{
    return to_c_string( value );
}

inline Balance from_string( std::string const& str, Balance dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "n" || str_ == "none")
        return Balance::None;
    else if (str_ == "p" || str_ == "permute")
        return Balance::Permute;
    else if (str_ == "s" || str_ == "scale")
        return Balance::Scale;
    else if (str_ == "b" || str_ == "both")
        return Balance::Both;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char balance2char( Balance value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* balance2str( Balance value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Balance char2balance( char ch )
{
    return from_string( std::string( 1, ch ), Balance() );
}

// -----------------------------------------------------------------------------
// stebz, larrd, stein docs
enum class Order : char {
    Block       = 'B',
    Entire      = 'E',
};

extern const char* Order_help;

//--------------------
inline char to_char( Order value )
{
    return char( value );
}

inline const char* to_c_string( Order value )
{
    switch (value) {
        case Order::Block:  return "block";
        case Order::Entire: return "entire";
    }
    return "?";
}

inline std::string to_string( Order value )
{
    return to_c_string( value );
}

inline Order from_string( std::string const& str, Order dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "b" || str_ == "block")
        return Order::Block;
    else if (str_ == "e" || str_ == "entire")
        return Order::Entire;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char order2char( Order value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* order2str( Order value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline Order char2order( char ch )
{
    return from_string( std::string( 1, ch ), Order() );
}

// -----------------------------------------------------------------------------
// check_ortho (LAPACK testing zunt01)
enum class RowCol : char {
    Col = 'C',
    Row = 'R',
};

extern const char* RowCol_help;

//--------------------
inline char to_char( RowCol value )
{
    return char( value );
}

inline const char* to_c_string( RowCol value )
{
    switch (value) {
        case RowCol::Col: return "col";
        case RowCol::Row: return "row";
    }
    return "?";
}

inline std::string to_string( RowCol value )
{
    return to_c_string( value );
}

inline RowCol from_string( std::string const& str, RowCol dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "c" || str_ == "col")
        return RowCol::Col;
    else if (str_ == "r" || str_ == "row")
        return RowCol::Row;
    else
        throw Error( "unknown Sides: " + str );
}

//--------------------
[[deprecated("use to_char. To be removed 2025-02.")]]
inline char rowcol2char( RowCol value )
{
    return char( value );
}

[[deprecated("use to_c_string or to_string. To be removed 2025-02.")]]
inline const char* rowcol2str( RowCol value )
{
    return to_c_string( value );
}

[[deprecated("use from_string. To be removed 2025-02.")]]
inline RowCol char2rowcol( char ch )
{
    return from_string( std::string( 1, ch ), RowCol() );
}

//------------------------------------------------------------------------------
// For %lld printf-style printing, cast to llong; guaranteed >= 64 bits.
using llong = long long;

}  // namespace lapack

#endif  // LAPACK_UTIL_HH
