#ifndef ICL_LAPACK_UTIL_HH
#define ICL_LAPACK_UTIL_HH

#include <exception>
#include <complex>
#include <ctype.h>

#include <assert.h>

#include "blas.hh"

namespace lapack {

// -----------------------------------------------------------------------------
/// Exception class for LAPACK errors.
class Error: public std::exception {
public:
    /// Constructs LAPACK error
    Error():
        std::exception()
    {}

    /// Constructs LAPACK error with message: "func: msg"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(func) + ": " + msg )
    {}

    /// Returns LAPACK error message
    virtual const char* what()
{
    return msg_.c_str();
}

private:
    std::string msg_;
};

// =============================================================================
namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by lapack_throw_if_ macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
        throw Error( condstr, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by lapack_throw_if_msg_ macro
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

} // namespace internal

// =============================================================================
// internal macro to get string #cond; throws Error if cond is true
// ex: lapack_throw_if_( a < b );
#define lapack_throw_if_( cond ) \
    internal::throw_if( cond, #cond, __func__ )

// internal macro takes cond and printf-style format for error message.
// throws Error if cond is true.
// ex: lapack_throw_if_msg_( a < b, "a %d < b %d", a, b );
#define lapack_throw_if_msg_( cond, ... ) \
    internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

// =============================================================================
// Callback logical functions of one, two, or three arguments are used
// to select eigenvalues to sort to the top left of the Schur form in gees and gges.
// The value is selected if function returns TRUE (non-zero).

typedef blas_int (*lapack_s_select2) ( float const* omega_real, float const* omega_imag );
typedef blas_int (*lapack_s_select3) ( float const* alpha_real, float const* alpha_imag, float const* beta );

typedef blas_int (*lapack_d_select2) ( double const* omega_real, double const* omega_imag );
typedef blas_int (*lapack_d_select3) ( double const* alpha_real, double const* alpha_imag, double const* beta );

typedef blas_int (*lapack_c_select1) ( std::complex<float> const* omega );
typedef blas_int (*lapack_c_select2) ( std::complex<float> const* alpha, std::complex<float> const* beta );

typedef blas_int (*lapack_z_select1) ( std::complex<double> const* omega );
typedef blas_int (*lapack_z_select2) ( std::complex<double> const* alpha, std::complex<double> const* beta );

// =============================================================================
typedef blas::Layout Layout;
typedef blas::Op Op;
typedef blas::Uplo Uplo;
typedef blas::Diag Diag;
typedef blas::Side Side;

// -----------------------------------------------------------------------------
enum class Norm {
    One = '1',  // or 'O'
    Two = '2',
    Inf = 'I',
    Fro = 'F',  // or 'E'
    Max = 'M',
};

inline char norm2char( lapack::Norm norm )
{
    return char( norm );
}

inline lapack::Norm char2norm( char norm )
{
    norm = char( toupper( norm ));
    if (norm == 'O')
        norm = '1';
    else if (norm == 'E')
        norm = 'F';
    lapack_throw_if_( norm != '1' && norm != '2' && norm != 'I' &&
                      norm != 'F' && norm != 'M' );
    return lapack::Norm( norm );
}

inline const char* norm2str( lapack::Norm norm )
{
    switch (norm) {
        case Norm::One: return "1";
        case Norm::Two: return "2";
        case Norm::Inf: return "inf";
        case Norm::Fro: return "fro";
        case Norm::Max: return "max";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// make EigJob, SVDJob?
enum class Job {
    NoVec        = 'N',  // syev, geev, gesvd, gesdd
    Vec          = 'V',  // syev, geev

    AllVec       = 'A',  // gesvd, gesdd
    SomeVec      = 'S',  // gesvd, gesdd
    OverwriteVec = 'O',  // gesvd, gesdd
};

inline char job2char( lapack::Job job )
{
    return char( job );
}

inline lapack::Job char2job( char job )
{
    job = char( toupper( job ));
    lapack_throw_if_( job != 'N' && job != 'V' && job != 'A' && job != 'S' &&
                      job != 'O' );
    return lapack::Job( job );
}

inline const char* job2str( lapack::Job job )
{
    switch (job) {
        case lapack::Job::NoVec:        return "novec";
        case lapack::Job::Vec:          return "vec";
        case lapack::Job::AllVec:       return "all";
        case lapack::Job::SomeVec:      return "some";
        case lapack::Job::OverwriteVec: return "overwrite";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// bbcsd
// todo: just generic yes/no?
enum class JobCS {
    Update      = 'Y',
    NoUpdate    = 'N',  // or any other
};

inline char jobcs2char( lapack::JobCS jobcs )
{
    return char( jobcs );
}

inline lapack::JobCS char2jobcs( char jobcs )
{
    jobcs = char( toupper( jobcs ));
    lapack_throw_if_( jobcs != 'Y' && jobcs != 'N' );
    return lapack::JobCS( jobcs );
}

inline const char* jobcs2str( lapack::JobCS jobcs )
{
    switch (jobcs) {
        case lapack::JobCS::Update:   return "yes";
        case lapack::JobCS::NoUpdate: return "no";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// ggsvd3
// todo: generic yes/no? would require special function to get lapack char
enum class JobU {
    Vec         = 'U',
    NoVec       = 'N',
};

inline char jobu2char( lapack::JobU jobu )
{
    return char( jobu );
}

inline lapack::JobU char2jobu( char jobu )
{
    jobu = char( toupper( jobu ));
    lapack_throw_if_( jobu != 'U' && jobu != 'N' );
    return lapack::JobU( jobu );
}

inline const char* jobu2str( lapack::JobU jobu )
{
    switch (jobu) {
        case lapack::JobU::Vec:   return "u-vec";
        case lapack::JobU::NoVec: return "novec";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// ggsvd3
// todo: generic yes/no?
enum class JobV {
    Vec         = 'V',
    NoVec       = 'N',
};

inline char jobv2char( lapack::JobV jobv )
{
    return char( jobv );
}

inline lapack::JobV char2jobv( char jobv )
{
    jobv = char( toupper( jobv ));
    lapack_throw_if_( jobv != 'V' && jobv != 'N' );
    return lapack::JobV( jobv );
}

inline const char* jobv2str( lapack::JobV jobv )
{
    switch (jobv) {
        case lapack::JobV::Vec:   return "v-vec";
        case lapack::JobV::NoVec: return "novec";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// ggsvd3
// todo: generic yes/no?
enum class JobQ {
    Vec         = 'Q',
    NoVec       = 'N',
};

inline char jobq2char( lapack::JobQ jobq )
{
    return char( jobq );
}

inline lapack::JobQ char2jobq( char jobq )
{
    jobq = char( toupper( jobq ));
    lapack_throw_if_( jobq != 'Q' && jobq != 'N' );
    return lapack::JobQ( jobq );
}

inline const char* jobq2str( lapack::JobQ jobq )
{
    switch (jobq) {
        case lapack::JobQ::Vec:   return "q-vec";
        case lapack::JobQ::NoVec: return "novec";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// gees
// todo: generic yes/no
enum class Sort {
    NotSorted   = 'N',
    Sorted      = 'S',
};

inline char sort2char( lapack::Sort sort )
{
    return char( sort );
}

inline lapack::Sort char2sort( char sort )
{
    sort = char( toupper( sort ));
    lapack_throw_if_( sort != 'N' && sort != 'S' );
    return lapack::Sort( sort );
}

inline const char* sort2str( lapack::Sort sort )
{
    switch (sort) {
        case lapack::Sort::NotSorted: return "not-sorted";
        case lapack::Sort::Sorted:    return "sorted";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// syevx
enum class Range {
    All         = 'A',
    Value       = 'V',
    Index       = 'I',
};

inline char range2char( lapack::Range range )
{
    return char( range );
}

inline lapack::Range char2range( char range )
{
    range = char( toupper( range ));
    lapack_throw_if_( range != 'A' && range != 'V' && range != 'I' );
    return lapack::Range( range );
}

inline const char* range2str( lapack::Range range )
{
    switch (range) {
        case lapack::Range::All:   return "all";
        case lapack::Range::Value: return "value";
        case lapack::Range::Index: return "index";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// orgbr, ormbr
enum class Vect {
    Q           = 'Q',
    P           = 'P',
};

inline char vect2char( lapack::Vect vect )
{
    return char( vect );
}

inline lapack::Vect char2vect( char vect )
{
    vect = char( toupper( vect ));
    lapack_throw_if_( vect != 'Q' && vect != 'P' );
    return lapack::Vect( vect );
}

inline const char* vect2str( lapack::Vect vect )
{
    switch (vect) {
        case lapack::Vect::P: return "p";
        case lapack::Vect::Q: return "q";
    }
    return "?";
}

// -----------------------------------------------------------------------------
enum class CompQ {
    NoVec       = 'N',  // bdsdc, gghd3
    Vec         = 'I',  // bdsdc, gghd3
    CompactVec  = 'P',  // bdsdc
    Update      = 'V',  // gghd3
};

inline char compq2char( lapack::CompQ compq )
{
    return char( compq );
}

inline lapack::CompQ char2compq( char compq )
{
    compq = char( toupper( compq ));
    lapack_throw_if_( compq != 'N' && compq != 'I' && compq != 'P' &&
                      compq != 'V' );
    return lapack::CompQ( compq );
}

inline const char* compq2str( lapack::CompQ compq )
{
    switch (compq) {
        case lapack::CompQ::NoVec:      return "novec";
        case lapack::CompQ::Vec:        return "i-vec";
        case lapack::CompQ::CompactVec: return "p-compactvec";
        case lapack::CompQ::Update:     return "vec";

    }
    return "?";
}

// -----------------------------------------------------------------------------
// larfb
enum class Direct {
    Forward     = 'F',
    Backward    = 'B',
};

inline char direct2char( lapack::Direct direct )
{
    return char( direct );
}

inline lapack::Direct char2direct( char direct )
{
    direct = char( toupper( direct ));
    lapack_throw_if_( direct != 'F' && direct != 'B' );
    return lapack::Direct( direct );
}

inline const char* direct2str( lapack::Direct direct )
{
    switch (direct) {
        case lapack::Direct::Forward:  return "forward";
        case lapack::Direct::Backward: return "backward";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// larfb
enum class StoreV {
    Columnwise  = 'C',
    Rowwise     = 'R',
};

inline char storev2char( lapack::StoreV storev )
{
    return char( storev );
}

inline lapack::StoreV char2storev( char storev )
{
    storev = char( toupper( storev ));
    lapack_throw_if_( storev != 'C' && storev != 'R' );
    return lapack::StoreV( storev );
}

inline const char* storev2str( lapack::StoreV storev )
{
    switch (storev) {
        case lapack::StoreV::Columnwise: return "columnwise";
        case lapack::StoreV::Rowwise:    return "rowwise";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// lascl, laset
enum class MatrixType {
    General     = 'G',
    Lower       = 'L',
    Upper       = 'U',
    Hessenberg  = 'H',
    LowerBand   = 'B',
    UpperBand   = 'Q',
    Band        = 'Z',
};

inline char matrixtype2char( lapack::MatrixType type )
{
    return char( type );
}

inline lapack::MatrixType char2matrixtype( char type )
{
    type = char( toupper( type ));
    lapack_throw_if_( type != 'G' && type != 'L' && type != 'U' &&
                      type != 'H' && type != 'B' && type != 'Q' && type != 'Z' );
    return lapack::MatrixType( type );
}

inline const char* matrixtype2str( lapack::MatrixType type )
{
    switch (type) {
        case lapack::MatrixType::General:    return "general";
        case lapack::MatrixType::Lower:      return "lower";
        case lapack::MatrixType::Upper:      return "upper";
        case lapack::MatrixType::Hessenberg: return "hessenberg";
        case lapack::MatrixType::LowerBand:  return "band-lower";
        case lapack::MatrixType::UpperBand:  return "q-band-upper";
        case lapack::MatrixType::Band:       return "z-band";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// todo: trevc needs BothSides
// trevc
enum class HowMany {
    All           = 'A',
    Backtransform = 'B',
    Select        = 'S',
};

inline char howmany2char( lapack::HowMany howmany )
{
    return char( howmany );
}

inline lapack::HowMany char2howmany( char howmany )
{
    howmany = char( toupper( howmany ));
    lapack_throw_if_( howmany != 'A' && howmany != 'B' && howmany != 'S' );
    return lapack::HowMany( howmany );
}

inline const char* howmany2str( lapack::HowMany howmany )
{
    switch (howmany) {
        case lapack::HowMany::All:           return "all";
        case lapack::HowMany::Backtransform: return "backtransform";
        case lapack::HowMany::Select:        return "select";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// *svx, *rfsx
enum class Equed {
    None        = 'N',
    Row         = 'R',
    Col         = 'C',
    Both        = 'B',
};

inline char equed2char( lapack::Equed equed )
{
    return char( equed );
}

inline lapack::Equed char2equed( char equed )
{
    equed = char( toupper( equed ));
    lapack_throw_if_( equed != 'N' && equed != 'R' && equed != 'C' &&
                      equed != 'B' );
    return lapack::Equed( equed );
}

inline const char* equed2str( lapack::Equed equed )
{
    switch (equed) {
        case lapack::Equed::None: return "none";
        case lapack::Equed::Row:  return "row";
        case lapack::Equed::Col:  return "col";
        case lapack::Equed::Both: return "both";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// *svx
// todo: what's good name for this?
enum class Factored {
    Factored    = 'F',
    NotFactored = 'N',
    Equilibrate = 'E',
};

inline char factored2char( lapack::Factored factored )
{
    return char( factored );
}

inline lapack::Factored char2factored( char factored )
{
    factored = char( toupper( factored ));
    lapack_throw_if_( factored != 'F' && factored != 'N' && factored != 'E' );
    return lapack::Factored( factored );
}

inline const char* factored2str( lapack::Factored factored )
{
    switch (factored) {
        case lapack::Factored::Factored:    return "factored";
        case lapack::Factored::NotFactored: return "notfactored";
        case lapack::Factored::Equilibrate: return "equilibrate";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// geesx
enum class Sense {
    None        = 'N',
    Eigenvalues = 'E',
    Subspace    = 'V',
    Both        = 'B',
};

inline char sense2char( lapack::Sense sense )
{
    return char( sense );
}

inline lapack::Sense char2sense( char sense )
{
    sense = char( toupper( sense ));
    lapack_throw_if_( sense != 'N' && sense != 'E' && sense != 'V' &&
                      sense != 'B' );
    return lapack::Sense( sense );
}

inline const char* sense2str( lapack::Sense sense )
{
    switch (sense) {
        case lapack::Sense::None:        return "none";
        case lapack::Sense::Eigenvalues: return "eigenvalues";
        case lapack::Sense::Subspace:    return "subspace";
        case lapack::Sense::Both:        return "both";
    }
    return "?";
}

// -----------------------------------------------------------------------------
enum class Balance {
    None        = 'N',
    Permute     = 'P',
    Scale       = 'S',
    Both        = 'B',
};

inline char balance2char( lapack::Balance balance )
{
    return char( balance );
}

inline lapack::Balance char2balance( char balance )
{
    balance = char( toupper( balance ));
    lapack_throw_if_( balance != 'N' && balance != 'P' && balance != 'S' &&
                      balance != 'B' );
    return lapack::Balance( balance );
}

inline const char* balance2str( lapack::Balance balance )
{
    switch (balance) {
        case lapack::Balance::None:    return "none";
        case lapack::Balance::Permute: return "permute";
        case lapack::Balance::Scale:   return "scale";
        case lapack::Balance::Both:    return "both";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// check_ortho (LAPACK testing zunt01)
enum class RowCol {
    Col = 'C',
    Row = 'R',
};

inline char rowcol2char( lapack::RowCol rowcol )
{
    return char( rowcol );
}

inline lapack::RowCol char2rowcol( char rowcol )
{
    rowcol = char( toupper( rowcol ));
    lapack_throw_if_( rowcol != 'C' && rowcol != 'R' );
    return lapack::RowCol( rowcol );
}

inline const char* rowcol2str( lapack::RowCol rowcol )
{
    switch (rowcol) {
        case lapack::RowCol::Col: return "col";
        case lapack::RowCol::Row: return "row";
    }
    return "?";
}

}  // namespace lapack

#endif  // ICL_LAPACK_UTIL_HH
