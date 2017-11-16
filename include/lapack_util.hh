#ifndef ICL_LAPACK_UTIL_HH
#define ICL_LAPACK_UTIL_HH

#include <exception>
#include <complex>

#include <assert.h>

#include "blas.hh"

namespace lapack {

// -----------------------------------------------------------------------------
typedef blas::Layout Layout;
typedef blas::Op Op;
typedef blas::Uplo Uplo;
typedef blas::Diag Diag;
typedef blas::Side Side;

enum class Norm {
    One = 'O',
    Two = '2',
    Inf = 'I',
    Fro = 'F',
    Max = 'M',
};

// make EigJob, SVDJob?
enum class Job {
    NoVec       = 'N',  // syev, geev, gesvd, gesdd
    Vec         = 'V',  // syev, geev

    AllVec      = 'A',  // gesvd, gesdd
    SomeVec     = 'S',  // gesvd, gesdd
    OverwriteVec = 'O', // gesvd, gesdd
};

// bbcsd
enum class JobCS {
    Update      = 'Y',
    NoUpdate    = 'N',  // or any other
};

// ggsvd3
enum class JobU {
    Vec         = 'U',
    NoVec       = 'N',
};

// ggsvd3
enum class JobV {
    Vec         = 'V',
    NoVec       = 'N',
};

// ggsvd3
enum class JobQ {
    Vec         = 'Q',
    NoVec       = 'N',
};

// gees
enum class Sort {
    NotSorted   = 'N',
    Sorted      = 'S',
};

// syevx
enum class Range {
    All         = 'A',
    Value       = 'V',
    Index       = 'I',
};

// orgbr, ormbr
enum class Vect {
    Q           = 'Q',
    P           = 'P',
};

enum class CompQ {
    NoVec       = 'N',  // bdsdc, gghd3
    Vec         = 'I',  // bdsdc, gghd3
    CompactVec  = 'P',  // bdsdc
    Update      = 'V',  // gghd3
};

// larfb
enum class Direct {
    Forward     = 'F',
    Backward    = 'B',
};

// larfb
enum class StoreV {
    Columnwise  = 'C',
    Rowwise     = 'R',
};

// lascl
enum class MatrixType {
    General     = 'G',
    Lower       = 'L',
    Upper       = 'U',
    Hessenberg  = 'H',
    LowerBand   = 'B',
    UpperBand   = 'Q',
    Band        = 'Z',
};

// todo: trevc needs BothSides
// trevc
enum class HowMany {
    All              = 'A',
    BacktransformAll = 'B',
    Select           = 'S',
};

// *svx, *rfsx
enum class Equed {
    None        = 'N',
    Row         = 'R',
    Col         = 'C',
    Both        = 'B',
};

// *svx
// todo: what's good name for this?
enum class Factored {
    Factored    = 'F',
    NotFactored = 'N',
    Equilibrate = 'E',
};

// geesx
enum class Sense {
    None        = 'N',
    Eigenvalues = 'E',
    Subspace    = 'V',
    Both        = 'B',
};

enum class Balance {
    None        = 'N',
    Permute     = 'P',
    Scale       = 'S',
    Both        = 'B',
};

inline char    norm2char( lapack::Norm    norm    ) { return char(norm);    }
inline char    sort2char( lapack::Sort    sort    ) { return char(sort);    }
inline char   range2char( lapack::Range   range   ) { return char(range);   }
inline char    vect2char( lapack::Vect    vect    ) { return char(vect);    }
inline char   compq2char( lapack::CompQ   compq   ) { return char(compq);   }
inline char  direct2char( lapack::Direct  direct  ) { return char(direct);  }
inline char  storev2char( lapack::StoreV  storev  ) { return char(storev);  }
inline char howmany2char( lapack::HowMany howmany ) { return char(howmany); }
inline char   equed2char( lapack::Equed   equed   ) { return char(equed);   }
inline char   sense2char( lapack::Sense   sense   ) { return char(sense);   }
inline char balance2char( lapack::Balance balance ) { return char(balance); }

inline char     job2char( lapack::Job     job     ) { return char(job);     }
inline char   jobcs2char( lapack::JobCS   jobcs   ) { return char(jobcs);   }
inline char    jobu2char( lapack::JobU    jobu    ) { return char(jobu);    }
inline char    jobv2char( lapack::JobV    jobv    ) { return char(jobv);    }
inline char    jobq2char( lapack::JobQ    jobq    ) { return char(jobq);    }

inline char   factored2char( lapack::Factored   factored ) { return char(factored); }
inline char matrixtype2char( lapack::MatrixType type     ) { return char(type);     }

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
    virtual const char* what() { return msg_.c_str(); }

private:
    std::string msg_;
};

// =============================================================================
namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by throw_if_ macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
        throw Error( condstr, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by throw_if_msg_ macro
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
// ex: throw_if_( a < b );
#define throw_if_( cond ) \
    internal::throw_if( cond, #cond, __func__ )

// internal macro takes cond and printf-style format for error message.
// throws Error if cond is true.
// ex: throw_if_msg_( a < b, "a %d < b %d", a, b );
#define throw_if_msg_( cond, ... ) \
    internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

}  // namespace lapack

#endif  // ICL_LAPACK_UTIL_HH
