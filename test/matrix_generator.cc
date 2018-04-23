#include <exception>
#include <string>
#include <vector>
#include <limits>
#include <complex>

#include "matrix_params.hh"
#include "matrix_generator.hh"

// -----------------------------------------------------------------------------
// ANSI color codes
const char *ansi_esc    = "\x1b[";
const char *ansi_red    = "\x1b[31m";
const char *ansi_bold   = "\x1b[1m";
const char *ansi_normal = "\x1b[0m";

// -----------------------------------------------------------------------------
template< typename scalar_t >
inline scalar_t rand( scalar_t max_ )
{
    return max_ * rand() / scalar_t(RAND_MAX);
}

// -----------------------------------------------------------------------------
/// Splits a string by any of the delimiters.
/// Adjacent delimiters will give empty tokens.
/// See https://stackoverflow.com/questions/53849
std::vector< std::string >
    split( const std::string& str, const std::string& delims )
{
    size_t npos = std::string::npos;
    std::vector< std::string > tokens;
    size_t start = (str.size() > 0 ? 0 : npos);
    while (start != npos) {
        size_t end = str.find_first_of( delims, start );
        tokens.push_back( str.substr( start, end - start ));
        start = (end == npos ? npos : end + 1);
    }
    return tokens;
}

// =============================================================================
namespace lapack {

// -----------------------------------------------------------------------------
template< typename scalar_t >
void generate_sigma(
    MatrixParams& params,
    Dist dist, bool rand_sign,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    Vector< blas::real_type<scalar_t> >& sigma,
    Matrix<scalar_t>& A )
{
    using real_t = blas::real_type<scalar_t>;

    // constants
    const scalar_t c_zero = 0;

    // locals
    int64_t minmn = std::min( A.m, A.n );
    assert( minmn == sigma.n );

    switch (dist) {
        case Dist::arith:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - i / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::rarith:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - (minmn - 1 - i) / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::geo:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -i / real_t(minmn - 1) );
            }
            break;

        case Dist::rgeo:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -(minmn - 1 - i) / real_t(minmn - 1) );
            }
            break;

        case Dist::cluster:
            sigma[0] = 1;
            for (int64_t i = 1; i < minmn; ++i) {
                sigma[i] = 1/cond;
            }
            break;

        case Dist::rcluster:
            for (int64_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1/cond;
            }
            sigma[minmn-1] = 1;
            break;

        case Dist::cluster2:
            for (int64_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1;
            }
            sigma[minmn-1] = 1/cond;
            break;

        case Dist::rcluster2:
            sigma[0] = 1/cond;
            for (int64_t i = 1; i < minmn; ++i) {
                sigma[i] = 1;
            }
            break;

        case Dist::logrand: {
            real_t range = log( 1/cond );
            lapack::larnv( idist_rand, params.iseed, sigma.n, sigma(0) );
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = exp( sigma[i] * range );
            }
            // make cond exact
            if (minmn >= 2) {
                sigma[0] = 1;
                sigma[1] = 1/cond;
            }
            break;
        }

        case Dist::randn:
        case Dist::randu:
        case Dist::rand: {
            int64_t idist = (int64_t) dist;
            lapack::larnv( idist, params.iseed, sigma.n, sigma(0) );
            break;
        }

        case Dist::specified:
            // user-specified sigma values; don't modify
            sigma_max = 1;
            rand_sign = false;
            break;

        case Dist::none:
            assert( false );
            break;
    }

    if (sigma_max != 1) {
        blas::scal( sigma.n, sigma_max, sigma(0), 1 );
    }

    if (rand_sign) {
        // apply random signs
        for (int64_t i = 0; i < minmn; ++i) {
            if (rand() > RAND_MAX/2) {
                sigma[i] = -sigma[i];
            }
        }
    }

    // copy sigma => A
    lapack::laset( lapack::MatrixType::General, A.m, A.n, c_zero, c_zero, A(0,0), A.ld );
    for (int64_t i = 0; i < minmn; ++i) {
        *A(i,i) = sigma[i];
    }
}


// -----------------------------------------------------------------------------
/// Given matrix A with singular values such that sum(sigma_i^2) = n,
/// returns A with columns of unit norm, with the same condition number.
/// see: Davies and Higham, 2000, Numerically stable generation of correlation
/// matrices and their factors.
template< typename scalar_t >
void generate_correlation_factor( Matrix<scalar_t>& A )
{
    //const scalar_t eps = std::numeric_limits<scalar_t>::epsilon();

    Vector<scalar_t> x( A.n );
    for (int64_t j = 0; j < A.n; ++j) {
        x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
    }

    for (int64_t i = 0; i < A.n; ++i) {
        for (int64_t j = 0; j < A.n; ++j) {
            if ((x[i] < 1 && 1 < x[j]) || (x[i] > 1 && 1 > x[j])) {
                scalar_t xij, d, t, c, s;
                xij = blas::dot( A.m, A(0,i), 1, A(0,j), 1 );
                d = sqrt( xij*xij - (x[i] - 1)*(x[j] - 1) );
                t = (xij + std::copysign( d, xij )) / (x[j] - 1);
                c = 1 / sqrt(1 + t*t);
                s = c*t;
                blas::rot( A.m, A(0,i), 1, A(0,j), 1, c, -s );
                x[i] = blas::dot( A.m, A(0,i), 1, A(0,i), 1 );
                //if (x[i] - 1 > 30*eps) {
                //    printf( "i %d, x[i] %.6f, x[i] - 1 %.6e, 30*eps %.6e\n",
                //            i, x[i], x[i] - 1, 30*eps );
                //}
                //assert( x[i] - 1 < 30*eps );
                x[i] = 1;
                x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
                break;
            }
        }
    }
}


// -----------------------------------------------------------------------------
// specialization to complex
// can't use Higham's algorithm in complex
template<>
void generate_correlation_factor( Matrix<std::complex<float>>& A )
{
    assert( false );
}

template<>
void generate_correlation_factor( Matrix<std::complex<double>>& A )
{
    assert( false );
}


// -----------------------------------------------------------------------------
template< typename scalar_t >
void generate_svd(
    MatrixParams& params,
    Dist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    Vector< blas::real_type<scalar_t> >& sigma,
    Matrix<scalar_t>& A )
{
    using real_t = blas::real_type<scalar_t>;

    // locals
    int64_t m = A.m;
    int64_t n = A.n;
    int64_t maxmn = std::max( m, n );
    int64_t minmn = std::min( m, n );
    int64_t sizeU;
    int64_t info = 0;
    Matrix<scalar_t> U( maxmn, minmn );
    Vector<scalar_t> tau( minmn );

    // query for workspace
    lapack::unmqr( lapack::Side::Left, lapack::Op::NoTrans, A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    assert( info == 0 );
    lapack::unmqr( lapack::Side::Right, lapack::Op::Trans, A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    assert( info == 0 );

    // ----------
    generate_sigma( params, dist, false, cond, sigma_max, sigma, A );

    // for generate correlation factor, need sum sigma_i^2 = n
    // scaling doesn't change cond
    if (condD != 1) {
        real_t sum_sq = blas::dot( sigma.n, sigma(0), 1, sigma(0), 1 );
        real_t scale = sqrt( sigma.n / sum_sq );
        blas::scal( sigma.n, scale, sigma(0), 1 );
        // copy sigma to diag(A)
        for (int64_t i = 0; i < sigma.n; ++i) {
            *A(i,i) = *sigma(i);
        }
    }

    // random U, m-by-minmn
    // just make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    sizeU = U.size();
    lapack::larnv( idist_randn, params.iseed, sizeU, U(0,0) );
    for (int64_t j = 0; j < minmn; ++j) {
        int64_t mj = m - j;
        lapack::larfg( mj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    // A = U*A
    lapack::unmqr( lapack::Side::Left, lapack::Op::NoTrans, A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    assert( info == 0 );

    // random V, n-by-minmn (stored column-wise in U)
    lapack::larnv( idist_randn, params.iseed, sizeU, U(0,0) );
    for (int64_t j = 0; j < minmn; ++j) {
        int64_t nj = n - j;
        lapack::larfg( nj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    // A = A*V^H
    lapack::unmqr( lapack::Side::Right, lapack::Op::Trans, A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    assert( info == 0 );

    if (condD != 1) {
        // A = A*W, W orthogonal, such that A has unit column norms
        // i.e., A'*A is a correlation matrix with unit diagonal
        generate_correlation_factor( A );

        // A = A*D col scaling
        Vector<real_t> D( A.n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, params.iseed, D.n, D(0) );
        for (int64_t i = 0; i < D.n; ++i) {
            D[i] = exp( D[i] * range );
        }
        // TODO: add argument to return D to caller?
        if (params.verbose) {
            printf( "D = [" );
            for (int64_t i = 0; i < D.n; ++i) {
                printf( " %11.8g", D[i] );
            }
            printf( " ];\n" );
        }
        for (int64_t j = 0; j < A.n; ++j) {
            for (int64_t i = 0; i < A.m; ++i) {
                *A(i,j) *= D[j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void generate_heev(
    MatrixParams& params,
    Dist dist, bool rand_sign,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    Vector< blas::real_type<scalar_t> >& sigma,
    Matrix<scalar_t>& A )
{
    using real_t = blas::real_type<scalar_t>;

    // check inputs
    assert( A.m == A.n );

    // locals
    int64_t n = A.n;
    int64_t sizeU;
    int64_t info = 0;
    Matrix<scalar_t> U( n, n );
    Vector<scalar_t> tau( n );

    // query for workspace
    lapack::unmqr( lapack::Side::Left, lapack::Op::NoTrans, n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );

    lapack::unmqr( lapack::Side::Right, lapack::Op::ConjTrans, n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    assert( info == 0 );

    // ----------
    generate_sigma( params, dist, rand_sign, cond, sigma_max, sigma, A );

    // random U, n-by-n
    // just make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    sizeU = U.size();
    lapack::larnv( idist_randn, params.iseed, sizeU, U(0,0) );
    for (int64_t j = 0; j < n; ++j) {
        int64_t nj = n - j;
        lapack::larfg( nj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    // A = U*A
    lapack::unmqr( lapack::Side::Left, lapack::Op::NoTrans, n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    assert( info == 0 );

    // A = A*U^H
    lapack::unmqr( lapack::Side::Right, lapack::Op::ConjTrans, n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    assert( info == 0 );

    // make diagonal real
    // usually LAPACK ignores imaginary part anyway, but Matlab doesn't
    for (int i = 0; i < n; ++i) {
        *A(i,i) = std::real( *A(i,i) );
    }

    if (condD != 1) {
        // A = D*A*D row & column scaling
        Vector<real_t> D( n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, params.iseed, n, D(0) );
        for (int64_t i = 0; i < n; ++i) {
            D[i] = exp( D[i] * range );
        }
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                *A(i,j) *= D[i] * D[j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void generate_geev(
    MatrixParams& params,
    Dist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    Vector< blas::real_type<scalar_t> >& sigma,
    Matrix<scalar_t>& A )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void generate_geevx(
    MatrixParams& params,
    Dist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    Vector< blas::real_type<scalar_t> >& sigma,
    Matrix<scalar_t>& A )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
/// Purpose
/// -------
/// Generate an m-by-n test matrix A.
/// Similar to but does not use LAPACK's libtmg.
///
/// Arguments
/// ---------
/// @param[in]
/// params    MAGMA options. Uses matrix, cond, condD; see further details.
///
/// @param[in,out]
/// sigma   Real array, dimension (min(m,n))
///         For matrix with "_specified", on input contains user-specified
///         singular or eigenvalues.
///         On output, contains singular or eigenvalues, if known,
///         else set to NaN. sigma is not necesarily sorted.
///
/// @param[out]
/// A       Complex array, dimension (lda, n).
///         On output, the m-by-n test matrix A in an lda-by-n array.
///
/// Further Details
/// ---------------
/// The --matrix command line option specifies the matrix name according to the
/// table below. Where indicated, names take an optional distribution suffix (#)
/// and an optional scaling suffix (*). The default distribution is rand.
/// Examples: rand, rand_small, svd_arith, heev_geo_small.
///
/// The --cond and --condD command line options specify condition numbers as
/// described below. Default cond = sqrt( 1/eps ) = 6.7e7 for double, condD = 1.
///
/// Sigma is a diagonal matrix with entries sigma_i for i = 1, ..., n;
/// Lambda is a diagonal matrix with entries lambda_i = sigma_i with random sign;
/// U and V are random orthogonal matrices from the Haar distribution
/// (See: Stewart, The efficient generation of random orthogonal matrices
///  with an application to condition estimators, 1980);
/// X is a random matrix.
///
/// See LAPACK Working Note (LAWN) 41:
/// Table  5 (Test matrices for the nonsymmetric eigenvalue problem)
/// Table 10 (Test matrices for the symmetric eigenvalue problem)
/// Table 11 (Test matrices for the singular value decomposition)
///
/// Matrix      Description
/// zero
/// identity
/// jordan      ones on diagonal and first subdiagonal
///
/// rand*       matrix entries random uniform on (0, 1)
/// randu*      matrix entries random uniform on (-1, 1)
/// randn*      matrix entries random normal with mean 0, sigma 1
///
/// diag#*      A = Sigma
/// svd#*       A = U Sigma V^H
/// poev#*      A = V Sigma V^H  (eigenvalues positive [1], i.e., matrix SPD)
/// spd#*       alias for poev
/// heev#*      A = V Lambda V^H (eigenvalues mixed signs)
/// syev#*      alias for heev
/// geev#*      A = V T V^H, Schur-form T                       [not yet implemented]
/// geevx#*     A = X T X^{-1}, Schur-form T, X ill-conditioned [not yet implemented]
///
/// # optional distribution suffix
/// _rand       sigma_i random uniform on (0, 1) [default]
/// _randu      sigma_i random uniform on (-1, 1)
/// _randn      sigma_i random normal with mean 0, std 1
///             [1] Note for _randu and _randn, Sigma contains negative values.
///             _rand* do not use cond, so the condition number is arbitrary.
///
/// _logrand    log(sigma_i) uniform on (log(1/cond), log(1))
/// _arith      sigma_i = 1 - (i - 1)/(n - 1)*(1 - 1/cond); sigma_{i+1} - sigma_i is constant
/// _geo        sigma_i = (cond)^{ -(i-1)/(n-1) };          sigma_{i+1} / sigma_i is constant
/// _cluster    sigma = [ 1, 1/cond, ..., 1/cond ]; 1 unit value, n-1 small values
/// _cluster2   sigma = [ 1, ..., 1, 1/cond ];      n-1 unit values, 1 small value
/// _rarith     _arith,    reversed order
/// _rgeo       _geo,      reversed order
/// _rcluster   _cluster,  reversed order
/// _rcluster2  _cluster2, reversed order
/// _specified  user specified sigma on input
///
/// * optional scaling & modifier suffix
/// _ufl        scale near underflow         = 1e-308 for double
/// _ofl        scale near overflow          = 2e+308 for double
/// _small      scale near sqrt( underflow ) = 1e-154 for double
/// _large      scale near sqrt( overflow  ) = 6e+153 for double
/// _dominant   diagonally dominant: set A_ii = ± max( sum_j |A_ij|, sum_j |A_ji| )
///             Note _dominant changes the singular or eigenvalues.
///
/// [below scaling by D implemented, scaling by K not yet implemented]
/// If condD != 1, then:
/// For SVD, A = (U Sigma V^H) K D, where
/// K is diagonal such that columns of (U Sigma V^H K) have unit norm,
/// hence (A^T A) has unit diagonal,
/// and D has log-random entries in ( log(1/condD), log(1) ).
///
/// For heev, A0 = U Lambda U^H, A = D K A0 K D, where
/// K is diagonal such that (K A0 K) has unit diagonal, and D is as above.
///
/// Note using condD changes the singular or eigenvalues; on output, sigma
/// contains the singular or eigenvalues of A0, not of A.
/// See: Demmel and Veselic, Jacobi's method is more accurate than QR, 1992.
///
/// @ingroup testing
template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    Vector< blas::real_type<scalar_t> >& sigma,
    Matrix<scalar_t>& A )
{
    using real_t = blas::real_type<scalar_t>;

    // constants
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    const real_t d_zero = 0;
    const real_t d_one  = 1;
    const real_t ufl = std::numeric_limits< real_t >::min();      // == lamch("safe min")  ==  1e-38 or  2e-308
    const real_t ofl = 1 / ufl;                                   //                            8e37 or   4e307
    const real_t eps = std::numeric_limits< real_t >::epsilon();  // == lamch("precision") == 1.2e-7 or 2.2e-16
    const scalar_t c_zero = 0;
    const scalar_t c_one  = 1;

    // locals
    std::string kind = params.kind.value();
    std::vector< std::string > tokens = split( kind, "-_" );

    real_t cond = params.cond.value();
    bool cond_default = std::isnan( cond );
    if (cond_default) {
        cond = 1 / sqrt( eps );
    }

    real_t condD = params.condD.value();
    bool condD_default = std::isnan( condD );
    if (condD_default) {
        condD = 1;
    }

    real_t sigma_max = 1;
    int64_t minmn = std::min( A.m, A.n );

    // ----------
    // set sigma to unknown (nan)
    lapack::laset( lapack::MatrixType::General, sigma.n, 1, nan, nan, sigma(0), sigma.n );

    // ----- decode matrix type
    auto token = tokens.begin();
    if (token == tokens.end()) {
        throw std::runtime_error( "Error: empty matrix kind\n" );
    }
    std::string base = *token;
    ++token;
    TestMatrixType type = TestMatrixType::identity;
    if      (base == "zero"    ) { type = TestMatrixType::zero;     }
    else if (base == "identity") { type = TestMatrixType::identity; }
    else if (base == "jordan"  ) { type = TestMatrixType::jordan;   }
    else if (base == "randn"   ) { type = TestMatrixType::randn;    }
    else if (base == "randu"   ) { type = TestMatrixType::randu;    }
    else if (base == "rand"    ) { type = TestMatrixType::rand;     }
    else if (base == "diag"    ) { type = TestMatrixType::diag;     }
    else if (base == "svd"     ) { type = TestMatrixType::svd;      }
    else if (base == "poev" ||
             base == "spd"     ) { type = TestMatrixType::poev;     }
    else if (base == "heev" ||
             base == "syev"    ) { type = TestMatrixType::heev;     }
    else if (base == "geevx"   ) { type = TestMatrixType::geevx;    }
    else if (base == "geev"    ) { type = TestMatrixType::geev;     }
    else {
        fprintf( stderr, "%sUnrecognized matrix '%s'%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
        throw std::exception();
    }

    // ----- decode distribution
    std::string suffix;
    Dist dist = Dist::none;
    if (token != tokens.end()) {
        suffix = *token;
        if      (suffix == "randn"    ) { dist = Dist::randn;     }
        else if (suffix == "randu"    ) { dist = Dist::randu;     }
        else if (suffix == "rand"     ) { dist = Dist::rand;      }
        else if (suffix == "logrand"  ) { dist = Dist::logrand;   }
        else if (suffix == "arith"    ) { dist = Dist::arith;     }
        else if (suffix == "geo"      ) { dist = Dist::geo;       }
        else if (suffix == "cluster2" ) { dist = Dist::cluster2;  }
        else if (suffix == "cluster"  ) { dist = Dist::cluster;   }
        else if (suffix == "rarith"   ) { dist = Dist::rarith;    }
        else if (suffix == "rgeo"     ) { dist = Dist::rgeo;      }
        else if (suffix == "rcluster2") { dist = Dist::rcluster2; }
        else if (suffix == "rcluster" ) { dist = Dist::rcluster;  }
        else if (suffix == "specified") { dist = Dist::specified; }

        // if found, move to next token
        if (dist != Dist::none) {
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::diag ||
                   type == TestMatrixType::svd  ||
                   type == TestMatrixType::poev ||
                   type == TestMatrixType::heev ||
                   type == TestMatrixType::geev ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " distribution suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }
    if (dist == Dist::none)
        dist = Dist::logrand;  // default

    // ----- decode scaling
    sigma_max = 1;
    if (token != tokens.end()) {
        suffix = *token;
        if      (suffix == "small") { sigma_max = sqrt( ufl ); }
        else if (suffix == "large") { sigma_max = sqrt( ofl ); }
        else if (suffix == "ufl"  ) { sigma_max = ufl; }
        else if (suffix == "ofl"  ) { sigma_max = ofl; }

        // if found, move to next token
        if (sigma_max != 1) {
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::rand  ||
                   type == TestMatrixType::randu ||
                   type == TestMatrixType::randn ||
                   type == TestMatrixType::svd   ||
                   type == TestMatrixType::poev  ||
                   type == TestMatrixType::heev  ||
                   type == TestMatrixType::geev  ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " scaling suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }

    // ----- decode modifier
    bool dominant = false;
    if (token != tokens.end()) {
        suffix = *token;
        if (suffix == "dominant") {
            dominant = true;

            // move to next token
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::rand  ||
                   type == TestMatrixType::randu ||
                   type == TestMatrixType::randn ||
                   type == TestMatrixType::svd   ||
                   type == TestMatrixType::poev  ||
                   type == TestMatrixType::heev  ||
                   type == TestMatrixType::geev  ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " modifier suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }

    if (token != tokens.end()) {
        fprintf( stderr, "%sError in '%s': unknown suffix '%s'.%s\n",
                 ansi_red, kind.c_str(), token->c_str(), ansi_normal );
        throw std::exception();
    }

    // ----- check compatability of options
    if (A.m != A.n &&
        (type == TestMatrixType::jordan ||
         type == TestMatrixType::poev   ||
         type == TestMatrixType::heev   ||
         type == TestMatrixType::geev   ||
         type == TestMatrixType::geevx))
    {
        fprintf( stderr, "%sError: matrix '%s' requires m == n.%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
        throw std::exception();
    }

    if (type == TestMatrixType::zero      ||
        type == TestMatrixType::identity  ||
        type == TestMatrixType::jordan    ||
        type == TestMatrixType::randn     ||
        type == TestMatrixType::randu     ||
        type == TestMatrixType::rand)
    {
        // warn first time if user set cond and matrix doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s' ignores cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond.value(), ansi_normal );
        }
        params.cond_used.value() = libtest::no_data_flag;
    }
    else if (dist == Dist::randn ||
             dist == Dist::randu ||
             dist == Dist::rand)
    {
        // warn first time if user set cond and distribution doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s': rand, randn, and randu "
                     "singular/eigenvalue distributions ignore cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond.value(), ansi_normal );
        }
        params.cond_used.value() = libtest::no_data_flag;
    }
    else {
        params.cond_used.value() = cond;
    }

    if (! (type == TestMatrixType::svd ||
           type == TestMatrixType::heev ||
           type == TestMatrixType::poev))
    {
        // warn first time if user set condD and matrix doesn't use it
        static std::string last;
        if (! condD_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s' ignores condD %.2e.%s\n",
                     ansi_red, kind.c_str(), params.condD.value(), ansi_normal );
        }
    }

    if (type == TestMatrixType::poev &&
        (dist == Dist::randu ||
         dist == Dist::randn))
    {
        fprintf( stderr, "%sWarning: matrix '%s' using randu or randn "
                 "will not generate SPD matrix; use rand instead.%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
    }

    // ----- generate matrix
    switch (type) {
        case TestMatrixType::zero:
            lapack::laset( lapack::MatrixType::General, A.m, A.n, c_zero, c_zero, A(0,0), A.ld );
            lapack::laset( lapack::MatrixType::General, sigma.n, 1, d_zero, d_zero, sigma(0), sigma.n );
            break;

        case TestMatrixType::identity:
            lapack::laset( lapack::MatrixType::General, A.m, A.n, c_zero, c_one, A(0,0), A.ld );
            lapack::laset( lapack::MatrixType::General, sigma.n, 1, d_one, d_one, sigma(0), sigma.n );
            break;

        case TestMatrixType::jordan: {
            int64_t n1 = A.n - 1;
            lapack::laset( lapack::MatrixType::Upper, A.n, A.n, c_zero, c_one, A(0,0), A.ld );  // ones on diagonal
            lapack::laset( lapack::MatrixType::Lower, n1,  n1,  c_zero, c_one, A(1,0), A.ld );  // ones on sub-diagonal
            break;
        }

        case TestMatrixType::rand:
        case TestMatrixType::randu:
        case TestMatrixType::randn: {
            //int64_t idist = (int64_t) type;
            int64_t idist = 1;
            int64_t sizeA = A.ld * A.n;
            lapack::larnv( idist, params.iseed, sizeA, A(0,0) );
            if (sigma_max != 1) {
                scalar_t scale = sigma_max;
                blas::scal( sizeA, scale, A(0,0), 1 );
            }
            break;
        }

        case TestMatrixType::diag:
            generate_sigma( params, dist, false, cond, sigma_max, sigma, A );
            break;

        case TestMatrixType::svd:
            generate_svd( params, dist, cond, condD, sigma_max, sigma, A );
            break;

        case TestMatrixType::poev:
            generate_heev( params, dist, false, cond, condD, sigma_max, sigma, A );
            break;

        case TestMatrixType::heev:
            generate_heev( params, dist, true, cond, condD, sigma_max, sigma, A );
            break;

        case TestMatrixType::geev:
            generate_geev( params, dist, cond, sigma_max, sigma, A );
            break;

        case TestMatrixType::geevx:
            generate_geevx( params, dist, cond, sigma_max, sigma, A );
            break;
    }

    if (dominant) {
        // make diagonally dominant; strict unless diagonal has zeros
        for (int i = 0; i < minmn; ++i) {
            real_t sum = std::max( blas::asum( A.m, A(0,i), 1    ),    // i-th col
                                   blas::asum( A.n, A(i,0), A.ld ) );  // i-th row
            *A(i,i) = sum;
        }
        // reset sigma to unknown (nan)
        lapack::laset( lapack::MatrixType::General, sigma.n, 1, nan, nan, sigma(0), sigma.n );
    }
}


// -----------------------------------------------------------------------------
/// Traditional interface with m, n, lda instead of Matrix object.
template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    blas::real_type<scalar_t>* sigma_ptr,
    scalar_t* A_ptr, int64_t lda )
{
    using real_t = blas::real_type<scalar_t>;

    // vector & matrix wrappers
    // if sigma is null, create new vector; data is discarded later
    Vector<real_t> sigma( sigma_ptr, std::min( m, n ) );
    if (sigma_ptr == nullptr) {
        sigma = Vector<real_t>( std::min( m, n ) );
    }
    Matrix<scalar_t> A( A_ptr, m, n, lda );
    generate_matrix( params, sigma, A );
}


// -----------------------------------------------------------------------------
// explicit instantiations
template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    float* sigma_ptr,
    float* A_ptr, int64_t lda );

template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    double* sigma_ptr,
    double* A_ptr, int64_t lda );

template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    float* sigma_ptr,
    std::complex<float>* A_ptr, int64_t lda );

template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    double* sigma_ptr,
    std::complex<double>* A_ptr, int64_t lda );

} // namespace lapack
