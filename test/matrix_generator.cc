// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <exception>
#include <string>
#include <vector>
#include <limits>
#include <complex>

#include "matrix_params.hh"
#include "matrix_generator.hh"

// -----------------------------------------------------------------------------
// ANSI color codes
using testsweeper::ansi_esc;
using testsweeper::ansi_red;
using testsweeper::ansi_bold;
using testsweeper::ansi_normal;

// -----------------------------------------------------------------------------
/// Splits a string by any of the delimiters.
/// Adjacent delimiters will give empty tokens.
/// See https://stackoverflow.com/questions/53849
/// @ingroup util
std::vector< std::string >
    split( const std::string& str, const std::string& delims );

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
/// Generates sigma vector of singular or eigenvalues, according to distribution.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_sigma(
    MatrixParams& params,
    Dist dist, bool rand_sign,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    using real_t = blas::real_type<scalar_t>;

    // constants
    const scalar_t c_zero = 0;

    // locals
    int64_t minmn = std::min( A.m, A.n );
    require( minmn == sigma.n );

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

        case Dist::cluster0:
            sigma[0] = 1;
            for (int64_t i = 1; i < minmn; ++i) {
                sigma[i] = 1/cond;
            }
            break;

        case Dist::rcluster0:
            for (int64_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1/cond;
            }
            sigma[minmn-1] = 1;
            break;

        case Dist::cluster1:
            for (int64_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1;
            }
            sigma[minmn-1] = 1/cond;
            break;

        case Dist::rcluster1:
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
        case Dist::rands:
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
            throw lapack::Error();
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
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
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
    throw lapack::Error( "not implemented" );
}

template<>
void generate_correlation_factor( Matrix<std::complex<double>>& A )
{
    throw lapack::Error( "not implemented" );
}


// -----------------------------------------------------------------------------
/// Generates matrix using SVD, $A = U Sigma V^H$.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_svd(
    MatrixParams& params,
    Dist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
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

    // ----------
    generate_sigma( params, dist, false, cond, sigma_max, A, sigma );

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
    require( info == 0 );

    // random V, n-by-minmn (stored column-wise in U)
    lapack::larnv( idist_randn, params.iseed, sizeU, U(0,0) );
    for (int64_t j = 0; j < minmn; ++j) {
        int64_t nj = n - j;
        lapack::larfg( nj, U(j,j), U(j+1,j), 1, tau(j) );
    }

    // A = A*V^H
    lapack::unmqr( lapack::Side::Right, lapack::Op::ConjTrans, A.m, A.n, minmn,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    require( info == 0 );

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
/// Generates matrix using Hermitian eigenvalue decomposition, $A = U Sigma U^H$.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_heev(
    MatrixParams& params,
    Dist dist, bool rand_sign,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    using real_t = blas::real_type<scalar_t>;

    // check inputs
    require( A.m == A.n );

    // locals
    int64_t n = A.n;
    int64_t sizeU;
    int64_t info = 0;
    Matrix<scalar_t> U( n, n );
    Vector<scalar_t> tau( n );

    // ----------
    generate_sigma( params, dist, rand_sign, cond, sigma_max, A, sigma );

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
    require( info == 0 );

    // A = A*U^H
    lapack::unmqr( lapack::Side::Right, lapack::Op::ConjTrans, n, n, n,
                   U(0,0), U.ld, tau(0), A(0,0), A.ld );
    require( info == 0 );

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
/// Generates matrix using general eigenvalue decomposition, $A = V T V^H$,
/// with orthogonal eigenvectors.
/// Not yet implemented.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_geev(
    MatrixParams& params,
    Dist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
/// Generates matrix using general eigenvalue decomposition, $A = X T X^{-1}$,
/// with random eigenvectors.
/// Not yet implemented.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_geevx(
    MatrixParams& params,
    Dist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
void generate_matrix_usage()
{
    printf(
    "The --matrix, --cond, and --condD parameters specify a test matrix.\n"
    "See Test routines: generate_matrix in the HTML documentation for a\n"
    "complete description.\n"
    "\n"
    "%s--matrix%s is one of following:\n"
    "\n"
    "%sMatrix%s    |  %sDescription%s\n"
    "----------|-------------\n"
    "zero      |  all zero\n"
    "identity  |  ones on diagonal, rest zero\n"
    "jordan    |  ones on diagonal and first subdiagonal, rest zero\n"
    "          |  \n"
    "rand@     |  matrix entries random uniform on (0, 1)\n"
    "rands@    |  matrix entries random uniform on (-1, 1)\n"
    "randn@    |  matrix entries random normal with mean 0, std 1\n"
    "          |  \n"
    "diag^@    |  A = Sigma\n"
    "svd^@     |  A = U Sigma V^H\n"
    "poev^@    |  A = V Sigma V^H  (eigenvalues positive, i.e., matrix SPD)\n"
    "spd^@     |  alias for poev\n"
    "heev^@    |  A = V Lambda V^H (eigenvalues mixed signs)\n"
    "syev^@    |  alias for heev\n"
    "geev^@    |  A = V T V^H, Schur-form T                       [not yet implemented]\n"
    "geevx^@   |  A = X T X^{-1}, Schur-form T, X ill-conditioned [not yet implemented]\n"
    "\n"
    "^ and @ denote optional suffixes described below.\n"
    "\n"
    "%s^ Distribution%s  |  %sDescription%s\n"
    "----------------|-------------\n"
    "_logrand        |  log(sigma_i) random uniform on [ log(1/cond), log(1) ]; default\n"
    "_arith          |  sigma_i = 1 - frac{i - 1}{n - 1} (1 - 1/cond); arithmetic: sigma_{i+1} - sigma_i is constant\n"
    "_geo            |  sigma_i = (cond)^{ -(i-1)/(n-1) };             geometric:  sigma_{i+1} / sigma_i is constant\n"
    "_cluster0       |  sigma = [ 1, 1/cond, ..., 1/cond ];  1  unit value,  n-1 small values\n"
    "_cluster1       |  sigma = [ 1, ..., 1, 1/cond ];      n-1 unit values,  1  small value\n"
    "_rarith         |  _arith,    reversed order\n"
    "_rgeo           |  _geo,      reversed order\n"
    "_rcluster0      |  _cluster0, reversed order\n"
    "_rcluster1      |  _cluster1, reversed order\n"
    "_specified      |  user specified sigma on input\n"
    "                |  \n"
    "_rand           |  sigma_i random uniform on (0, 1)\n"
    "_rands          |  sigma_i random uniform on (-1, 1)\n"
    "_randn          |  sigma_i random normal with mean 0, std 1\n"
    "\n"
    "%s@ Scaling%s       |  %sDescription%s\n"
    "----------------|-------------\n"
    "_ufl            |  scale near underflow         = 1e-308 for double\n"
    "_ofl            |  scale near overflow          = 2e+308 for double\n"
    "_small          |  scale near sqrt( underflow ) = 1e-154 for double\n"
    "_large          |  scale near sqrt( overflow  ) = 6e+153 for double\n"
    "\n"
    "%s@ Modifier%s      |  %sDescription%s\n"
    "----------------|-------------\n"
    "_dominant       |  make matrix diagonally dominant\n"
    "\n",
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal
    );
}

// -----------------------------------------------------------------------------
/// Generates an m-by-n test matrix.
/// Similar to LAPACK's libtmg functionality, but a level 3 BLAS implementation.
///
/// @param[in] params
///     Test matrix parameters. Uses matrix, cond, condD parameters;
///     see further details.
///
/// @param[out] A
///     Complex array, dimension (lda, n).
///     On output, the m-by-n test matrix A in an lda-by-n array.
///
/// @param[in,out] sigma
///     Real array, dimension (min(m,n))
///     - On input with matrix distribution "_specified",
///       contains user-specified singular or eigenvalues.
///     - On output, contains singular or eigenvalues, if known,
///       else set to NaN. sigma is not necesarily sorted.
///
/// ### Further Details
///
/// The **matrix** parameter specifies the matrix kind according to the
/// tables below. As indicated, kinds take an optional distribution suffix (^)
/// and an optional scaling and modifier suffix (@).
/// The default distribution is logrand.
/// Examples: rand, rand_small, svd_arith, heev_geo_small.
///
/// The **cond** parameter specifies the condition number $cond(S)$, where $S$ is either
/// the singular values $\Sigma$ or the eigenvalues $\Lambda$, as described by the
/// distributions below. It does not apply to some matrices and distributions.
/// For geev and geevx, cond(A) is generally much worse than cond(S).
/// If _dominant is applied, cond(A) generally improves.
/// By default, cond(S) = sqrt( 1/eps ) = 6.7e7 for double, 2.9e3 for single.
///
/// The **condD** parameter specifies the condition number cond(D), where D is
/// a diagonal scaling matrix [1]. By default, condD = 1. If condD != 1, then:
/// - For matrix = svd, set $A = A_0 K D$, where $A_0 = U \Sigma V^H$,
///   $D$ has log-random entries in $[ \log(1/condD), \log(1) ]$, and
///   $K$ is diagonal such that columns of $B = A_0 K$ have unit norm,
///   hence $B^T B$ has unit diagonal.
///
/// - For matrix = heev, set $A = D A_0 D$, where $A_0 = U \Lambda U^H$,
///   $D$ has log-random entries in $[ \log(1/condD), \log(1) ]$.
///   TODO: set $A = D K A_0 K D$ where
///   $K$ is diagonal such that $B = K A_0 K$ has unit diagonal.
///
/// Note using condD changes the singular or eigenvalues of $A$;
/// on output, sigma contains the singular or eigenvalues of $A_0$, not of $A$.
///
/// Notation used below:
/// $\Sigma$ is a diagonal matrix with entries $\sigma_i$ for $i = 1, \dots, n$;
/// $\Lambda$ is a diagonal matrix with entries $\lambda_i = \pm \sigma_i$,
/// with random sign;
/// $U$ and $V$ are random orthogonal matrices from the Haar distribution [2],
/// $X$ is a random matrix.
///
/// See LAPACK Working Note (LAWN) 41:\n
/// Table  5 (Test matrices for the nonsymmetric eigenvalue problem)\n
/// Table 10 (Test matrices for the symmetric eigenvalue problem)\n
/// Table 11 (Test matrices for the singular value decomposition)
///
/// Matrix   | Description
/// ---------|------------
/// zero     | all zero
/// identity | ones on diagonal, rest zero
/// jordan   | ones on diagonal and first subdiagonal, rest zero
/// --       | --
/// rand@    | matrix entries random uniform on (0, 1)
/// rands@   | matrix entries random uniform on (-1, 1)
/// randn@   | matrix entries random normal with mean 0, std 1
/// --       | --
/// diag^@   | $A = \Sigma$
/// svd^@    | $A = U \Sigma V^H$
/// poev^@   | $A = V \Sigma V^H$  (eigenvalues positive, i.e., matrix SPD)
/// spd^@    | alias for poev
/// heev^@   | $A = V \Lambda V^H$ (eigenvalues mixed signs)
/// syev^@   | alias for heev
/// geev^@   | $A = V T V^H$, Schur-form $T$                         [not yet implemented]
/// geevx^@  | $A = X T X^{-1}$, Schur-form $T$, $X$ ill-conditioned [not yet implemented]
///
/// Note for geev that $cond(\Lambda)$ is specified, where $\Lambda = diag(T)$;
/// while $cond(T)$ and $cond(A)$ are usually much worse.
///
/// ^ and @ denote optional suffixes described below.
///
/// ^ Distribution  |   Description
/// ----------------|--------------
/// _logrand        |  $\log(\sigma_i)$ random uniform on $[ \log(1/cond), \log(1) ]$; default
/// _arith          |  $\sigma_i = 1 - \frac{i - 1}{n - 1} (1 - 1/cond)$; arithmetic: $\sigma_{i+1} - \sigma_i$ is constant
/// _geo            |  $\sigma_i = (cond)^{ -(i-1)/(n-1) }$;              geometric:  $\sigma_{i+1} / \sigma_i$ is constant
/// _cluster0       |  $\Sigma = [ 1, 1/cond, ..., 1/cond ]$;  1     unit value,  $n-1$ small values
/// _cluster1       |  $\Sigma = [ 1, ..., 1, 1/cond ]$;       $n-1$ unit values, 1     small value
/// _rarith         |  _arith,    reversed order
/// _rgeo           |  _geo,      reversed order
/// _rcluster0      |  _cluster0, reversed order
/// _rcluster1      |  _cluster1, reversed order
/// _specified      |  user specified sigma on input
/// --              |  --
/// _rand           |  $\sigma_i$ random uniform on (0, 1)
/// _rands          |  $\sigma_i$ random uniform on (-1, 1)
/// _randn          |  $\sigma_i$ random normal with mean 0, std 1
///
/// Note _rand, _rands, _randn do not use cond; the condition number is random.
/// For randn, Expected( log( cond ) ) = log( 4.65 n ) [Edelman, 1988].
///
/// Note for _rands and _randn, $\Sigma$ contains negative values.
/// This means poev_rands and poev_randn will not generate an SPD matrix.
///
/// @ Scaling       |  Description
/// ----------------|-------------
/// _ufl            |  scale near underflow         = 1e-308 for double
/// _ofl            |  scale near overflow          = 2e+308 for double
/// _small          |  scale near sqrt( underflow ) = 1e-154 for double
/// _large          |  scale near sqrt( overflow  ) = 6e+153 for double
///
/// Note scaling changes the singular or eigenvalues, but not the condition number.
///
/// @ Modifier      |  Description
/// ----------------|-------------
/// _dominant       |  diagonally dominant: set $A_{i,i} = \pm \max_i( \sum_j |A_{i,j}|, \sum_j |A_{j,i}| )$.
///
/// Note _dominant changes the singular or eigenvalues, and the condition number.
///
/// ### References
///
/// [1] Demmel and Veselic, Jacobi's method is more accurate than QR, 1992.
///
/// [2] Stewart, The efficient generation of random orthogonal matrices
///     with an application to condition estimators, 1980.
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
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
    std::string kind = params.kind();
    std::vector< std::string > tokens = split( kind, "-_" );

    real_t cond = params.cond();
    bool cond_default = std::isnan( cond );
    if (cond_default) {
        cond = 1 / sqrt( eps );
    }

    real_t condD = params.condD();
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
    else if (base == "rands"   ) { type = TestMatrixType::rands;    }
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
        else if (suffix == "rands"    ) { dist = Dist::rands;     }
        else if (suffix == "rand"     ) { dist = Dist::rand;      }
        else if (suffix == "logrand"  ) { dist = Dist::logrand;   }
        else if (suffix == "arith"    ) { dist = Dist::arith;     }
        else if (suffix == "geo"      ) { dist = Dist::geo;       }
        else if (suffix == "cluster1" ) { dist = Dist::cluster1;  }
        else if (suffix == "cluster0" ) { dist = Dist::cluster0;  }
        else if (suffix == "rarith"   ) { dist = Dist::rarith;    }
        else if (suffix == "rgeo"     ) { dist = Dist::rgeo;      }
        else if (suffix == "rcluster1") { dist = Dist::rcluster1; }
        else if (suffix == "rcluster0") { dist = Dist::rcluster0; }
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
                   type == TestMatrixType::rands ||
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
                   type == TestMatrixType::rands ||
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
        type == TestMatrixType::rands     ||
        type == TestMatrixType::rand)
    {
        // warn first time if user set cond and matrix doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s' ignores cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond(), ansi_normal );
        }
        params.cond_used() = testsweeper::no_data_flag;
    }
    else if (dist == Dist::randn ||
             dist == Dist::rands ||
             dist == Dist::rand)
    {
        // warn first time if user set cond and distribution doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s': rand, randn, and rands "
                     "singular/eigenvalue distributions ignore cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond(), ansi_normal );
        }
        params.cond_used() = testsweeper::no_data_flag;
    }
    else {
        params.cond_used() = cond;
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
                     ansi_red, kind.c_str(), params.condD(), ansi_normal );
        }
    }

    if (type == TestMatrixType::poev &&
        (dist == Dist::rands ||
         dist == Dist::randn))
    {
        fprintf( stderr, "%sWarning: matrix '%s' using rands or randn "
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
        case TestMatrixType::rands:
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
            generate_sigma( params, dist, false, cond, sigma_max, A, sigma );
            break;

        case TestMatrixType::svd:
            generate_svd( params, dist, cond, condD, sigma_max, A, sigma );
            break;

        case TestMatrixType::poev:
            generate_heev( params, dist, false, cond, condD, sigma_max, A, sigma );
            break;

        case TestMatrixType::heev:
            generate_heev( params, dist, true, cond, condD, sigma_max, A, sigma );
            break;

        case TestMatrixType::geev:
            generate_geev( params, dist, cond, sigma_max, A, sigma );
            break;

        case TestMatrixType::geevx:
            generate_geevx( params, dist, cond, sigma_max, A, sigma );
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
/// Generates an m-by-n test matrix.
/// Traditional interface with m, n, lda instead of Matrix object.
///
/// @see generate_matrix()
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    scalar_t* A_ptr, int64_t lda,
    blas::real_type<scalar_t>* sigma_ptr )
{
    using real_t = blas::real_type<scalar_t>;

    // vector & matrix wrappers
    // if sigma is null, create new vector; data is discarded later
    Vector<real_t> sigma( sigma_ptr, std::min( m, n ) );
    if (sigma_ptr == nullptr) {
        sigma = Vector<real_t>( std::min( m, n ) );
    }
    Matrix<scalar_t> A( A_ptr, m, n, lda );
    generate_matrix( params, A, sigma );
}


// -----------------------------------------------------------------------------
// explicit instantiations
template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    float* A_ptr, int64_t lda,
    float* sigma_ptr );

template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    double* A_ptr, int64_t lda,
    double* sigma_ptr );

template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    std::complex<float>* A_ptr, int64_t lda,
    float* sigma_ptr );

template
void generate_matrix(
    MatrixParams& params,
    int64_t m, int64_t n,
    std::complex<double>* A_ptr, int64_t lda,
    double* sigma_ptr );

} // namespace lapack
