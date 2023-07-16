// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>
#include <ctgmath>

// -----------------------------------------------------------------------------
// Generate the Kahan matrix and its eigenvalues on three vectors.
// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_sturm_Kahan(int64_t n, std::vector<scalar_t>& diag,
                      std::vector<scalar_t>& offd, std::vector<scalar_t>& eigv,
                      scalar_t* one_norm)
{
    using real_t = blas::real_type< scalar_t >;
    real_t x;
    int64_t i,k;
    x = 1.e-5;
    for (k = 1; k <= (n/2); ++k) {  // generate the eigenvalues.
        real_t ev;
        ev = (M_PI*k+0.)/(n+1.0);   // angle in radians.
        ev = cos(ev);               // cos(angle)
        ev *= 4.*ev;                // 4*cos^2(angle)
        ev += x*x;                  // x^2 + 4*cos^2(angle)
        ev = sqrt(ev);              // (x^2 + 4*cos^2(angle))^(0.5)
        eigv[k-1] = -ev;            // Store the eigvalues in ascending order.
        eigv[n+1-k-1] = ev;
    }

    for (i = 0; i < n-1; ++i) {     // generate the diagonal and off-diagonal.
        offd[i] = 1.0;
        // (i & 1) = 1 if k is odd, 0 if k is even.
        diag[i] = (i & 1)?-x:x;     // use -x if k is odd, +x if k is even.
    }

    // Final entry; i=(n-1). We don't set offd[n-1].
    diag[i] = (i & 1)?-x:x;         // use -x if k is odd, +x if k is even.

    // We compute the one norm of the matrix; the maximum abs column sum.
    // This routine is generic for any ST matrix.
    real_t norm = std::abs(diag[0])+std::abs(offd[0]);
    real_t test;
    for (i = 1; i < n-1; ++i) {
        test = std::abs(diag[i])+std::abs(offd[i-1])+std::abs(offd[i+1]);
        if (test > norm) {
            norm = test;
        }
    }

    test = std::abs(diag[n-1])+std::abs(offd[n-2]);
    if (test > norm) {
        norm = test;
    }

    one_norm[0] = norm;
}

template< typename scalar_t >
void test_sturm_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t verbose = params.verbose();

    if (! run) {
        return;
    }

    // ---------- setup
    std::vector< scalar_t > diag( (size_t) n );
    std::vector< scalar_t > eigv( (size_t) n );
    std::vector< scalar_t > offd( (size_t) (n-1) );
    real_t real_max=std::numeric_limits< real_t >::max();

    real_t one_norm, my_ulp;
    test_sturm_Kahan(n, diag, offd, eigv, &one_norm);
    my_ulp = (real_t(2.))*(nextafter(one_norm, real_max) - one_norm);

    // 2*my_ulp seems to be the accuracy we can get from the matrix.
    if (verbose >= 2) {
        printf("\n"
               "One-norm = %.16e, 2*ulp=%.16e\n",
               one_norm, my_ulp);
    }

    // zero-rel, so if n=100, idx=50, eigv[50] is actually the 51st eigenvalue.
    int64_t eig_mid_idx = (n / 2);
    real_t eig_min, eig_min_before, eig_min_after;
    eig_min=eigv[0];
    eig_min_before = eig_min-my_ulp;
    eig_min_after  = eig_min+my_ulp;

    real_t eig_mid, eig_mid_before, eig_mid_after;
    eig_mid=eigv[(eig_mid_idx)];
    eig_mid_before = eig_mid-my_ulp;
    eig_mid_after  = eig_mid+my_ulp;

    real_t eig_max, eig_max_before, eig_max_after;
    eig_max=eigv[n-1];
    eig_max_before = eig_max-my_ulp;
    eig_max_after = eig_max+my_ulp;

    if (verbose >= 2) {
        printf( "\n"
                "eig_min=%.16e, eig_minbefore=%.16e, eig_minafter=%.16e\n",
                eig_min, eig_min_before, eig_min_after );
        printf( "\n"
                "eig_mid=%.16e, eig_midbefore=%.16e, eig_midafter=%.16e\n",
                eig_mid, eig_mid_before, eig_mid_after );
        printf( "\n"
                "eigv[eig_mid_idx-1]=%.16e diff=%.16e\n",
                eigv[eig_mid_idx-1], eigv[eig_mid_idx]-eigv[eig_mid_idx-1]);
        printf( "\n"
                "eig_max=%.16e, eig_maxbefore=%.16e, eig_maxafter=%.16e\n",
                eig_max, eig_max_before, eig_max_after );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t r_min_before, r_min_after;
    int64_t r_mid_before, r_mid_after;
    int64_t r_max_before, r_max_after;
    int64_t error;
    error = 0;

    r_min_before = lapack::sturm( n, &diag[0], &offd[0], eig_min_before);
    r_min_after  = lapack::sturm( n, &diag[0], &offd[0], eig_min_after );

    if (verbose >= 2) {
        printf( "\n"
                "r_minbefore=%lld r_minafter=%lld. Expected =0, >=1\n",
                llong(r_min_before), llong(r_min_after));
    }

    if (r_min_before > 0) {
        ++error;
    }

    if (r_min_after  < 1) {
        ++error;
    }

    r_mid_before = lapack::sturm( n, &diag[0], &offd[0], eig_mid_before);
    r_mid_after  = lapack::sturm( n, &diag[0], &offd[0], eig_mid_after );

    if (verbose >= 2) {
        printf( "\n"
                "r_midbefore=%lld r_midafter=%lld."
                " Expected <%lld, >=%lld\n",
                llong(r_mid_before), llong(r_mid_after), llong(eig_mid_idx+1),
                llong(eig_mid_idx+1));
    }

    // The number of eigenvalues less than mid should be the middle index itself.
    // If n==100, n/2=50, but zero relative, eigv[50] is actually the 51st
    // eigenvalue, so we'd expect 50 less than that.
    if (r_mid_before > eig_mid_idx) {
        ++error;
    }

    if (r_mid_after  < (eig_mid_idx+1)) {
        ++error;
    }

    r_max_before = lapack::sturm( n, &diag[0], &offd[0], eig_max_before);
    r_max_after  = lapack::sturm( n, &diag[0], &offd[0], eig_max_after );

    if (verbose >= 2) {
        printf( "\n"
                "r_maxbefore=%lld r_maxafter=%lld."
                " Expected <%lld, =%lld\n",
                llong(r_max_before), llong(r_max_after), llong(n), llong(n));
    }

    if (r_max_before > (n-1)) {
        ++error;
    }

    if (r_max_after !=  n   ) {
        ++error;
    }

    time = testsweeper::get_wtime() - time;

    params.ref_time() = time;
    params.error() = error;
    params.okay() = (error == 0);
}

// -----------------------------------------------------------------------------
void test_sturm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_sturm_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_sturm_work< double >( params, run );
            break;

        default:
            throw std::runtime_error( "unsupported datatype" );
            break;
    }
}
