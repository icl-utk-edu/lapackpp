// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/util.hh"

#include <vector>

namespace lapack {

const char* Sides_help          = "l=left, r=right";
const char* Norm_help           = "1=o=one norm, i=inf norm, 2=two norm, f=Frobenius norm, m=max norm";
const char* Job_help            = "job";
const char* Job_eig_help        = "job eig";
const char* Job_eig_left_help   = "job eig";
const char* Job_eig_right_help  = "job eig";
const char* Job_svd_left_help   = "job eig";
const char* Job_svd_right_help  = "job eig";
const char* JobSchur_help       = "job schur";
const char* Sort_help           = "n=not sorted, s=sorted";
const char* Range_help          = "range";
const char* Vect_help           = "vect";
const char* Direction_help      = "f=forward, b=backward";
const char* StoreV_help         = "storev";
const char* MatrixType_help     = "matrix type";
const char* HowMany_help        = "howmany";
const char* Equed_help          = "equed";
const char* Factored_help       = "factored";
const char* Sense_help          = "sense";
const char* JobCond_help        = "jobcond";
const char* Balance_help        = "balance";
const char* Order_help          = "order";
const char* RowCol_help         = "rowcol";

}  // namespace lapack
