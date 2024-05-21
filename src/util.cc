// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/util.hh"

namespace lapack {

const char* Sides_help          = "eigvec: L=Left, R=Right, "
                                  "B=Both left and right";

const char* Norm_help           = "1=O=One norm, I=Inf norm, 2=Two norm, "
                                  "F=Fro Frobenius norm, M=Max norm";

const char* itype_help          = "generalized eigenvalue problem type: "
                                  "1, 2, or 3. "
                                  "Type 1: Ax = lBx, "
                                  "type 2: ABx = lx, "
                                  "type 3: BAx = lx";

#define Job_eig                   "N=NoVec, V=Vectors"
const char* Job_eig_help        = "eigvec: "       Job_eig;
const char* Job_eig_left_help   = "left eigvec: "  Job_eig;
const char* Job_eig_right_help  = "right eigvec: " Job_eig;

#define Job_svd                   "singular vectors: A=All, " \
                                  "S=Some ('economy size'), " \
                                  "O=Overwrite A, N=NoVec"
const char* Job_svd_left_help   = "left "  Job_svd;
const char* Job_svd_right_help  = "right " Job_svd;

const char* JobSchur_help       = "E=Eigval, S=Schur form";

const char* Sort_help           = "N=NotSorted, S=Sorted";

const char* Range_help          = "eig/svd value range: A=All, V=Value, I=Index";

const char* Vect_help           = "form Q or P: Q, P, N=None, B=Both";

const char* Direction_help      = "apply Householder H: F=Forward H = H1...Hk, "
                                  "B=Backward H = Hk...H1";

const char* StoreV_help         = "Householder vectors stored: "
                                  "R=Row=Rowwise, C=Col=Colwise";

const char* MatrixType_help     = "G=General, L=Lower, U=Upper, H=Hessenberg, "
                                  "B=Band-lower, Q=Band-upper, Z=Band";

const char* HowMany_help        = "A=All, B=Backtransform all, S=Select";

const char* Equed_help          = "Equilibrate: N=None, R=Row, C=Col, B=Both, "
                                  "Y=Yes";

const char* Factored_help       = "F=Factored, N=NotFactored, E=Equilibrate";

const char* Sense_help          = "N=None, E=Eigval, V=Subspace, B=Both";

const char* JobCond_help        = "E=Eigvec (Hermitian), L=Left singular vectors, "
                                  "R=Right singular vectors";

const char* Balance_help        = "N=None, P=Permute, S=Scale, B=Both";

const char* Order_help          = "order eigvals within: "
                                  "B=Block, E=Entire matrix";

const char* RowCol_help         = "check orthogonality of: R=Row, C=Col";

}  // namespace lapack
