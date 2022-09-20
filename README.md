
     |      \   _ \  \   __| |  /    |     |
     |     _ \  __/ _ \ (      <  __ __|__ __|
    ____|_/  _\_| _/  _\___|_|\_\   _|    _|

**C++ API for the Linear Algebra PACKage**

**Innovative Computing Laboratory**

**University of Tennessee**

* * *

[TOC]

* * *

About
--------------------------------------------------------------------------------

The Linear Algebra PACKage (LAPACK) is a standard software library
for numerical linear algebra. It provides routines for solving
systems of linear equations and linear least squares problems,
eigenvalue problems, and singular value decomposition.
It also includes routines to implement the associated matrix factorizations
such as LU, QR, Cholesky, etc. LAPACK was originally written in FORTRAN 77,
and moved to Fortran 90 in version 3.2 (2008). LAPACK provides routines
for handling both real and complex matrices in both single and double precision.

The objective of LAPACK++ is to provide a convenient, performance oriented API
for development in the C++ language, that, for the most part,
preserves established conventions, while, at the same time, takes advantages
of modern C++ features, such as: namespaces, templates, exceptions, etc.

LAPACK++ is part of the SLATE project
([Software for Linear Algebra Targeting Exascale](http://icl.utk.edu/slate/)),
which is funded by the [Department of Energy](https://energy.gov)
as part of its [Exascale Computing Initiative](https://exascaleproject.org)
(ECP).
Closely related to LAPACK++ is the
[BLAS++](https://github.com/icl-utk-edu/blaspp) project,
which provides a C++ API for BLAS and Batch BLAS.

![LAPACKPP](http://icl.bitbucket.io/slate/artwork/Bitbucket/lapackpp_stack.png)

* * *

Documentation
--------------------------------------------------------------------------------

* [INSTALL.md](INSTALL.md) for installation notes.
* [LAPACK++ Doxygen](https://icl.bitbucket.io/lapackpp/)
* [SLATE Working Note 2: C++ API for BLAS and LAPACK](http://www.icl.utk.edu/publications/swan-002)

* * *

Getting Help
--------------------------------------------------------------------------------

For assistance, visit the *SLATE User Forum* at
<https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user>.
Join by signing in with your Google credentials, then clicking
*Join group to post*.

Bug reports can be filed directly on Github's issue tracker:
<https://github.com/icl-utk-edu/lapackpp/issues>.

* * *

Resources
--------------------------------------------------------------------------------

* Visit the [BLAS++ repository](https://github.com/icl-utk-edu/blaspp)
  for more information about the C++ API for the standard BLAS.
* Visit the [SLATE website](http://icl.utk.edu/slate/)
  for more information about the SLATE project.
* Visit the [SLATE Working Notes](http://www.icl.utk.edu/publications/series/swans)
  to find out more about ongoing SLATE developments.
* Visit the [ECP website](https://exascaleproject.org)
  to find out more about the DOE Exascale Computing Initiative.

* * *

Contributing
--------------------------------------------------------------------------------

The SLATE project welcomes contributions from new developers.
Contributions can be offered through the standard Github pull request model.
We strongly encourage you to coordinate large contributions with the SLATE
development team early in the process.

* * *

Acknowledgments
--------------------------------------------------------------------------------

This research was supported by the Exascale Computing Project (17-SC-20-SC), a
joint project of the U.S. Department of Energy's Office of Science and National
Nuclear Security Administration, responsible for delivering a capable exascale
ecosystem, including software, applications, and hardware technology, to support
the nation’s exascale computing imperative.

This research uses resources of the Oak Ridge Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.
This research also uses resources of the Argonne Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

* * *

License
--------------------------------------------------------------------------------

Copyright (c) 2017-2022, University of Tennessee. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the University of Tennessee nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

**This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall the copyright holders or contributors be liable
for any direct, indirect, incidental, special, exemplary, or consequential
damages (including, but not limited to, procurement of substitute goods or
services; loss of use, data, or profits; or business interruption) however
caused and on any theory of liability, whether in contract, strict liability, or
tort (including negligence or otherwise) arising in any way out of the use of
this software, even if advised of the possibility of such damage.**
