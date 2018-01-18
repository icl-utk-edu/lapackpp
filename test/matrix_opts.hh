#ifndef MATRIX_OPTS_HH
#define MATRIX_OPTS_HH

#include "lapack_util.hh"

#define MAX_NTEST 1050
#define MAXGPUS 8

typedef enum {
    MatrixOptsDefault = 0,
    MatrixOptsBatched = 1000
} matrix_opts_t;

typedef enum {
    MatrixRangeAll      = 311,  /* syevx, etc. */
    MatrixRangeV        = 312,
    MatrixRangeI        = 313
} matrix_range_t;

class matrix_opts
{
public:
    // constructor
    matrix_opts( matrix_opts_t flag=MatrixOptsDefault );

    // parse command line
    //void parse_opts( int argc, char** argv );

    // set range, vl, vu, il, iu for eigen/singular value problems (gesvdx, syevdx, ...)
    /*
    void get_range( int64_t n, lapack::Range* range,
                    double* vl, double* vu,
                    int64_t* il, int64_t* iu );

    void get_range( int64_t n, lapack::Range* range,
                    float* vl, float* vu,
                    int64_t* il, int64_t* iu );

    // deallocate queues, etc.
    void cleanup();

    // matrix size
    int64_t ntest;
    int64_t msize[ MAX_NTEST ];
    int64_t nsize[ MAX_NTEST ];
    int64_t ksize[ MAX_NTEST ];
    int64_t batchcount;

    int64_t default_nstart;
    int64_t default_nend;
    int64_t default_nstep;

    // scalars
    int64_t device;
    int64_t cache;
    int64_t align;
    int64_t nb;
    int64_t nrhs;
    int64_t nqueue;
    int64_t ngpu;
    int64_t nsub;
    int64_t niter;
    int64_t nthread;
    int64_t offset;
    int64_t itype;     // hegvd: problem type
    int64_t version;
    int64_t check;
    int64_t verbose;

    // ranges for eigen/singular values (gesvdx, heevdx, ...)
    double      fraction_lo;
    double      fraction_up;
    int64_t     irange_lo;
    int64_t     irange_up;
    double      vrange_lo;
    double      vrange_up;

    double      tolerance;

    // boolean arguments
    bool magma;
    bool lapack;
    bool warmup;
    */
    int64_t verbose;

    // lapack options
    lapack::Uplo    uplo;
    lapack::Op      transA;
    lapack::Op      transB;
    lapack::Side    side;
    lapack::Diag    diag;
    //magma_vec_t     jobz;    // heev:   no eigen vectors
    lapack::Job     jobz;    // heev:   no eigen vectors
    //magma_vec_t     jobvr;   // geev:   no right eigen vectors
    lapack::Job     jobvr;
    //magma_vec_t     jobvl;   // geev:   no left  eigen vectors
    lapack::Job     jobvl;

    // vectors of options
    //std::vector< magma_svd_work_t > svd_work;
    //std::vector< lapack::Job > jobu;
    //std::vector< lapack::Job > jobv;

    // LAPACK test matrix generation
    std::string matrix;
    double      cond;
    double      condD;
    int64_t     iseed[4];

};

#endif  // #ifndef MATRIX_OPTS_HH
