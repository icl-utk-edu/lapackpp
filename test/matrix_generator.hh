
#ifndef MATRIX_GENERATOR_HPP
#define MATRIX_GENERATOR_HPP

#include <algorithm>  // copy, swap
#include "test.hh"
#include "lapack.hh"

/******************************************************************************/
// Uses copy-and-swap idiom.
// https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//

#define LAPACK_D_ZERO 0.0
#define LAPACK_D_ONE 1.0
#define LAPACK_S_ZERO 0.0
#define LAPACK_S_ONE 1.0

const int64_t idist_rand  = 1;
const int64_t idist_randu = 2;
const int64_t idist_randn = 3;

enum class MatrixType {
    rand      = 1,  // maps to larnv idist
    randu     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    zero,
    identity,
    jordan,
    diag,
    svd,
    poev,
    heev,
    geev,
    geevx,
};

enum class Dist {
    rand      = 1,  // maps to larnv idist
    randu     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    arith,
    geo,
    cluster,
    cluster2,
    rarith,
    rgeo,
    rcluster,
    rcluster2,
    logrand,
    specified,
};

template< typename FloatType >
class Vector
{
public:
    // constructor allocates new memory (unless n == 0)
    Vector( int64_t in_n=0 ):
        n    ( in_n ),
        data_( n > 0 ? new FloatType[n] : nullptr ),
        own_ ( true )
    {
        if (n < 0) { throw std::exception(); }
    }

    // constructor wraps existing memory; caller maintains ownership
    Vector( FloatType* data, int64_t in_n ):
        n    ( in_n ),
        data_( data ),
        own_ ( false )
    {
        if (n < 0) { throw std::exception(); }
    }

    // copy constructor
    Vector( Vector const &other ):
        n    ( other.n ),
        data_( nullptr ),
        own_ ( other.own_ )
    {
        if (other.own_) {
            if (n > 0) {
                data_ = new FloatType[n];
                std::copy( other.data_, other.data_ + n, data_ );
            }
        }
        else {
            data_ = other.data_;
        }
    }

    // move constructor, using copy & swap idiom
    Vector( Vector&& other )
        : Vector()
    {
        swap( *this, other );
    }

    // assignment operator, using copy & swap idiom
    Vector& operator= (Vector other)
    {
        swap( *this, other );
        return *this;
    }

    // destructor deletes memory if constructor allocated it
    // (i.e., not if wrapping existing memory)
    ~Vector()
    {
        if (own_) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    friend void swap( Vector& first, Vector& second )
    {
        using std::swap;
        swap( first.n,     second.n     );
        swap( first.data_, second.data_ );
        swap( first.own_,  second.own_  );
    }

    // returns pointer to element i, because that's what we normally need to
    // call BLAS / LAPACK, which avoids littering the code with &.
    FloatType*       operator () ( int64_t i )       { return &data_[ i ]; }
    FloatType const* operator () ( int64_t i ) const { return &data_[ i ]; }

    // return element i itself, as usual in C/C++.
    // unfortunately, this won't work for matrices.
    FloatType&       operator [] ( int64_t i )       { return data_[ i ]; }
    FloatType const& operator [] ( int64_t i ) const { return data_[ i ]; }

    int64_t size() const { return n; }
    bool        own()  const { return own_; }

public:
    int64_t n;

private:
    FloatType *data_;
    bool own_;
};

/******************************************************************************/
template< typename FloatType >
class Matrix
{
public:
    // constructor allocates new memory
    // ld = m by default
    Matrix( int64_t in_m, int64_t in_n, int64_t in_ld=0 ):
        m( in_m ),
        n( in_n ),
        ld( in_ld == 0 ? m : in_ld ),
        data_( ld*n )
    {
        if (m  < 0) { throw std::exception(); }
        if (n  < 0) { throw std::exception(); }
        if (ld < m) { throw std::exception(); }
    }

    // constructor wraps existing memory; caller maintains ownership
    // ld = m by default
    Matrix( FloatType* data, int64_t in_m, int64_t in_n, int64_t in_ld=0 ):
        m( in_m ),
        n( in_n ),
        ld( in_ld == 0 ? m : in_ld ),
        data_( data, ld*n )
    {
        if (m  < 0) { throw std::exception(); }
        if (n  < 0) { throw std::exception(); }
        if (ld < m) { throw std::exception(); }
    }

    int64_t size() const { return data_.size(); }
    bool        own()  const { return data_.own(); }

    // returns pointer to element (i,j), because that's what we normally need to
    // call BLAS / LAPACK, which avoids littering the code with &.
    FloatType* operator () ( int i, int j )
        { return &data_[ i + j*ld ]; }

    FloatType const* operator () ( int i, int j ) const
        { return &data_[ i + j*ld ]; }

public:
    int64_t m, n, ld;

protected:
    Vector<FloatType> data_;
};

template< typename FloatT >
void lapack_generate_matrix(
    matrix_opts& opts,
    Vector< typename blas::traits<FloatT>::real_t >& sigma,
    Matrix< FloatT >& A );

template< typename FloatT >
void lapack_generate_matrix(
    matrix_opts& opts,
    int64_t m, int64_t n,
    typename blas::traits<FloatT>::real_t* sigma,
    FloatT* A, int64_t lda );

#endif        // #ifndef MATRIX_GENERATOR_HPP
