#include <cstdlib>
#include <new>
#include <limits>
#include <iostream>
#include <vector>

#include <type_traits>
#include <memory>
#include <blas.hh>

namespace lapack {
/**
 *  \brief Base type for allocators which allocate typed raw memory
 *  without construction/destruction
 *
 *  Provides a prototype for allocators which must allocate typed raw
 *  memory without construction / destruction (e.g. on device memory).
 *  no_constuct_allocator_base does not, in and of itself, satisfy the
 *  Allocator concept.
 *
 *  To satisfy the Allocator concept, the derived class must define
 *  the proper allocate / deallocate semantics oulined in the standard.
 *
 *  @tparam Type of the data to be allocated. Only valid for trivial
 *  types.
 */
//template < typename T, typename = typename std::enable_if< std::is_trivial<T>::value>::type > //original, gesvd does not compile
template < typename T, typename = typename std::enable_if< std::is_trivial<T>::value || blas::is_complex<T>::value >::type > // gesvd compiles
struct NoConstructAllocatorBase {

  using value_type = T;

  /**
   *  \brief Construction given an allocated pointer is a null-op.
   *
   *  @tparam Args Parameter pack which handles all possible calling
   *  signatures of construct outlined in the Allocator concept.
   */
  template <typename... Args>
  void construct( T* ptr, Args&&... args ){ }

  /**
   *  \brief Destruction of an object in allocated memory is a null-op
   */
  void destroy( T* ptr ){ }

}; // struct NoConstructAllocatorBase


// No-construct allocator type which allocates / deallocates via std::allocator
template <typename T>
struct NoConstructAllocator : public NoConstructAllocatorBase<T> {

  using value_type = T;

  //T* allocate( size_t n ) { return std::allocator<T>().allocate(n); }
  //void deallocate( T* ptr, size_t n ) { std::allocator<T>().deallocate( ptr, n ); }

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        void *memPtr = NULL;
        #if defined( _WIN32 ) || defined( _WIN64 )
            memPtr = _aligned_malloc( n*sizeof(T, 64 );
            if ( memPtr != NULL ) {
                auto p = static_cast<T*>(memPtr);
                report(p, n);
                return p;
            }
        #else
            int err = posix_memalign( &memPtr, 64, n*sizeof(T) );
            if ( err == 0 ) {
                auto p = static_cast<T*>(memPtr);
                //report(p, n);
                return p;
            }
        #endif

        auto p = static_cast<T*>( std::malloc( n*sizeof(T) ) );
        if ( memPtr != NULL ) {
            report(p, n);
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t n) noexcept {
        #if defined( _WIN32 ) || defined( _WIN64 )
             _aligned_free( p );
        #else
            std::free( p );
            //report( p, n, 0 );
        #endif
    }

private:
    void report( T* p, std::size_t n, bool alloc = true ) const {}
    #if 0
        void report(T* p, std::size_t n, bool alloc = true) const {
            std::cout << (alloc ? "Alloc: " : "Dealloc: ")
                << "nElements = " << n
                << " bytes = " << sizeof(T)*n
                << " at " << std::hex << std::showbase << std::dec
                << reinterpret_cast<void*>(p) << std::dec << '\n';
        }
    #endif
};

template <class T, class U>
bool operator==(const NoConstructAllocator <T>&, const NoConstructAllocator <U>&) { return true; }
template <class T, class U>
bool operator!=(const NoConstructAllocator <T>&, const NoConstructAllocator <U>&) { return false; }

template <typename T>
using vector = std::vector< T, NoConstructAllocator<T> >;

}  // namespace lapack