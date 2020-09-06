# CXX compiler must match the one used to compiler LAPACK++.
# Set it in your environment.

cmake_minimum_required( VERSION 3.8 )

project(
    lapackpp_example
    LANGUAGES CXX
)

add_executable(
    example_potrf
    example_potrf.cc
)

find_package( lapackpp REQUIRED )

target_link_libraries(
    example_potrf
    lapackpp
)
