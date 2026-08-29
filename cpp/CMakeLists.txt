# etrade — C++17 port of the electronic-trading toolkit
# (Almgren-Chriss optimal execution, Avellaneda-Stoikov market making,
# execution analytics).
# Targets: etrade (static lib), etrade_demo, etrade_tests (GoogleTest).
# Tests run with the working directory set to this source dir so the golden
# file resolves at ../data/golden/golden.json (per the contract).
cmake_minimum_required(VERSION 3.16)
project(etrade LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

add_library(etrade
  src/almgren_chriss.cpp
  src/avellaneda_stoikov.cpp
  src/analytics.cpp)
target_include_directories(etrade PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(etrade_demo src/demo.cpp)
target_link_libraries(etrade_demo PRIVATE etrade)

enable_testing()
find_package(GTest REQUIRED)

add_executable(etrade_tests
  tests/test_golden.cpp
  tests/test_almgren_chriss.cpp
  tests/test_avellaneda_stoikov.cpp
  tests/test_analytics.cpp)
target_include_directories(etrade_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(etrade_tests PRIVATE etrade GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(etrade_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
