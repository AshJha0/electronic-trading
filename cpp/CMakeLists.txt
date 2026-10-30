# etrade — C++17 port of the execution and market-making research library
# (Almgren-Chriss optimal execution, Avellaneda-Stoikov market making,
# execution analytics).
#
# Targets: etrade::etrade (static library), etrade_demo, etrade_tests.
# Options:
#   ETRADE_BUILD_TESTS  build the GoogleTest suite (default ON when top level)
#   ETRADE_WERROR       treat warnings as errors (default OFF; CI turns it on)
#   ETRADE_SANITIZE     build with AddressSanitizer + UBSan (GCC/Clang only)
# GoogleTest is taken from the system when available, otherwise fetched.
cmake_minimum_required(VERSION 3.20)
project(etrade VERSION 1.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ETRADE_BUILD_TESTS "Build the etrade test suite" ${PROJECT_IS_TOP_LEVEL})
option(ETRADE_WERROR "Treat compiler warnings as errors" OFF)
option(ETRADE_SANITIZE "Build with ASan and UBSan (GCC/Clang)" OFF)

# Warning flags per compiler family (MSVC rejects -Wextra outright).
add_library(etrade_warnings INTERFACE)
if(MSVC)
  target_compile_options(etrade_warnings INTERFACE /W4 /permissive-
                         $<$<BOOL:${ETRADE_WERROR}>:/WX>)
else()
  target_compile_options(etrade_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow
                         $<$<BOOL:${ETRADE_WERROR}>:-Werror>)
endif()
if(ETRADE_SANITIZE AND NOT MSVC)
  target_compile_options(etrade_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(etrade_warnings INTERFACE -fsanitize=address,undefined)
endif()

add_library(etrade
  src/almgren_chriss.cpp
  src/avellaneda_stoikov.cpp
  src/analytics.cpp
  src/market.cpp)
add_library(etrade::etrade ALIAS etrade)
target_include_directories(etrade PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(etrade PUBLIC cxx_std_17)
target_link_libraries(etrade PRIVATE $<BUILD_INTERFACE:etrade_warnings>)

add_executable(etrade_demo src/demo.cpp)
target_link_libraries(etrade_demo PRIVATE etrade etrade_warnings)

include(GNUInstallDirs)
install(TARGETS etrade EXPORT etradeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT etradeTargets NAMESPACE etrade:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/etrade)

if(ETRADE_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(etrade_tests
    tests/test_golden.cpp
    tests/test_almgren_chriss.cpp
    tests/test_avellaneda_stoikov.cpp
    tests/test_analytics.cpp
    tests/test_market.cpp)
  target_include_directories(etrade_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_definitions(etrade_tests PRIVATE
    ETRADE_GOLDEN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../data/golden/golden.json"
    ETRADE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")
  target_link_libraries(etrade_tests PRIVATE etrade etrade_warnings GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(etrade_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       DISCOVERY_TIMEOUT 60)
endif()
