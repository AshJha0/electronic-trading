# Hot-path benchmarks (Google Benchmark). Enabled with -DETRADE_BUILD_BENCHMARKS=ON;
# CI does not build them. Run: build/benchmarks/etrade_bench (see docs/BENCHMARKS.md).
include(FetchContent)
FetchContent_Declare(googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
  URL_HASH SHA256=32131c08ee31eeff2c8968d7e874f3cb648034377dfc32a4c377fa8796d84981
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(etrade_bench bench_hot_paths.cpp)
target_link_libraries(etrade_bench PRIVATE etrade etrade_warnings benchmark::benchmark)
