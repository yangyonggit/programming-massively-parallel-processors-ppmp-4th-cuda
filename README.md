# Programming Massively Parallel Processors 4th – CUDA Examples & Exercises

## What this project is
CUDA examples following "Programming Massively Parallel Processors". Currently includes basic kernels, image filters, matrix ops with cuBLAS validation, and device query utility.

## Project layout
- `CMakeLists.txt` – builds all examples (CUDA 12+, CMake 3.20+, C++17)
- `include/` – shared headers (`cdiv.h` for ceiling division)
- `src/chapter02/` – `vectorAdd.cu`
- `src/chapter03/` – `rgb_to_gray.cu`, `blur.cu`, `matrixMul.cu`, `exercise_3_1.cu`, `exercise_3_2.cu`
- `src/chapter04/` – `query_device.cu`
- `assets/` – copied next to binaries after build (if present)

## Prerequisites
- CUDA Toolkit 11.x (nvcc available on PATH)
- CMake ≥ 3.20
- C++17 toolchain (MSVC/Clang/GCC); on Windows, MSVC 2022 works
- GPU with CUDA support

## Configure & build
```bash
# from repo root
cmake -B build -S .
cmake --build build --config Release
```

## Build & run on WSL (Ubuntu)
- Ensure NVIDIA drivers and CUDA are installed in WSL (CUDA Toolkit 12.x recommended) and `/usr/local/cuda/bin` is on `PATH`.
- Use GCC/Clang toolchain; CMake will detect `nvcc` automatically.

```bash
# inside WSL, from repo root
cmake -B build -S .
cmake --build build --config Release

# run examples
./build/bin/Release/chapter04/query_device    # list devices
./build/bin/Release/chapter03/exercise_3_2 1024
```

## Targets & outputs
- Chapter 02
  - `vectorAdd` → `build/bin/Release/chapter02/vectorAdd.exe`
- Chapter 03
  - `rgbToGray` → `build/bin/Release/chapter03/rgbToGray.exe`
  - `blurFilter` → `build/bin/Release/chapter03/blurFilter.exe`
  - `matrixMul` (cuBLAS SGEMM validation) → `build/bin/Release/chapter03/matrixMul.exe`
  - `exercise_3_1` (row vs column sweep multiply, cuBLAS compare) → `build/bin/Release/chapter03/exercise_3_1.exe`
  - `exercise_3_2` (matrix–vector multiply, cuBLAS SGEMV compare) → `build/bin/Release/chapter03/exercise_3_2.exe`
- Chapter 04
  - `query_device` (device properties dump) → `build/bin/Release/chapter04/query_device.exe`

Assets (if present) are auto-copied beside the binaries.

## Running examples (Windows PowerShell)
```powershell
# run with default sizes
./build/bin/Release/chapter02/vectorAdd.exe
./build/bin/Release/chapter03/rgbToGray.exe
./build/bin/Release/chapter03/blurFilter.exe
./build/bin/Release/chapter03/matrixMul.exe
./build/bin/Release/chapter03/exercise_3_1.exe
./build/bin/Release/chapter03/exercise_3_2.exe 1024   # optional size argument
./build/bin/Release/chapter04/query_device.exe
```

## Implementation notes
- Timing uses CUDA events; cuBLAS paths include a warm-up call to remove first-call overhead.
- Matrix kernels are row-major; cuBLAS expects column-major, so validation transposes host matrices before GEMM/GEMV.
- `exercise_3_1` compares row-sweep and column-sweep kernels against cuBLAS SGEMM.
- `exercise_3_2` computes `C = A * b` (matrix–vector); validated with cuBLAS SGEMV.

## Next steps
- Add more chapter04 samples (occupancy, bandwidth tests).
- Extend performance reporting (GFLOP/s, bandwidth) for existing exercises.
