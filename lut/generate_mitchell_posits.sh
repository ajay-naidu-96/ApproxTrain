#!/usr/bin/env bash
# generate_mitchell_posits.sh — Generate Mitchell-approximated SoftPosit LUTs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTPOSIT_SRC="${SCRIPT_DIR}/../SoftPosit/source"
SOFTPOSIT_INC="${SOFTPOSIT_SRC}/include"
SOFTPOSIT_PLATFORM="${SCRIPT_DIR}/../SoftPosit/build/Linux-x86_64-GCC"
# Collect all SoftPosit .c files needed for p8, px1, and px2.
# Exclude p16 and p32 files to avoid compilation errors on macOS when treating C as C++.
SOFTPOSIT_SRCS=$(find "${SOFTPOSIT_SRC}" -name "*.c" ! -name ".*" ! -path "*/8086-SSE/*" ! -name "*p16*" ! -name "*p32*")

echo "=== Generating Mitchell SoftPosit LUTs (5-8 bits, es=1-2) ==="

# Pre-compile SoftPosit .c files to object files once to save time and avoid C/C++ mismatch.
echo "  Pre-compiling SoftPosit source..."
mkdir -p objs
# Compile each .c file to an object file using the C compiler.
# We exclude p16 and p32 files to keep it focused on the target bit widths.
# Use -w to suppress warnings during this large batch compile.
for src in ${SOFTPOSIT_SRCS}; do
    obj="objs/$(basename "${src}" .c).o"
    if [ ! -f "${obj}" ] || [ "${src}" -nt "${obj}" ]; then
        # Use C++ compiler for all files with -w to avoid complex language mismatch issues
        # that occur when compiling SoftPosit with modern standards on macOS.
        g++ -O2 -w -std=c++17 -I"${SOFTPOSIT_INC}" -I"${SOFTPOSIT_PLATFORM}" -c "${src}" -o "${obj}"
    fi
done

for N in 5 6 7 8; do
    for ES in 1 2; do
        echo "  Generating N=${N}, ES=${ES} with Mitchell..."
        # Link the C++ generator with all SoftPosit object files.
        g++ -O2 -std=c++17 "-DPOSIT_N=${N}" "-DPOSIT_ES=${ES}" -DMITCHELL_APPROX \
            -I"${SOFTPOSIT_INC}" \
            -I"${SOFTPOSIT_PLATFORM}" \
            lut_gen.cc \
            objs/*.o \
            -lm \
            -o lut_gen_tmp
        ./lut_gen_tmp
        rm -f lut_gen_tmp
    done
done

echo "=== Done ==="
ls -lh POS*MIT.bin
