#!/usr/bin/env bash
# lut_gen.sh — Generate all LUT binary files for ApproxTrain.
#
# For non-posit multipliers: standard g++ only.
# For posit8 multipliers: compiles with SoftPosit sources for accurate
# round-to-nearest posit8 conversions (replaces the naive .inl loops).
#
# Usage: cd lut && bash lut_gen.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTPOSIT_SRC="${SCRIPT_DIR}/../SoftPosit/source"
SOFTPOSIT_INC="${SOFTPOSIT_SRC}/include"
SOFTPOSIT_PLATFORM="${SCRIPT_DIR}/../SoftPosit/build/Linux-x86_64-GCC"

# Collect all SoftPosit .c files (excluding the 8086-SSE SIMD folder
# which is optional and platform-specific, and ignoring macOS dotfiles).
SOFTPOSIT_SRCS=$(find "${SOFTPOSIT_SRC}" -name "*.c" ! -name ".*" ! -path "*/8086-SSE/*")

NON_POSIT_MULTIPLIERS=(
    "FMBM16_MULTIPLIER"
    "FMBM14_MULTIPLIER"
    "FMBM12_MULTIPLIER"
    "FMBM10_MULTIPLIER"
    "MITCHEL16_MULTIPLIER"
    "MITCHEL14_MULTIPLIER"
    "MITCHEL12_MULTIPLIER"
    "MITCHEL10_MULTIPLIER"
    "BFLOAT"
    "ZEROS"
)

POSIT_MULTIPLIERS=(
    "POSIT8E0"
    "POSIT8E1"
)

echo "=== Building non-posit LUTs ==="
for M in "${NON_POSIT_MULTIPLIERS[@]}"; do
    echo "  Generating LUT for -D${M} ..."
    g++ -O2 "-D${M}" \
        -I"${SOFTPOSIT_INC}" \
        -I"${SOFTPOSIT_PLATFORM}" \
        lut_gen.cc -o lut_gen_tmp
    ./lut_gen_tmp
    rm -f lut_gen_tmp
done

echo ""
echo "=== Building posit LUTs (with SoftPosit) ==="
for M in "${POSIT_MULTIPLIERS[@]}"; do
    echo "  Generating LUT for -D${M} (SoftPosit) ..."
    g++ -O2 "-D${M}" \
        -I"${SOFTPOSIT_INC}" \
        -I"${SOFTPOSIT_PLATFORM}" \
        lut_gen.cc \
        ${SOFTPOSIT_SRCS} \
        -lm \
        -o lut_gen_tmp
    ./lut_gen_tmp
    rm -f lut_gen_tmp
    echo "    Done."
done

echo ""
echo "=== Generated LUT files ==="
ls -lh ./*.bin 2>/dev/null || echo "(none found)"
