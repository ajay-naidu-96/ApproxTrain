#ifndef POSIT8E1_INL_
#define POSIT8E1_INL_

#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define POSIT_HD __host__ __device__
#else
#define POSIT_HD
#endif

// 8-bit posit with es=1 (useed = 4).
// This is a lightweight implementation intended for LUT generation and runtime
// quantization. It is not a full SoftPosit replacement.

POSIT_HD static inline uint8_t posit8e1_from_float(float x) {
  if (x == 0.0f) {
    return 0x00;
  }
  if (!isfinite(x)) {
    return 0x80; // NaR
  }

  bool sign = x < 0.0f;
  double ax = sign ? -(double)x : (double)x;

  // Determine regime k such that useed^k <= ax < useed^(k+1), useed = 4.
  const double useed = 4.0;
  int k = 0;
  double y = ax;
  while (y >= useed) {
    y /= useed;
    k++;
    if (k > 10) break;
  }
  while (y < 1.0) {
    y *= useed;
    k--;
    if (k < -10) break;
  }

  // Exponent (es=1): 0 or 1.
  int exp = 0;
  if (y >= 2.0) {
    exp = 1;
    y /= 2.0;
  }

  // Fraction in [0,1)
  double frac = y - 1.0;

  int regime_bits = (k >= 0) ? (k + 2) : (-k + 1); // includes terminating bit
  int frac_bits = 8 - 1 - regime_bits - 1; // sign + regime + es=1
  if (frac_bits < 0) {
    frac_bits = 0;
  }

  uint8_t bits = 0;
  int idx = 7; // start at MSB

  // Sign bit is 0 for magnitude representation
  idx--;

  // Regime
  if (k >= 0) {
    for (int i = 0; i < k + 1 && idx >= 0; i++, idx--) {
      bits |= (uint8_t)(1u << idx);
    }
    if (idx >= 0) {
      // terminating 0 bit (already 0)
      idx--;
    }
  } else {
    for (int i = 0; i < -k && idx >= 0; i++, idx--) {
      // zeros, nothing to set
    }
    if (idx >= 0) {
      // terminating 1 bit
      bits |= (uint8_t)(1u << idx);
      idx--;
    }
  }

  // Exponent (1 bit)
  if (idx >= 0) {
    if (exp) {
      bits |= (uint8_t)(1u << idx);
    }
    idx--;
  }

  // Fraction
  uint32_t frac_int = 0;
  if (frac_bits > 0) {
    double scaled = frac * (double)(1u << frac_bits);
    frac_int = (uint32_t)(scaled + 0.5); // round to nearest
    if (frac_int >= (1u << frac_bits)) {
      // Clamp on overflow (simpler than carry handling)
      frac_int = (1u << frac_bits) - 1;
    }
  }
  for (int i = frac_bits - 1; i >= 0 && idx >= 0; i--, idx--) {
    if (frac_int & (1u << i)) {
      bits |= (uint8_t)(1u << idx);
    }
  }

  if (sign) {
    bits = (uint8_t)((~bits + 1) & 0xFF);
  }

  return bits;
}

POSIT_HD static inline float posit8e1_to_float(uint8_t p) {
  if (p == 0x00) {
    return 0.0f;
  }
  if (p == 0x80) {
    return NAN;
  }

  bool sign = (p & 0x80) != 0;
  uint8_t ui = p;
  if (sign) {
    ui = (uint8_t)((~ui + 1) & 0xFF);
  }

  int idx = 6; // start after sign bit
  int regime_sign = (ui >> idx) & 0x1;
  int run = 0;
  while (idx >= 0 && (((ui >> idx) & 0x1) == regime_sign)) {
    run++;
    idx--;
  }

  int k = regime_sign ? (run - 1) : -run;

  int exp = 0;
  if (idx >= 0) {
    exp = (ui >> idx) & 0x1;
    idx--;
  }

  double frac = 0.0;
  double scale = 0.5;
  while (idx >= 0) {
    if ((ui >> idx) & 0x1) {
      frac += scale;
    }
    scale *= 0.5;
    idx--;
  }

  const double useed = 4.0;
  double value = pow(useed, k) * pow(2.0, exp) * (1.0 + frac);
  if (sign) {
    value = -value;
  }
  return (float)value;
}

#endif // POSIT8E1_INL_
