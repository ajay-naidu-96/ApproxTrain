#ifndef POSIT8E0_INL_
#define POSIT8E0_INL_

#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define POSIT_HD __host__ __device__
#else
#define POSIT_HD
#endif

// 8-bit posit with es=0 (useed = 2).

POSIT_HD static inline uint8_t posit8e0_from_float(float x) {
  if (x == 0.0f) {
    return 0x00;
  }
  if (!isfinite(x)) {
    return 0x80;
  }

  bool sign = x < 0.0f;
  double ax = sign ? -(double)x : (double)x;

  const double useed = 2.0;
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

  double frac = y - 1.0;

  int regime_bits = (k >= 0) ? (k + 2) : (-k + 1);
  int frac_bits = 8 - 1 - regime_bits;
  if (frac_bits < 0) {
    frac_bits = 0;
  }

  uint8_t bits = 0;
  int idx = 7;
  idx--;

  if (k >= 0) {
    for (int i = 0; i < k + 1 && idx >= 0; i++, idx--) {
      bits |= (uint8_t)(1u << idx);
    }
    if (idx >= 0) {
      idx--;
    }
  } else {
    for (int i = 0; i < -k && idx >= 0; i++, idx--) {
    }
    if (idx >= 0) {
      bits |= (uint8_t)(1u << idx);
      idx--;
    }
  }

  uint32_t frac_int = 0;
  if (frac_bits > 0) {
    double scaled = frac * (double)(1u << frac_bits);
    frac_int = (uint32_t)(scaled + 0.5);
    if (frac_int >= (1u << frac_bits)) {
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

POSIT_HD static inline float posit8e0_to_float(uint8_t p) {
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

  int idx = 6;
  int regime_sign = (ui >> idx) & 0x1;
  int run = 0;
  while (idx >= 0 && (((ui >> idx) & 0x1) == regime_sign)) {
    run++;
    idx--;
  }

  int k = regime_sign ? (run - 1) : -run;

  double frac = 0.0;
  double scale = 0.5;
  while (idx >= 0) {
    if ((ui >> idx) & 0x1) {
      frac += scale;
    }
    scale *= 0.5;
    idx--;
  }

  const double useed = 2.0;
  double value = pow(useed, k) * (1.0 + frac);
  if (sign) {
    value = -value;
  }
  return (float)value;
}

#endif // POSIT8E0_INL_
