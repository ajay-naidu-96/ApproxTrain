// lut_gen.cc — LUT generator for ApproxTrain approximate multipliers.
//
// Compile with SoftPosit for posit8 LUTs (handled by lut_gen.sh):
//   g++ -O2 -DPOSIT8E1 \
//       -I../SoftPosit/source/include \
//       -I../SoftPosit/build/Linux-x86_64-GCC \
//       lut_gen.cc \
//       $(find ../SoftPosit/source -name "*.c" ! -path "*/8086-SSE/*") \
//       -o lut_gen && ./lut_gen

#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <bitset>
#include <string>
#include <cmath>
#include "posit8e0.inl"
#include "posit8e1.inl"
#include "posit8e2.inl"
void floatToBinary(float f, std::string& str)
{

    union { float f; uint32_t i; } u;
    u.f = f;
    str.clear();

    for (int i = 0; i < 32; i++)
    {
        if (u.i % 2)  str.push_back('1');
        else str.push_back('0');
        u.i >>= 1;
    }

    // Reverse the string since now it's backwards
    std::string temp(str.rbegin(), str.rend());
    str = temp;
}
#ifdef FMBM16_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast16((a),(b));
    #include "FPmultMBM_fast16.inl"
    #define MANTISSA_BITWIDTH 7
    std::string lut_save = "MBM_7.bin";
#elif FMBM14_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast14((a),(b));
    #include "FPmultMBM_fast14.inl"
    #define MANTISSA_BITWIDTH 5
    std::string lut_save = "MBM_5.bin";
#elif FMBM12_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast12((a),(b));
    #include "FPmultMBM_fast12.inl"
    #define MANTISSA_BITWIDTH 3
    std::string lut_save = "MBM_3.bin";
#elif FMBM10_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast10((a),(b));
    #include "FPmultMBM_fast10.inl"
    #define MANTISSA_BITWIDTH 1
    std::string lut_save = "MBM_1.bin";
#elif MITCHEL16_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast16((a),(b));
    #include "Mitchell_16.inl"
    #define MANTISSA_BITWIDTH 7
    std::string lut_save = "MIT_7.bin";
#elif MITCHEL14_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast14((a),(b));
    #include "Mitchell_14.inl"
    #define MANTISSA_BITWIDTH 5
    std::string lut_save = "MIT_5.bin";
#elif MITCHEL12_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast12((a),(b));
    #include "Mitchell_12.inl"
    #define MANTISSA_BITWIDTH 3
    std::string lut_save = "MIT_3.bin";
#elif MITCHEL10_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast10((a),(b));
    #include "Mitchell_10.inl"
    #define MANTISSA_BITWIDTH 1
    std::string lut_save = "MIT_1.bin";
#elif BFLOAT
    #define MULTIPLY(a,b) bfloat16mul((a),(b));
    #include "bfloat.inl"
    #define MANTISSA_BITWIDTH 7
    std::string lut_save = "ACC_7.bin";
#elif ZEROS
    #define MULTIPLY(a,b) 0;
    #define MANTISSA_BITWIDTH 7
    std::string lut_save = "ZEROS_7.bin";
#elif defined(POSIT_N)
    #define POSIT_LUT
    #ifndef POSIT_ES
      #define POSIT_ES 1
    #endif
    #ifdef MITCHELL_APPROX
      #include "Mitchell_16.inl"
      #define MULTIPLY(a,b) FPmultMitch_fast16((a),(b))
      std::string lut_save = "POS" + std::to_string(POSIT_N) + "E" + std::to_string(POSIT_ES) + "_MIT.bin";
    #else
      #define MULTIPLY(a,b) ((a)*(b))
      std::string lut_save = "POS" + std::to_string(POSIT_N) + "E" + std::to_string(POSIT_ES) + ".bin";
    #endif
#elif POSIT8E1
    #define POSIT_LUT
    #define POSIT_N 8
    #define POSIT_ES 1
    #define MULTIPLY(a,b) ((a)*(b))
    std::string lut_save = "POS8E1_8.bin";
#elif POSIT8E0
    #define POSIT_LUT
    #define POSIT_N 8
    #define POSIT_ES 0
    #define MULTIPLY(a,b) ((a)*(b))
    std::string lut_save = "POS8E0_8.bin";
#elif POSIT8E2
    #define POSIT_LUT
    #define POSIT_N 8
    #define POSIT_ES 2
    #define MULTIPLY(a,b) ((a)*(b))
    std::string lut_save = "POS8E2_8.bin";
#endif

#define EMPTYFP32 0x00000000
//#define SIGN_MASK_ 0x80000000
#define EXPONENT127 0x3f800000
#define EXPONENT_MASK_ 0x7f800000
#define MANTISSA_MASK_ ((uint32_t(pow(2,MANTISSA_BITWIDTH))-1) << (23-MANTISSA_BITWIDTH))
// implementation for approximate mantissa multiplications lookup table generation
int main(){
#ifdef POSIT_LUT
    // POSIT LUT: maps (positN, positN) -> float (rounded via posit)
    int nbits = POSIT_N;
    int es = POSIT_ES;
    uint32_t size = 1 << nbits;
    std::vector<float> lut;
    lut.resize(size * size);
    
    for (uint32_t a = 0; a < size; ++a) {
        for (uint32_t b = 0; b < size; ++b) {
            float va, vb;
            
            // Generic approach using PX1/PX2
            double da, db;
            if (es == 1) {
                posit_1_t pa; pa.v = ((uint32_t)a) << (32 - nbits);
                posit_1_t pb; pb.v = ((uint32_t)b) << (32 - nbits);
                da = convertPX1ToDouble(pa);
                db = convertPX1ToDouble(pb);
            } else if (es == 2) {
                posit_2_t pa; pa.v = ((uint32_t)a) << (32 - nbits);
                posit_2_t pb; pb.v = ((uint32_t)b) << (32 - nbits);
                da = convertPX2ToDouble(pa);
                db = convertPX2ToDouble(pb);
            } else { // es = 0
                va = posit8e0_to_float(static_cast<uint8_t>(a << (8-nbits)));
                vb = posit8e0_to_float(static_cast<uint8_t>(b << (8-nbits)));
                da = va; db = vb;
            }
            
            if (es != 0) {
                va = (float)da;
                vb = (float)db;
            }

            float prod = MULTIPLY(va, vb);
            
            // Prevent NaNs in LUT by checking for Inf/NaN from multiplier
            if (std::isnan(prod) || std::isinf(prod)) {
                // Clip to a very large but finite value
                if (prod > 0) prod = 1e30f; 
                else prod = -1e30f;
            }

            float out;
            if (es == 1) {
                posit_1_t pc = convertDoubleToPX1((double)prod, nbits);
                out = (float)convertPX1ToDouble(pc);
            } else if (es == 2) {
                posit_2_t pc = convertDoubleToPX2((double)prod, nbits);
                out = (float)convertPX2ToDouble(pc);
            } else {
                uint8_t pc_bits = posit8e0_from_float(prod);
                out = posit8e0_to_float(pc_bits);
            }
            
            // Final safety check: if 'out' is still NaN (NaR), force to 0 or max representable
            if (std::isnan(out)) out = 0.0f; 

            lut[(a << nbits) | b] = out;
        }
    }
    char *lut_save_name = &lut_save[0];
    FILE *f = fopen(lut_save_name, "wb");
    fwrite(lut.data(), sizeof(float), lut.size(), f);
    fclose(f);
    return 0;
#else
    // create a and b
    float a = 0;
    float b = 0;
    // cast to uint32_t
    uint32_t  at = *(uint32_t *)&a;
	uint32_t  bt = *(uint32_t *)&b;
    // FP32 with bits set to all zeros
    at = at & EMPTYFP32;
    bt = bt & EMPTYFP32;
    // set sign to 0 or 1
    // set exponents A B C (output of A*B) should be normal case
    // 0b0011 1111 1000 0000 0000 0000 0000 0000 Biased exponent = 127
    at = at | EXPONENT127;
    bt = bt | EXPONENT127;



    char *lut_save_name = &lut_save[0];
    FILE *f = fopen(lut_save_name, "wb");
    for(uint32_t i = 0; i < uint32_t(pow(2,MANTISSA_BITWIDTH)); ++i){
        for(uint32_t j = 0; j < uint32_t(pow(2,MANTISSA_BITWIDTH)); ++j){
            uint32_t newat = at | (i<<(23-MANTISSA_BITWIDTH));
            uint32_t newbt = bt | (j<<(23-MANTISSA_BITWIDTH));
            float newa = *(float*)&newat;
            float newb = *(float*)&newbt;
            float c = MULTIPLY(newa, newb);
            uint32_t ct = *(uint32_t *)&c;
            uint8_t MANTISSA = (ct & MANTISSA_MASK_) >> (23-MANTISSA_BITWIDTH);
            uint32_t c_exp = ct & EXPONENT_MASK_;
            uint32_t un_normalized_exp = ((EXPONENT127>>23) + (EXPONENT127>>23) - 127)<<23;
            uint8_t carry = 0;
            if(un_normalized_exp < c_exp)
                carry = 0x80;
            uint8_t result = carry | MANTISSA;
            fwrite(&result, sizeof(uint8_t), 1, f);
        }
    }

    fclose(f);
    return 0;
#endif
}
