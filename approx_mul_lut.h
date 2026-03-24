#ifndef APPROX_MUL_LUT_H_
#define APPROX_MUL_LUT_H_
#include <fstream>
#include <string>
#include <tensorflow/core/framework/op_kernel.h>
#include "lut/posit8e0.inl"
#include "lut/posit8e1.inl"
#ifdef GOOGLE_CUDA
#include <cuda_runtime_api.h>
#else
typedef unsigned long long cudaTextureObject_t;
#endif
class approx_mul_lut_base {
public:
  explicit approx_mul_lut_base(tensorflow::OpKernelConstruction *context)
      : mant_mul_lut_{0} {
    load_lut_binary(context);
  }
  virtual ~approx_mul_lut_base() = default;
  // same for both CPU and GPU
  auto load_lut_binary(tensorflow::OpKernelConstruction *context) -> void {
    auto mant_lut_file_name = std::string{};
    OP_REQUIRES_OK(context,
                   context->GetAttr("mant_mul_lut", &mant_lut_file_name));
    if (mant_lut_file_name.empty()) {
      std::cerr << "no mant lut file name given" << std::endl;
      exit(1);
    }
    if (mant_lut_file_name.find("POS8E0") != std::string::npos) {
      use_posit_lut_ = true;
      posit_es_ = 0;
    } else if (mant_lut_file_name.find("POS8E1") != std::string::npos) {
      use_posit_lut_ = true;
      posit_es_ = 1;
    } else {
      use_posit_lut_ = false;
      posit_es_ = -1;
    }

    if (use_posit_lut_) {
      mant_width_ = 8;
      a_shift_ = 0;
      b_shift_ = 0;
      mant_mask_ = 0;
      std::ifstream file(mant_lut_file_name, std::ios::in | std::ios::binary);
      if (file.fail()) {
        std::cerr << "posit lut file read failed" << std::endl;
        exit(1);
      }
      if (!file.is_open()) {
        std::cerr << "posit lut file open failed" << std::endl;
        exit(1);
      }
      posit_mul_lut_.resize(256 * 256);
      file.read(reinterpret_cast<char *>(posit_mul_lut_.data()),
                posit_mul_lut_.size() * sizeof(float));
      return;
    }
    unsigned start_delimiter = mant_lut_file_name.find_last_of("_");
    unsigned stop_deliminter = mant_lut_file_name.find_last_of(".");
    auto mant_width_str = mant_lut_file_name.substr(
        start_delimiter + 1, stop_deliminter - start_delimiter - 1);
    mant_width_ = std::stoi(mant_width_str);
    a_shift_ = 23 - mant_width_ * 2;
    b_shift_ = 23 - mant_width_;
    mant_mask_ = ((1 << mant_width_) - 1) << (23 - mant_width_);

    // open mant mul file
    std::ifstream file(mant_lut_file_name, std::ios::in | std::ios::binary);
    if (file.fail()) {
      std::cerr << "lut file read failed" << std::endl;
      exit(1);
    }
    if (!file.is_open()) {
      std::cerr << "lut file open failed" << std::endl;
      exit(1);
    }
    mant_mul_lut_.resize(uint32_t(pow(2, mant_width_ * 2)));
    file.read(reinterpret_cast<char *>(mant_mul_lut_.data()),
              mant_mul_lut_.size() * sizeof(uint32_t));
  }
  auto get_mant_mul_lut_text_() -> cudaTextureObject_t & {
    return mant_mul_lut_text_;
  }
  auto get_mant_mul_lut_() -> uint8_t * { return mant_mul_lut_cuda_; }
  auto get_use_posit_lut_() const -> bool { return use_posit_lut_; }
  auto get_posit_es_() const -> int { return posit_es_; }
  auto get_posit_mul_lut_text_() -> cudaTextureObject_t & {
    return posit_mul_lut_text_;
  }
  auto get_posit_mul_lut_() -> float * { return posit_mul_lut_cuda_; }
  auto get_posit_mul_lut_host_() -> const float * {
    return posit_mul_lut_.data();
  }
  auto get_mant_mask_() -> uint32_t { return mant_mask_; };
  auto get_a_shift_() -> uint8_t { return a_shift_; };
  auto get_b_shift_() -> uint8_t { return b_shift_; };
  auto get_mant_width_() -> uint8_t { return mant_width_; };

protected:
  std::vector<uint8_t> mant_mul_lut_;
  std::vector<float> posit_mul_lut_;
  uint8_t *mant_mul_lut_cuda_ = nullptr;
  float *posit_mul_lut_cuda_ = nullptr;
  cudaTextureObject_t mant_mul_lut_text_ = 0;
  cudaTextureObject_t posit_mul_lut_text_ = 0;
  std::string lut_file_name;
  uint8_t mant_width_;
  uint32_t mant_mask_;
  uint8_t a_shift_;
  uint8_t b_shift_;
  bool use_posit_lut_ = false;
  int posit_es_ = -1;
};
template <typename Device> class approx_mul_lut : public approx_mul_lut_base {
public:
  explicit approx_mul_lut(tensorflow::OpKernelConstruction *context);
  ~approx_mul_lut();
};

inline float posit_lut_mul_host(float a, float b, const float *lut) {
  uint8_t pa = posit8e1_from_float(a);
  uint8_t pb = posit8e1_from_float(b);
  uint32_t idx = (uint32_t(pa) << 8) | uint32_t(pb);
  return lut[idx];
}

inline float posit_lut_mul_host_es(float a, float b, const float *lut, int es) {
  uint8_t pa = es == 0 ? posit8e0_from_float(a) : posit8e1_from_float(a);
  uint8_t pb = es == 0 ? posit8e0_from_float(b) : posit8e1_from_float(b);
  uint32_t idx = (uint32_t(pa) << 8) | uint32_t(pb);
  return lut[idx];
}
#endif
