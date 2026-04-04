CC       = gcc
CXX      = g++
# Default flags
CXXFLAGS += -g -Wall -O2 -std=c++17 -fPIC
LDFLAGS  +=

# SoftPosit — used for accurate posit8 es=0/es=1 conversions at runtime.
# Platform header (platform.h) lives in the build directory.
SOFTPOSIT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))SoftPosit
SOFTPOSIT_INC  := $(SOFTPOSIT_ROOT)/source/include
SOFTPOSIT_PLAT := $(SOFTPOSIT_ROOT)/build/Linux-x86_64-GCC
# Collect all SoftPosit .c sources (exclude the optional 8086-SSE SIMD folder and macOS dotfiles).
SOFTPOSIT_SRCS := $(shell find $(SOFTPOSIT_ROOT)/source -name '*.c' ! -name '.*' ! -path '*/8086-SSE/*')


# Handle Multiplier Flag
ifeq ($(MULTIPLIER),AMSIMULATOR)
    CPPFLAGS += -DAMSIMULATOR
endif

# Get TF flags
TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Add TF flags to CXXFLAGS (for C++)
CXXFLAGS += $(TF_CFLAGS)
LDFLAGS += $(TF_LFLAGS)

# OS Detection
UNAME_S := $(shell uname -s)

# Targets
CONV_BINARY = convam.so
DENSE_BINARY = denseam.so
MATMUL_BINARY = matmulam.so

# Base Sources
CONV_SRCS = convam.cc
DENSE_SRCS = denseam.cc
MATMUL_SRCS = matmulam.cc

# Platform specific configuration
ifeq ($(UNAME_S),Darwin)
    # Mac / CPU Only
    CPPFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
    # On Mac, we might need undefined dynamic_lookup for python symbols if not linking properly
    LDFLAGS += -undefined dynamic_lookup
    # No SoftPosit on Mac (training runs on the Linux server)
    
    # Define objects (just replace .cc with .o)
    CONV_OBJS = $(CONV_SRCS:.cc=.o)
    DENSE_OBJS = $(DENSE_SRCS:.cc=.o)
    MATMUL_OBJS = $(MATMUL_SRCS:.cc=.o)

else
    # Linux / CUDA assumed (or check for nvcc)
    NVCC := $(shell which nvcc 2>/dev/null)
    
    ifneq ($(NVCC),)
        # CUDA FOUND
        CPPFLAGS += -DGOOGLE_CUDA=1
        CUDA_ROOT ?= /usr/local/cuda
        ifneq ($(wildcard $(CUDA_ROOT)/lib64),)
            CUDA_LIB ?= $(CUDA_ROOT)/lib64
        else
            CUDA_LIB ?= $(CUDA_ROOT)/lib
        endif
        
        # Accurate posit8 runtime quantization via SoftPosit.
        CXXFLAGS += -I$(SOFTPOSIT_INC) -I$(SOFTPOSIT_PLAT)
        CUDA_CFLAGS += -I$(SOFTPOSIT_INC) -I$(SOFTPOSIT_PLAT)
        
        # Add TF include paths to CUDA flags
        CUDA_CFLAGS += -g -std=c++17 -Xcompiler -Wall -Xcompiler -fPIC --expt-relaxed-constexpr -ccbin $(CXX) $(TF_CFLAGS) -I. -I.
        CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart
        
        # Add CUDA sources
        CONV_CUDA_SRCS = cuda/cuda_kernel.cu cuda/approx_mul_lut.cu
        DENSE_CUDA_SRCS = cuda/denseam_kernel.cu cuda/approx_mul_lut.cu
        MATMUL_CUDA_SRCS = cuda/matmulam_kernel.cu cuda/approx_mul_lut.cu

        # Define Objects
        CONV_OBJS = $(CONV_SRCS:.cc=.o) $(CONV_CUDA_SRCS:.cu=.o)
        DENSE_OBJS = $(DENSE_SRCS:.cc=.o) $(DENSE_CUDA_SRCS:.cu=.o)
        MATMUL_OBJS = $(MATMUL_SRCS:.cc=.o) $(MATMUL_CUDA_SRCS:.cu=.o)
        
        # NVCC Compilation Rule
        %.o: %.cu
		$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -c $< -o $@
    endif
endif

# Generic C++ Compilation Rule
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

.PHONY: all clean test convam denseam matmulam convam_gpu.so denseam_gpu.so matmulam_gpu.so

all: $(CONV_BINARY) $(DENSE_BINARY) $(MATMUL_BINARY)

# Aliases
convam: $(CONV_BINARY)
	@:
denseam: $(DENSE_BINARY)
	@:
matmulam: $(MATMUL_BINARY)
	@:

# Compatibility Aliases for old scripts
convam_gpu.so: $(CONV_BINARY)
	@cp $(CONV_BINARY) $@
denseam_gpu.so: $(DENSE_BINARY)
	@cp $(DENSE_BINARY) $@
matmulam_gpu.so: $(MATMUL_BINARY)
	@cp $(MATMUL_BINARY) $@

$(CONV_BINARY): $(CONV_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CXX) $(CXXFLAGS) -shared $(CONV_OBJS) $(LDFLAGS) -o $@
else
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -shared $(CONV_OBJS) $(SOFTPOSIT_SRCS) $(LDFLAGS) $(CUDA_LDFLAGS) -lm -o $@
endif

$(DENSE_BINARY): $(DENSE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CXX) $(CXXFLAGS) -shared $(DENSE_OBJS) $(LDFLAGS) -o $@
else
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -shared $(DENSE_OBJS) $(SOFTPOSIT_SRCS) $(LDFLAGS) $(CUDA_LDFLAGS) -lm -o $@
endif

$(MATMUL_BINARY): $(MATMUL_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CXX) $(CXXFLAGS) -shared $(MATMUL_OBJS) $(LDFLAGS) -o $@
else
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -shared $(MATMUL_OBJS) $(SOFTPOSIT_SRCS) $(LDFLAGS) $(CUDA_LDFLAGS) -lm -o $@
endif

clean:
	rm -f *.o *.so cuda/*.o

test: all
	python3 mnist_example.py
