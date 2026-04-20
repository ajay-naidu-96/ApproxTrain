# ApproxTrain: Extension Fast Simulation of Approximate Multipliers for Transformer Training and Inference.

Mulipliers simulated: 
1. Native FP32
2. 8-bit Posit (pos8e1)
3. 8-bit Posit (pos8e0)
4. Mitchell Logarithm-based multiplier (mit_7)
5. Minimally Biased Multiplier (mbm_7)
6. 3 bit variants of 4 and 5. 

**Author:** Ajay Gopi

## Overview
This repository is a fork of the original [ApproxTrain framework](https://github.com/AaronJing/ApproxTrain). ApproxTrain is an open-source framework for evaluating DNN training and inference using simulated approximate floating-point (FP) multipliers.

**Author's Contributions:**
- Upgraded the deep learning backend to utilize the latest stable TensorFlow variants.
- Expanded evaluation capabilities to support deep modern architectures, specifically **Decoder-only Transformers** on language modeling tasks.
- Integrated the official **SoftPosit C Library** to emulate exact, hardware-accurate **8-bit Posit Arithmetic** (both `pos8e0` fixed and `pos8e1` dynamic precision).
- **Recent Progress:** Added support for **Mitchell-approximated SoftPosit multipliers** (`posNeES_mit`), combining logarithmic approximation with Posit-based formats for enhanced efficiency.
- **Automation:** Developed **multi-GPU batch training scripts** to evaluate diverse approximate configurations (bits 5-8, es 1-2) at scale.

## Directory Hierarchy
```text
ApproxTrain/
├── ammha/                            # Custom Multi-Head Attention TF ops
├── checkpoints/                      # Saved Transformer weights for each run
├── cuda/                             # Core C++/CUDA kernels for approximate math
├── data/                             # Raw text and structured datasets (Shakespeare)
├── docs/                             # Project reports and reference papers
├── figures/                          # Generated dataset plots and diagrams
├── logs/ & train_logs/               # TensorBoard logging output and screenshots
├── lut/                              # C++ headers and bash macros to generate LUTs
│   ├── lut_gen.sh
│   └── posit8e1.inl                  # 8-bit Posit mathematical implementations
├── python/                           # Core utilities and data pipelines
├── Makefile                          # Build commands for the Custom TF ops
├── approx_mul_lut.h                  # Core header connecting LUTs to TensorFlow
├── benchmark.py & evaluate_model.py  # Inference testing scripts
├── compare_baselines.py              # Evaluation script to compare runs
├── config.py                         # Hyperparameters and model configurations
├── convam.cc, denseam.cc, matmulam.cc# Custom TensorFlow Operator C++ Bindings
├── run_posit_train_jobs.sh           # Main bash entrypoint for bulk training loops
├── shakespeare_data.py               # Dataset processing and tokenization logic
├── train_transformer.py              # Core training loop for the Transformer
├── transformer_model.py              # Definition of the Decoder-only architecture
├── run_mitchell_batch.sh             # Batch training script for Mitchell-Posit experiments
└── docs/                             # Project reports, posters, and reference papers
    └── Gopi_Capstone_Poster.pdf      # Visual overview of research and results
```

## Running the Project

### 1. Environment & Dependencies
The framework requires TensorFlow 2.3.0, CUDA Toolkit 10.1, cuDNN 7.6.5, g++ 8.4, and Python 3.6 - 3.8.
```bash
# Check compiler and GPU mapping
g++ -v
python3 -c 'import tensorflow as tf; print(tf.__version__)'
nvidia-smi

# Install required python packages
pip3 install --user tensorflow-datasets
```
*Note: A conda environment file `env_remote.yml` is provided for automatic environment setup.*

### 2. Data Preparation
The repository uses a character-level Shakespeare dataset. The data is automatically downloaded, cleaned, and tokenized by `shakespeare_data.py` upon initial run. The target vocabulary is ~65 unique characters.
```bash
python3 shakespeare_data.py
```
*(See `figures/char_frequencies.png` for a visualization of the dataset distribution).*

### 3. Training and Testing Demo Scripts

**To generate the C++ Multiplier Lookup Tables (LUTs):**
The `AMSIMULATOR` backend relies on binary Lookup Table (LUT) files (`.bin`) to emulate the approximate math natively in hardware. You generate all supported LUTs simultaneously by calculating the matrix results and saving them via the generation script:
```bash
cd lut
./lut_gen.sh
cd ..
```

**To compile the Custom TensorFlow C++ Operations:**
You must compile the custom TF operations (`convam`, `denseam`, `matmulam`) with the `AMSIMULATOR` flag so they know to read the LUTs from disk. 
**Note: You only need to compile the operations ONCE.** Because the C++ ops read the `.bin` files dynamically at runtime based on your python `--multiplier` argument, you do not need to recompile the C++ package if you decide to regenerate new LUTs or test out different multiplier modes.
```bash
make clean && make convam MULTIPLIER=AMSIMULATOR && make denseam MULTIPLIER=AMSIMULATOR && make matmulam MULTIPLIER=AMSIMULATOR
```

**To train the Transformer:**
You can train a single baseline natively using FP32, or using the 8-bit Posit emulator:
```bash
python3 train_transformer.py --multiplier="fp32" --epochs=50
python3 train_transformer.py --multiplier="pos8e1" --epochs=50
```
Alternatively, run the automated batch scripts to train multiple configurations sequentially:
```bash
./run_posit_train_jobs.sh    # Standard 8-bit Posit baselines
./run_mitchell_batch.sh      # Mitchell-approximated SoftPosit variants (5-8 bits)
```

**To evaluate the model (Inference Demo):**
```bash
python3 evaluate_model.py --multiplier="pos8e1"
python3 generate_text.py --multiplier="pos8e1"
```

## Key Project Results
The table below summarizes the final validation metrics (Cross-Entropy Loss and Perplexity) attained on the Shakespeare dataset run for 50 epochs on the base architecture defined.

| Multiplier Type         | Precision | Val Loss | Val Perplexity | Training Stability |
|-------------------------|-----------|----------|----------------|--------------------|
| **Native (Baseline)**   | FP32      | ~1.38    | ~3.98         | High               |
| **Posit (pos8e1)**      | 8-bit  (es=1)   |     ~2.36| ~10.63         | Low               |
| **MBM (mbm_7)**         | FP Emul   | ~1.86    | ~6.45        | Medium             |
| **Mitchell (mit_7)**    | FP Emul   | ~1.86   | ~6.43       | Medium                |


## Reference Information and Acknowledgments
This project builds upon the original `ApproxTrain` framework and references state-of-the-art approximations for deep learning environments.

* **Original ApproxTrain Framework & Minimally Biased Multipliers:** [ApproxTrain: Fast Simulation of Approximate Multipliers for DNN Training and Inference](https://ieeexplore.ieee.org/document/3253045) (Gong et al., 2023, IEEE TCAD).
* **SoftPosit Library:** [SoftPosit Posit Arithmetic Package](https://gitlab.com/cerlane/SoftPosit) by S. H. Leong (Cerlane) and John Gustafson.
* **Research Poster:** [Capstone Project Poster (Visual Overview)](docs/Gopi_Capstone_Poster.pdf).
* **Posit Arithmetic Training (GANs):** "Posit Arithmetic for the Training and Deployment of Generative Adversarial Networks".
* **MINOTAUR Accelerator:** "MINOTAUR: A Posit-Based 0.420–50-TOPS/W Edge Transformer Inference and Training Accelerator".
* **Mitchell Logarithm-based multiplier:** "Computer Multiplication and Division Using Binary Logarithms".

Some C++ TF-custom operation code snippets and Makefile logic were adapted from the official [tensorflow/custom-op](https://github.com/tensorflow/custom-op) guides.
