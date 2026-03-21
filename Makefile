NVCC = nvcc
CUDA_PATH ?= /usr/local/cuda

NVCC_FLAGS = -O3 -arch=sm_86 -std=c++17 -Xcompiler -Wall
INCLUDES = -I$(CUDA_PATH)/include -Isrc
LIBS = -L$(CUDA_PATH)/lib64 -lcusparse -lcublas -lcusolver -llapack -lblas

SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data

SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/naive_lanczos.cu \
          $(SRC_DIR)/dgks_lanczos.cu $(SRC_DIR)/irlm_lanczos.cu \
          $(SRC_DIR)/cast_kernels.cu
HEADERS = $(SRC_DIR)/lanczos_types.cuh $(SRC_DIR)/lanczos_context.cuh \
          $(SRC_DIR)/lanczos_ops.cuh $(SRC_DIR)/tridiag.cuh
TARGET = $(BUILD_DIR)/lanczos_bench

.PHONY: all clean run bench ref plot

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(SOURCES) $(LIBS)

run: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(BUILD_DIR)/lanczos_bench --n 5000 --k 15 --eigs 20 --iters 1000 --freq 5 --outdir $(DATA_DIR)

bench: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(BUILD_DIR)/lanczos_bench --n 10000 --k 20 --eigs 50 --iters 1000 --freq 5 --outdir $(DATA_DIR)

ref:
	python3 benchmark/scipy_reference.py --eigs 20 --outdir $(DATA_DIR)

plot:
	python3 benchmark/plot_results.py --datadir $(DATA_DIR)

clean:
	rm -rf $(BUILD_DIR) $(DATA_DIR)/*.csv $(DATA_DIR)/*.png $(DATA_DIR)/*.pdf $(DATA_DIR)/*.mtx
