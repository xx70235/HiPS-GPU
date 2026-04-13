# HipsGen CUDA - Linux Makefile
# GPU加速版本

CXX ?= g++
NVCC ?= $(shell command -v nvcc 2>/dev/null)

ifeq ($(strip $(NVCC)),)
NVCC := $(firstword $(wildcard /usr/local/cuda/bin/nvcc /usr/local/cuda-11.8/bin/nvcc /usr/local/cuda-11/bin/nvcc))
endif

ifeq ($(strip $(NVCC)),)
$(error nvcc not found; set NVCC=/path/to/nvcc)
endif

CXXFLAGS = -O2 -fopenmp -std=c++17 -Wall -fPIC -pthread -DUSE_CFITSIO
NVCCFLAGS = -O2 -Xcompiler -fopenmp -arch=sm_75 --compiler-options "-fPIC -std=c++17 -pthread -DUSE_CFITSIO"

SRC_DIR = src/cpp
OBJ_DIR = obj
BIN_DIR = bin

LIBS = -lwcs -lm -lpthread -lhealpix_cxx -lcfitsio
CUDA_LIBS = -lcudart -lcurand -lgomp

INCLUDES = -I$(SRC_DIR) -I/usr/include/wcslib

.PHONY: all clean dirs hipsgen test_fits_cutout test_cutout_planner test_index_merge_utils test_parallel_utils test_pipeline_policy test_tile_coord_cache

all: dirs hipsgen

dirs:
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

# C++ objects
$(OBJ_DIR)/fits_io.o: $(SRC_DIR)/fits_io.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/fits_hdu_utils.o: $(SRC_DIR)/fits_hdu_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/gpu_full_batch_plan.o: $(SRC_DIR)/gpu_full_batch_plan.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/tile_image_index.o: $(SRC_DIR)/tile_image_index.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/cutout_planner.o: $(SRC_DIR)/cutout_planner.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/index_merge_utils.o: $(SRC_DIR)/index_merge_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/parallel_utils.o: $(SRC_DIR)/parallel_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/pipeline_policy.o: $(SRC_DIR)/pipeline_policy.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/tile_coord_cache.o: $(SRC_DIR)/tile_coord_cache.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/coordinate_transform.o: $(SRC_DIR)/coordinate_transform.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/healpix_util.o: $(SRC_DIR)/healpix_util.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/hpx_finder.o: $(SRC_DIR)/hpx_finder.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/hips_tile_generator.o: $(SRC_DIR)/hips_tile_generator.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/fits_writer.o: $(SRC_DIR)/fits_writer.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/allsky_generator.o: $(SRC_DIR)/allsky_generator.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/properties_generator.o: $(SRC_DIR)/properties_generator.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/gpu_image_cache.o: $(SRC_DIR)/gpu_image_cache.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/gpu_batch_processor.o: $(SRC_DIR)/gpu_batch_processor.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/gpu_full_processor.o: $(SRC_DIR)/gpu_full_processor.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/bilinear_interpolation_cuda.o: $(SRC_DIR)/bilinear_interpolation_cuda.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/index_gpu_processor.o: $(SRC_DIR)/index_gpu_processor.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/hipsgen_cuda.o: $(SRC_DIR)/hipsgen_cuda.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# CUDA kernels
$(OBJ_DIR)/bilinear_interpolation_cuda_kernels.o: $(SRC_DIR)/bilinear_interpolation_cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/gpu_image_cache_kernels.o: $(SRC_DIR)/gpu_image_cache_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/gpu_batch_processor_kernels.o: $(SRC_DIR)/gpu_batch_processor_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/gpu_full_processor_kernels.o: $(SRC_DIR)/gpu_full_processor_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/index_gpu_kernels.o: $(SRC_DIR)/index_gpu_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/healpix_cuda_kernels.o: $(SRC_DIR)/healpix_cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Link
OBJS = $(OBJ_DIR)/fits_io.o \
       $(OBJ_DIR)/fits_hdu_utils.o \
       $(OBJ_DIR)/gpu_full_batch_plan.o \
       $(OBJ_DIR)/tile_image_index.o \
       $(OBJ_DIR)/cutout_planner.o \
       $(OBJ_DIR)/index_merge_utils.o \
       $(OBJ_DIR)/parallel_utils.o \
       $(OBJ_DIR)/pipeline_policy.o \
       $(OBJ_DIR)/tile_coord_cache.o \
       $(OBJ_DIR)/coordinate_transform.o \
       $(OBJ_DIR)/healpix_util.o \
       $(OBJ_DIR)/hpx_finder.o \
       $(OBJ_DIR)/hips_tile_generator.o \
       $(OBJ_DIR)/fits_writer.o \
       $(OBJ_DIR)/allsky_generator.o \
       $(OBJ_DIR)/properties_generator.o \
       $(OBJ_DIR)/gpu_image_cache.o \
       $(OBJ_DIR)/gpu_batch_processor.o \
       $(OBJ_DIR)/gpu_full_processor.o \
       $(OBJ_DIR)/bilinear_interpolation_cuda.o \
       $(OBJ_DIR)/bilinear_interpolation_cuda_kernels.o \
       $(OBJ_DIR)/gpu_image_cache_kernels.o \
       $(OBJ_DIR)/gpu_batch_processor_kernels.o \
       $(OBJ_DIR)/gpu_full_processor_kernels.o \
       $(OBJ_DIR)/index_gpu_processor.o \
       $(OBJ_DIR)/index_gpu_kernels.o \
       $(OBJ_DIR)/healpix_cuda_kernels.o \
       $(OBJ_DIR)/hipsgen_cuda.o

hipsgen: dirs $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(BIN_DIR)/hipsgen_cuda $(LIBS) $(CUDA_LIBS)
	@echo "Built $(BIN_DIR)/hipsgen_cuda"

$(BIN_DIR)/test_fits_cutout: $(SRC_DIR)/test_fits_cutout.cpp $(OBJ_DIR)/fits_io.o $(OBJ_DIR)/fits_hdu_utils.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(OBJ_DIR)/fits_io.o $(OBJ_DIR)/fits_hdu_utils.o -o $@ $(LIBS)

test_fits_cutout: dirs $(BIN_DIR)/test_fits_cutout

$(BIN_DIR)/test_cutout_planner: $(SRC_DIR)/test_cutout_planner.cpp $(OBJ_DIR)/cutout_planner.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(OBJ_DIR)/cutout_planner.o -o $@ $(LIBS)

test_cutout_planner: dirs $(BIN_DIR)/test_cutout_planner

$(BIN_DIR)/test_index_merge_utils: $(SRC_DIR)/test_index_merge_utils.cpp $(OBJ_DIR)/index_merge_utils.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(OBJ_DIR)/index_merge_utils.o -o $@ $(LIBS)

test_index_merge_utils: dirs $(BIN_DIR)/test_index_merge_utils

$(BIN_DIR)/test_parallel_utils: $(SRC_DIR)/test_parallel_utils.cpp $(OBJ_DIR)/parallel_utils.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(OBJ_DIR)/parallel_utils.o -o $@ $(LIBS)

test_parallel_utils: dirs $(BIN_DIR)/test_parallel_utils

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)


$(BIN_DIR)/test_pipeline_policy: $(SRC_DIR)/test_pipeline_policy.cpp $(OBJ_DIR)/pipeline_policy.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(OBJ_DIR)/pipeline_policy.o -o $@ $(LIBS)

test_pipeline_policy: dirs $(BIN_DIR)/test_pipeline_policy

$(BIN_DIR)/test_tile_coord_cache: $(SRC_DIR)/test_tile_coord_cache.cpp $(OBJ_DIR)/tile_coord_cache.o $(OBJ_DIR)/healpix_util.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(OBJ_DIR)/tile_coord_cache.o $(OBJ_DIR)/healpix_util.o -o $@ $(LIBS)

test_tile_coord_cache: dirs $(BIN_DIR)/test_tile_coord_cache
