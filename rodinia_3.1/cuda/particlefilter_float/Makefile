#makefile

include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

all: naive float

naive: ex_particle_CUDA_naive_seq.cu getTime.o
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_60 ex_particle_CUDA_naive_seq.cu getTime.o -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu getTime.o
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_60 ex_particle_CUDA_float_seq.cu getTime.o -o particlefilter_float

getTime.o: getTime.cpp
	$(CC) -c getTime.cpp -o getTime.o

clean:
	rm particlefilter_naive particlefilter_float getTime.o
