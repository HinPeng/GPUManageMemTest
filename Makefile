CUDA_PATH		?= /usr/local

HOST_COMPILER	?= g++
NVCC			:= $(CUDA_PATH)/cuda-9.0/bin/nvcc -ccbin $(HOST_COMPILER)

INCLUDES	:= -I$(CUDA_PATH)/cuda-8.0/samples/common/inc
LIBRARIES	:= 

NVCCFLAGS := -m64

ALL_CCFLAGS := 
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += -D_MANAGEMEMORY #-D_USEPREFETCH

SMS ?= 60
ifeq ($(GENCODE_FLAGS),)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

#########################################################

all: build

build: unified_mem

unified_mem.o : unified_mem.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

unified_mem: unified_mem.o
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f unified_mem unified_mem.o

clobber: clean
