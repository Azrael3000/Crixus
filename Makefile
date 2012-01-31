#LFLAGS = -g -Wall -O3 -L/usr/local/lib -lhdf5 -lz 
#CFLAGS = -g -Wall -O3 -I/usr/local/include

OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:]) 
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:]) 
 
dbg = 1
SMVERSION = 11

CUDA_INSTALL_PATH = /usr/local/cuda
CUDA_SDK_PATH = /home/kassiotis/NVIDIA_GPU_Computing_SDK/

ROOTDIR = $(CUDA_SDK_PATH)C/common
INCLUDES = -I../cudpp/include/
INCLUDES += -I$(CUDA_SDK_PATH)/shared/inc/

BINDIR = bin
OBJDIR = obj/
SRCDIR = src/

CCFILES :=\
  src/crixus.cpp
CUFILES_sm_xx :=\
  src/crixus.cu\
  src/crixus_d.cu
CU_DEPS =\
  src/crixus_d.cuh\
  src/lock.cuh\
  src/cuda_local.cuh\

USECUDPP = 1
USEGLLIB = 1
USEGLUT = 1
OMIT_CUTIL_LIB = 1

ifeq ($(SMVERSION),11)
  CUFILES_sm_11 = $(CUFILES_sm_xx)
else
  CUFILES_sm_20 = $(CUFILES_sm_xx) 
endif

CXXFLAGS +=
NVCCFLAGS += -arch sm_11

EXECUTABLE = crixus

include $(CUDA_SDK_PATH)/C/common/common.mk
