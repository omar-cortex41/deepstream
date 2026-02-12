CUDA_VER?=12.8
DEEPSTREAM_PATH?=/opt/nvidia/deepstream/deepstream-7.1
CUDA_PATH?=/usr/local/cuda

CC:= g++
CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I$(DEEPSTREAM_PATH)/sources/includes
CFLAGS+= -I$(CUDA_PATH)/include

LFLAGS:= -shared

SRCFILES:= nvdsparsebbox_yolo.cpp
TARGET_LIB:= libnvdsinfer_custom_impl_yolo.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -f $(TARGET_LIB)

