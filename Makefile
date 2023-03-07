.PHONY: all
all: pxg

FLAGS = -Wall 
INC = -I. -I/engine/includes \
	  -I/home/teaonly/opt/nccl/include \
	  -I/usr/local/cuda/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi 

LINK = -L/home/teaonly/opt/nccl/lib -lnccl \
	   -L/usr/local/cuda/lib64 -lcudnn -lcudart \
	   -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx

OBJS_DIR = ./objs
PXG_SRCS = pxg.cpp embedding.cpp
PXG_OBJS = ${PXG_SRCS:%.cpp=$(OBJS_DIR)/%.o}

$(OBJS_DIR)/%.o : %.cpp
	@mkdir -p $(@D)
	g++ $(FLAGS) $(INC) -c -o $@ $< 

pxg: $(PXG_OBJS) 
	g++ $(FLAGS) -o $@ $(PXG_OBJS) install/libls-kernels.a $(LINK)

clean:
	rm -rf $(OBJS_DIR)
	rm -f pxg
