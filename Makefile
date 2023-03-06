.PHONY: all
all: pxg

FLAGS = -Wall 
INC = -I/home/teaonly/opt/nccl/include \
	  -I/usr/local/cuda/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi 

LINK = -L/home/teaonly/opt/nccl/lib -lnccl \
	   -L/usr/local/cuda/lib64 -lcudnn \
	   -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx

pxg: pxg.cpp
	g++ $(FLAGS) $(INC) -o $@ $< $(LINK) 

clean:
	rm -f pxg
