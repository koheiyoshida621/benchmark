
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o getTime.o
	nvcc main.o ./AVI/avilib.o ./AVI/avimod.o getTime.o -I/usr/local/cuda/include -lm -o heartwall

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c getTime.o
	nvcc $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3

getTime.o: getTime.cpp
	nvcc $(OUTPUT) $(KERNEL_DIM) getTime.cpp -I./AVI -c -O3

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
