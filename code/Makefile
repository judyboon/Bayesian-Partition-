CC=g++
LINKER=-L/usr/local/lib
INCLUDE=-I/usr/local/include/gsl -I./
LIBS=-lgsl -lgslcblas -lm
objects = main.o 

main : $(objects)
	$(CC) $(objects) -o main $(LINKER) $(LIBS) $(INCLUDE)

main.o : main.cpp header.h
	gcc -c main.cpp $(INCLUDE)

clean : 
	rm main *.o
