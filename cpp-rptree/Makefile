CC=g++
CFLAGS='-std=c++11'

all:
	$(CC) $(CFLAGS) -c -o bin/common.o nn/common.cpp
	$(CC) $(CFLAGS) -c -o bin/rp_trees.o nn/rp_trees/rp_trees.cpp
	$(CC) $(CFLAGS) -c -o bin/example.o example.cpp
	$(CC) $(CFLAGS) -o bin/example bin/*.o

clean:
	rm bin/*
