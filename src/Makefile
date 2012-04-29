# The Makefile for the C++ implementation of HLDA.

COMPILER = g++
OBJS = utils.o topic.o tree.o document.o corpus.o gibbs.o hlda_main.o

FLAGS = -g -Wall -I/usr/local/include/gsl

# GSL library
LIBS = -lgsl -lgslcblas

default: hlda

hlda: $(OBJS) 
	$(COMPILER) $(FLAGS) $(OBJS) -o hlda  $(LIBS)

.PHONY: clean
clean: 
	rm -f *.o