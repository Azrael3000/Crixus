CXX = g++
LFLAGS = -g -Wall -O3 -L/usr/local/lib -lhdf5 -lz 
CFLAGS = -g -Wall -O3 -I/usr/local/include

BINDIR = bin
OBJDIR = obj
SRCDIR = src
PROG = crixus

SRCS=$(shell for all in $(SRCDIR); do find $${all} -name '*.f90' -o -name '*.cpp'; done;)
OBJS=$(subst $(SRCDIR),$(OBJDIR),$(subst .cpp,.o,$(subst .f90,.o,$(SRCS))))
TARGET = $(BINDIR)/$(PROG)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LFLAGS) $+ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CFLAGS) -o $@ -c $+

clean:
	rm -rf obj/* bin/*

crixus.o: v3d.o
