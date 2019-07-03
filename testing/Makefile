#
# Make file for unit test - compares data in two binary files 
#

SRC=compareValuesBin.c

LIB=unittest.c

LIBOBJ=$(LIB:.c=.o)

OUT=$(SRC:.c=)

LIBOUT=libunittest.a

CFLAGS=-O3 -std=gnu99 -I../include

LIBFLAGS=-lm -L. -lunittest

GCC=gcc
AR=ar -cru

default: code

lib:
	$(CC) $(CFLAGS) -o $(LIBOBJ) -c $(LIB) 
	$(AR) $(LIBOUT) $(LIBOBJ)

code:	lib
	$(CC) $(SRC) $(CFLAGS) -o $(OUT) $(LIBFLAGS)

clean:
	rm -f *.o *.a $(OUT)
