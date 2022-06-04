CC 		= gcc
CFLAGS  =  -Wall -fopenmp -O3
LIBS    = -lm

SRCS = main$(type).c pcg_basic.c
OBJS = $(SRCS:.c=.o)

MAIN = main$(type)

.PHONY: depend clean

all: $(MAIN)

$(MAIN):$(OBJS)
		$(CC) $(CFLAGS) -o $(MAIN) $(OBJS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^
