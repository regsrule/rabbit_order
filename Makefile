CXXFLAGS += -Wall -Wextra -Wcast-align -Wcast-qual -Wconversion -Wfloat-equal \
	    -Wformat=2 -Winit-self -Wmissing-declarations \
	    -Wmissing-include-dirs -Wpointer-arith -Wredundant-decls \
	    -Wswitch-default -Wuninitialized -Wwrite-strings \
	    -Wno-sign-conversion -Wno-unused-function \
            -Wno-missing-declarations \
            -fopenmp -std=c++14 -mcx16 -O3 -DNDEBUG 

CC = g++
INC = ./
CFLAGS += -I$(INC) -g -O0 -lm -lpthread -fopenmp -ltcmalloc_minimal -std=c++14 -mcx16 -lnuma
HEADERS = \
	edge_list.h\
	rabbit_order.h\
	reorder.h\

OBJ = ./obj

SRC =\
    reorder.cpp\
    main_reorder.cpp\
    rabbit_order.cpp\
    edge_list.cpp

SRC_DIRS = ./ 

vpath %.h $(SRC_DIRS)
vpath %.cpp $(SRC_DIRS)

OBJS = $(addprefix $(OBJ)/, $(SRC:.cpp=.o))
	HEADS = $(addprefix $(INC)/, $(HEADERS))


reorder: $(SRC) $(OBJS) $(HEADERS)
	        $(CC) $(CFLAGS) $(OBJS) -o $@

$(OBJ)/%.o : %.cpp
	        $(CC) $(CFLAGS) -c $< -o $@

clean:
	        rm -f $(OBJ)/*.o cn-order
