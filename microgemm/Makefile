CC ?= cc
CXX ?= c++
AR ?= ar
CPU_FLAGS ?= -march=native
OMP_FLAGS ?= -fopenmp
THREAD_FLAGS ?= -pthread
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Iinclude $(CPU_FLAGS) $(OMP_FLAGS) $(THREAD_FLAGS)
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Iinclude $(CPU_FLAGS) $(OMP_FLAGS) $(THREAD_FLAGS)
LDFLAGS ?= $(OMP_FLAGS) $(THREAD_FLAGS)

LIB_NAME := libmicrogemm.a
CLI_NAME := microgemm
CONVERT_NAME := microgemm-convert
TEXT_NAME := microgemm-text

LDLIBS ?= -lm

CORE_SRCS := \
	src/microgemm_format.c \
	src/microgemm_runtime.c \
	src/microgemm_ops_cpu.c \
	src/microgemm_decode_cpu.c \
	src/microgemm_model_i8.c

CLI_SRCS := src/microgemm_cli.c
CONVERT_SRCS := src/microgemm_convert.cpp
TEXT_SRCS := src/microgemm_text.cpp

CORE_OBJS := $(CORE_SRCS:.c=.o)
CLI_OBJS := $(CLI_SRCS:.c=.o)
CONVERT_OBJS := $(CONVERT_SRCS:.cpp=.o)
TEXT_OBJS := $(TEXT_SRCS:.cpp=.o)

.PHONY: all clean

all: $(LIB_NAME) $(CLI_NAME) $(CONVERT_NAME) $(TEXT_NAME)

$(LIB_NAME): $(CORE_OBJS)
	$(AR) rcs $@ $^

$(CLI_NAME): $(CLI_OBJS) $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ $(CLI_OBJS) $(LIB_NAME) $(LDFLAGS) $(LDLIBS)

$(CONVERT_NAME): $(CONVERT_OBJS) $(LIB_NAME)
	$(CXX) $(CXXFLAGS) -o $@ $(CONVERT_OBJS) $(LIB_NAME) $(LDFLAGS) $(LDLIBS)

$(TEXT_NAME): $(TEXT_OBJS) $(LIB_NAME)
	$(CXX) $(CXXFLAGS) -o $@ $(TEXT_OBJS) $(LIB_NAME) $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) $(CORE_OBJS) $(CLI_OBJS) $(CONVERT_OBJS) $(TEXT_OBJS) $(LIB_NAME) $(CLI_NAME) $(CONVERT_NAME) $(TEXT_NAME)
