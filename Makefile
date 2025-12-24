# Compiler
NVCC = nvcc

# Flags
# -O3: Optimize heavily
# --use_fast_math: Enables fast math intrinsics (like rsqrtf) and aggressive FMA
# -arch=sm_75: Target architecture (Turing). Adjust as needed (e.g., sm_80 for Ampere, sm_86 for 3090, sm_90 for H100)
# -std=c++14: C++ standard
NVCC_FLAGS = -O3 --use_fast_math -arch=sm_75 -std=c++14

# Target executable
TARGET = rmsnorm_test

# Source files
SRCS = rmsnorm_standalone.cu src/rmsnorm_kernel.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
