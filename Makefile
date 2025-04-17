# Makefile for reduce_avg and 2.5D matrix multiply

# Primary executables
EXEC = reduce_avg
MUL  = multiply_2_5d

# All targets to build
OBJ = $(EXEC) $(EXEC)-debug $(EXEC)-trace \
      $(MUL)  $(MUL)-debug  $(MUL)-trace

VIEWER = jumpshot

# Compiler flags
OPT   = -O2 -g
DEBUG = -O0 -g

all: $(OBJ)

#------------------------------------------------
# reduce_avg builds
#------------------------------------------------

# Debug build
$(EXEC)-debug: $(EXEC).cpp
	mpicxx $(DEBUG) $(OMP) -o $(EXEC)-debug $(EXEC).cpp -lrt

# MPE‐logging build for Jumpshot
$(EXEC)-trace: $(EXEC).cpp
	mpecxx -mpilog $(OPT) -o $(EXEC)-trace $(EXEC).cpp -lrt

# Optimized build
$(EXEC): $(EXEC).cpp
	mpicxx $(OPT) $(OMP) -o $(EXEC) $(EXEC).cpp -lrt

#------------------------------------------------
# multiply_2_5d builds
#------------------------------------------------

# Debug build
$(MUL)-debug: $(MUL).cpp
	mpicxx $(DEBUG) $(OMP) -o $(MUL)-debug $(MUL).cpp -lrt

# MPE‐logging build for Jumpshot
$(MUL)-trace: $(MUL).cpp
	mpecxx -mpilog $(OPT) $(OMP) -o $(MUL)-trace $(MUL).cpp -lrt

# Optimized build
$(MUL): $(MUL).cpp
	mpicxx $(OPT) $(OMP) -o $(MUL) $(MUL).cpp -lrt

#------------------------------------------------
# Helpers
#------------------------------------------------

# Suggest how to run interactively
runp:
	echo "Try running like this on an interactive compute node:"
	echo "  srun -n <num_procs> your_program <arguments>"

# View an MPI trace with Jumpshot
view:
	$(VIEWER) Unknown.clog2

clean:
	/bin/rm -rf $(OBJ)
