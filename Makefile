#------------------------------------------------------------------------------
#  Makefile – 2.5 D matrix multiply + reduce_avg
#
#  Targets
#    make                Optimised builds        (multiply_2_5d  reduce_avg)
#    make debug          -O0 + g builds          (<prog>-debug)
#    make trace          MPE tracing builds      (<prog>-trace)  (for Jumpshot)
#    make hpc            Build with -g for HPCToolkit  (multiply_2_5d-hpc)
#    make clean          Remove all build artefacts
#
#  Optional environment overrides
#    BLAS=1    link OpenBLAS and add -DUSE_BLAS
#    VERIFY=1  compile with -DVERIFY (small‑N correctness check)
#------------------------------------------------------------------------------

# ---------------------------------------------------------------- Programs ---
EXEC  := reduce_avg
MUL   := multiply_2_5d

# ------------------------------------------------------- Compiler toolchain ---
MPICXX ?= mpicxx         # OpenMPI's C++ wrapper
MPECXX ?= mpecxx         # wrapper that links the MPE trace libs

# -------------------------------------------------------- Common flags -------
CXXSTD     := -std=c++17
OPTFLAGS   := -O3 -march=native -g
DEBUGFLAGS := -O0 -g

ifdef BLAS
  OPTFLAGS += -DUSE_BLAS
  LIBS     += -lopenblas
endif
ifdef VERIFY
  OPTFLAGS += -DVERIFY
endif

# ----------------------------------------------------------------- Default ---
all: $(EXEC) $(MUL)

# -------------------------------------------- Optimised (release) binaries ---
$(EXEC): $(EXEC).cpp
	$(MPICXX) $(CXXSTD) $(OPTFLAGS) -o $@ $< $(LIBS)

$(MUL):  $(MUL).cpp
	$(MPICXX) $(CXXSTD) $(OPTFLAGS) -o $@ $< $(LIBS)

# ----------------------------------------------------------- Debug builds ----
debug: $(EXEC)-debug $(MUL)-debug

$(EXEC)-debug: $(EXEC).cpp
	$(MPICXX) $(CXXSTD) $(DEBUGFLAGS) -o $@ $<

$(MUL)-debug: $(MUL).cpp
	$(MPICXX) $(CXXSTD) $(DEBUGFLAGS) -o $@ $<

# --------------------------------------------------------- Trace builds ------
trace: $(EXEC)-trace $(MUL)-trace

$(EXEC)-trace: $(EXEC).cpp
	$(MPECXX) -mpilog $(CXXSTD) $(OPTFLAGS) -o $@ $< $(LIBS)

$(MUL)-trace: $(MUL).cpp
	$(MPECXX) -mpilog $(CXXSTD) $(OPTFLAGS) -o $@ $< $(LIBS)

# ----------------------------------------------- HPCToolkit‑friendly build ---
hpc: $(MUL)-hpc

$(MUL)-hpc: $(MUL).cpp
	$(MPICXX) $(CXXSTD) $(OPTFLAGS) -g -o $@ $< $(LIBS)

# ----------------------------------------------------------------- Cleanup ---
clean:
	$(RM) $(EXEC) $(EXEC)-debug $(EXEC)-trace \
	      $(MUL)  $(MUL)-debug  $(MUL)-trace $(MUL)-hpc \
	      *.o *.clog2 *.slog2

.PHONY: all debug trace hpc clean
