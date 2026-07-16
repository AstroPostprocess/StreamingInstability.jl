using Test
using StreamingInstability

# Detect optional GPU backends before running any test body. Accelerator test
# bodies remain deferred until the final section of this file.
include("accelerator_test_setup.jl")

# 1. Physics modules --------------------------------------------------------- #
include("growthrate_classicalSI.jl")

# 2. Optional accelerators --------------------------------------------------- #
include("accelerator_tests.jl")
