################################################################################

# Test: CPU reference plus optional CUDA and Metal backends
#
# Ordering is intentional:
# 1. Backend package/device/extension detection already happened at the start
#    of runtests.jl, before the ordinary CPU physics suite.
# 2. Float32 and Float64 CPU references are verified here against published
#    linA--linD growth rates.
# 3. CUDA and Metal then run the same backend-neutral regression body against
#    those CPU calculations. Unavailable devices produce one skipped test;
#    broken imports or missing extensions are test failures.
#
# This entire file is included last by runtests.jl, so importing a functional
# backend early does not move GPU allocations or kernel compilation ahead of
# the CPU tests.
#
# Metal and bounds checking
# Pkg.test() uses --check-bounds=yes by default. TinyEigvals relies on mutable
# MMatrix workspaces being stack allocated inside the Metal kernel, but forced
# bounds checking prevents that optimization and leaves an unsupported device
# heap allocation in the generated IR. accelerator_test_setup.jl therefore
# skips Metal when `Base.JLOptions().check_bounds == 1`; Metal runs normally for
# `--check-bounds=auto` (0) and `--check-bounds=no` (2). CUDA is not gated by
# this check.

################################################################################

using Test
using StreamingInstability

if !isdefined(@__MODULE__, :metal_status)
    include("accelerator_test_setup.jl")
end

@static if !isdefined(@__MODULE__, :run_accelerator_test_suite)
    include("accelerator_test_common.jl")
end

function accelerator_test_unavailable(name, status)
    if status.state === :skip
        @info "$name tests skipped" reason = status.reason
        @test_skip false
    else
        @error "$name backend could not be tested" reason = status.reason
        @test false
    end
    return nothing
end

@testset "Accelerators -- optional backends" begin
    run_accelerator_cpu_test_suite()

    @testset "Accelerator -- CUDA backend" begin
        if cuda_status.state !== :ready
            accelerator_test_unavailable("CUDA", cuda_status)
        else
            CUDA_mod = cuda_status.module_ref
            config = (
                name = "CUDA",
                float_types = (Float32, Float64),
                to_device = getproperty(CUDA_mod, :cu),
                to_host = Array,
                synchronize = getproperty(CUDA_mod, :synchronize),
            )
            Base.invokelatest(run_accelerator_test_suite, config)
        end
    end

    @testset "Accelerator -- Metal backend" begin
        if metal_status.state !== :ready
            accelerator_test_unavailable("Metal", metal_status)
        else
            Metal_mod = metal_status.module_ref
            config = (
                name = "Metal",
                float_types = (Float32,),
                to_device = getproperty(Metal_mod, :mtl),
                to_host = Array,
                synchronize = getproperty(Metal_mod, :synchronize),
            )
            Base.invokelatest(run_accelerator_test_suite, config)
        end
    end
end
