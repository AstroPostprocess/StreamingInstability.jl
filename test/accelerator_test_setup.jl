################################################################################

# Optional accelerator package detection and startup loading.

################################################################################

function accelerator_test_find_package(name :: String)
    path = Base.find_package(name)
    path !== nothing && return path

    # Pkg.test() isolates LOAD_PATH from the user's default environment.
    default_environment = "@v#.#"
    if default_environment ∉ LOAD_PATH
        push!(LOAD_PATH, default_environment)
        path = Base.find_package(name)
    end
    return path
end

function accelerator_test_load_package(name :: Symbol)
    accelerator_test_find_package(String(name)) === nothing && return (
        state = :skip,
        module_ref = nothing,
        reason = "$(name).jl is not installed in the active or default Julia environment.",
    )

    module_ref = try
        Base.eval(@__MODULE__, :(import $(name)))
        Base.invokelatest(getfield, @__MODULE__, name)
    catch err
        return (
            state = :error,
            module_ref = nothing,
            reason = "$(name).jl was found but could not be imported: $(sprint(showerror, err))",
        )
    end

    functional = try
        Base.invokelatest(getproperty(module_ref, :functional))
    catch err
        return (
            state = :skip,
            module_ref = nothing,
            reason = "$(name).functional() could not initialise a usable device: $(sprint(showerror, err))",
        )
    end
    functional || return (
        state = :skip,
        module_ref = nothing,
        reason = "$(name).jl is installed, but $(name).functional() returned false.",
    )

    extension_name = Symbol(name, :Ext)
    extension = Base.invokelatest(Base.get_extension, StreamingInstability, extension_name)
    extension === nothing && return (
        state = :error,
        module_ref = nothing,
        reason = "StreamingInstability.$extension_name was not loaded after importing $(name).jl.",
    )

    return (state = :ready, module_ref = module_ref, reason = nothing)
end

# Load usable backends at startup, before the CPU test suite. Their test bodies
# are included only at the end of runtests.jl.
cuda_status = accelerator_test_load_package(:CUDA)

# Pkg.test() starts Julia with --check-bounds=yes by default. Under that global
# setting, the mutable StaticArrays workspaces allocated by TinyEigvals cannot
# be reduced to stack storage while Metal is generating device IR. The kernel
# consequently contains `gpu_gc_pool_alloc`, which Metal cannot lower, and
# compilation fails with InvalidIRError. `auto` (0) and `no` (2) both permit
# the required stack allocation optimization, so Metal tests run in those
# modes; explicit `yes` (1) is skipped before Metal is imported or initialized.
check_bounds = Base.JLOptions().check_bounds
metal_status = if !Sys.isapple()
    (
        state = :skip,
        module_ref = nothing,
        reason = "Metal.jl requires macOS.",
    )
elseif check_bounds == 1
    (
        state = :skip,
        module_ref = nothing,
        reason = "Metal tests require --check-bounds=auto or --check-bounds=no; --check-bounds=yes prevents TinyEigvals MMatrix workspaces from being stack allocated in Metal device IR.",
    )
else
    accelerator_test_load_package(:Metal)
end
