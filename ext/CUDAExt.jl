module CUDAExt
using CUDA
using StaticArrays
using StreamingInstability

include(joinpath(@__DIR__, "CUDAExt", "classical_SI_growth_rate.jl"))

end
