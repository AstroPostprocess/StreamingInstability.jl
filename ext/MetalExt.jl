module MetalExt
using Metal
using StaticArrays
using StreamingInstability

include(joinpath(@__DIR__, "MetalExt", "classical_SI_growth_rate.jl"))

end
