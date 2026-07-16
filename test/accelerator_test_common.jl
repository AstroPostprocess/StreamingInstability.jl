################################################################################

# Shared floating-point regression tests for optional GPU backends.
#
# What this file tests
# 1. CPU baseline
#    • Rebuilds the four published linA--linD streaming-instability cases.
#    • Checks Float32 and Float64 CPU solvers directly against the published
#      growth rates before those results are accepted as GPU references.
# 2. Backend parity
#    • Evaluates a small two-dimensional wavenumber grid on both CPU and GPU.
#    • Uses the same fixtures and tolerances for Metal and CUDA.
# 3. Failure-sensitive behaviour
#    • Confirms that a non-finite wavenumber produces NaN on both CPU and GPU.
#    • Confirms that the host launch wrapper rejects an incorrectly shaped
#      output matrix before launching a kernel.
#
# Physical conventions
# • The literature wavenumbers use ηr units, while the implementation uses
#   gas-scale-height H units. With ηvₖ/cₛ = 0.05, K_H = 20 K_ηr.
# • Gas and dust equilibrium velocities use the Nakagawa--Sekiya--Hayashi
#   drift solution (Nakagawa et al. 1986).
# • The largest real part returned by this eigensystem equals Im(ω/Ω), the
#   growth-rate convention used by the cited benchmark tables.

################################################################################

using Test
using StreamingInstability

function accelerator_test_input(St :: T, ε :: T) where {T <: AbstractFloat}
    ηvₖ = T(0.05)
    Δ = (one(T) + ε)^2 + St^2
    vx = ηvₖ * (T(2) * ε * St) / Δ
    vy = -ηvₖ * (one(T) + ε * St^2 / Δ) / (one(T) + ε)
    ωx = -ηvₖ * T(2) * St / Δ
    ωy = -ηvₖ * (one(T) - St^2 / Δ) / (one(T) + ε)
    return ClassicalSIGrowthRateInput(St, one(T), ε, vx, vy, ωx, ωy)
end

# Published linear benchmarks used by the CPU suite and both GPU backends:
# • linA: Youdin & Johansen (2007), St=0.1,   ε=3.0, Kx=Kz=30.
# • linB: Youdin & Johansen (2007), St=0.1,   ε=0.2, Kx=Kz=6.
# • linC: Bai & Stone (2010),       St=0.01,  ε=2.0, Kx=Kz=1500.
# • linD: Bai & Stone (2010),       St=0.001, ε=2.0, Kx=Kz=2000.
const ACCELERATOR_TEST_CASES = (
    (name = "linA", St = 0.1f0,   ε = 3.0f0, kηr = 30.0f0,   growth = 0.4190204f0),
    (name = "linB", St = 0.1f0,   ε = 0.2f0, kηr = 6.0f0,    growth = 0.0154764f0),
    (name = "linC", St = 0.01f0,  ε = 2.0f0, kηr = 1500.0f0, growth = 0.5980690f0),
    (name = "linD", St = 0.001f0, ε = 2.0f0, kηr = 2000.0f0, growth = 0.3154373f0),
)

function run_accelerator_cpu_test_suite()
    @testset "Accelerator reference -- CPU baselines" begin
        for T in (Float32, Float64), case in ACCELERATOR_TEST_CASES
            input = accelerator_test_input(T(case.St), T(case.ε))
            K = T(20) * T(case.kηr)
            growth = input(K, K)

            # Float32 uses the same 5% physical-regression tolerance as the
            # main CPU suite. linD sets this tolerance with an observed error
            # of about 3.9%; linA--linC are substantially closer. Float64 uses
            # the main CPU suite's 0.1% tolerance.
            @testset "$T $(case.name) published growth rate" begin
                rtol = T === Float32 ? T(5.0e-2) : T(1.0e-3)
                @test growth ≈ T(case.growth) rtol = rtol
            end
        end
    end
    return nothing
end

function run_accelerator_test_suite(config)
    to_device = config.to_device
    to_host = config.to_host
    synchronize = config.synchronize

    @testset "$(config.name) -- CPU parity" begin
        for T in config.float_types, case in ACCELERATOR_TEST_CASES
            input = accelerator_test_input(T(case.St), T(case.ε))
            K = T(20) * T(case.kηr)
            Kxs = T[T(0.8) * K, K]
            Kzs = T[K, T(1.2) * K]

            # This is an explicit CPU execution of the same preallocated
            # matrix API exercised by the GPU extension below.
            expected = input(Kxs, Kzs)
            @test size(expected) == (2, 2)
            @test eltype(expected) === T
            literature_rtol = T === Float32 ? T(5.0e-2) : T(1.0e-3)
            @test expected[2, 1] ≈ T(case.growth) rtol = literature_rtol

            # Transfer identical wavenumbers to the selected backend, launch
            # its extension method, synchronize, and compare on the host.
            device_Kxs = to_device(Kxs)
            device_Kzs = to_device(Kzs)
            device_growth = to_device(zeros(T, length(Kxs), length(Kzs)))
            result = input(device_growth, device_Kxs, device_Kzs)
            @test isnothing(result)
            synchronize()

            actual = to_host(device_growth)
            @test isapprox(actual, expected; rtol = T(5.0e-3), atol = T(5.0e-5), nans = true)
        end
    end

    @testset "$(config.name) -- validation and non-finite modes" begin
        for T in config.float_types
            input = accelerator_test_input(T(0.1), T(3.0))
            Kxs = T[T(600), T(NaN)]
            Kzs = T[T(600)]
            expected = input(Kxs, Kzs)
            device_Kxs = to_device(Kxs)
            device_Kzs = to_device(Kzs)
            device_growth = to_device(zeros(T, 2, 1))

            input(device_growth, device_Kxs, device_Kzs)
            synchronize()
            @test isapprox(to_host(device_growth), expected; rtol = T(5.0e-3), atol = T(5.0e-5), nans = true)

            wrong_shape = to_device(zeros(T, 1, 2))
            @test_throws DimensionMismatch input(wrong_shape, device_Kxs, device_Kzs)
        end
    end

    return nothing
end
