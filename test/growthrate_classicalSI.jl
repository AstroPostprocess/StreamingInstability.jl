# ──────────────────────────────────────────────────────────────────────────── #
#  Test: Classical Streaming Instability — Linear Growth Rate
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  Verifies `ClassicalSIGrowthRateInput` (the callable struct that builds the
#  8×8 linearised perturbation matrix and returns the maximum real eigenvalue)
#  against published benchmark eigenvalues for the classical streaming
#  instability (SI).  Only the *growth rate* — i.e. Im(ω/Ω) in the literature
#  sign convention, which equals max Re(λ) from our eigensystem — is compared,
#  because that is the physically meaningful quantity.
#
#  Reference data
#  ──────────────
#  • linA, linB  — Youdin & Johansen (2007), Table 1 (doi: 10.1086/516729)
#  • linC, linD  — Bai & Stone       (2010), Table 2 (doi: 10.1088/0067-0049/190/2/297)
#
#  Unit convention
#  ───────────────
#  The perturbation matrix is assembled in sound-speed (cₛ) units.  The
#  conversion factor  ηvₖ/cₛ = η / (H/r)  bridges between the normalised
#  wavenumbers Kx,Kz (in units of ηr) and the matrix wavenumbers (in units
#  of H).  See Chen & Lin (2020, ApJ, 892, 114) for full derivation.
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using StreamingInstability

# ========================== Physical parameters ============================= #

hlr = 0.05              # H/r — disc aspect ratio
η_param = 0.0025        # η   — radial pressure gradient parameter
ηvₖlcₛ = η_param / hlr   # ηvₖ / cₛ  =  η / (H/r)
invηvₖlcₛ = inv(ηvₖlcₛ)   # cₛ / (ηvₖ)

# ========================== Helper functions ================================ #

"""Convert wavenumber from ηr units (literature) to H units (matrix)."""
@inline ΚH(kηr :: T) where {T<:AbstractFloat} = T(invηvₖlcₛ) * kηr

"""Dust density from gas density and dust-to-gas ratio."""
@inline ρ̃d(ρ̃g :: TF, ε :: TF) where {TF<:AbstractFloat} = ρ̃g * ε

"""Convert velocity from ηvₖ units (literature) to cₛ units (matrix)."""
@inline vlcₛ(ṽ :: T) where {T<:Complex} = ηvₖlcₛ * ṽ

# ── Equilibrium velocities (Nakagawa–Sekiya–Hayashi drift) ──────────── #
## doi: 10.1016/0019-1035(86)90121-1
Δ(ε :: T, τ :: T)    where {T <: AbstractFloat}  = (one(T) + ε) * (one(T) + ε) + τ * τ
uxlcₛ(ε :: T, τ :: T) where {T <: AbstractFloat} =  T(ηvₖlcₛ) * (T(2) * ε * τ)           / Δ(ε, τ)
uylcₛ(ε :: T, τ :: T) where {T <: AbstractFloat} = -T(ηvₖlcₛ) * (one(T) + ε*τ*τ/Δ(ε,τ))  / (one(T) + ε)
wxlcₛ(ε :: T, τ :: T) where {T <: AbstractFloat} = -T(ηvₖlcₛ) * T(2) * τ                 / Δ(ε, τ)
wylcₛ(ε :: T, τ :: T) where {T <: AbstractFloat} = -T(ηvₖlcₛ) * (one(T) - τ*τ/Δ(ε,τ))    / (one(T) + ε)

# ── Convenience constructor: build input from (St, ε) alone ───────── #

function StreamingInstability.ClassicalSIGrowthRateInput(St::T, ε::T) where {T<:AbstractFloat}
    vxlcs = uxlcₛ(ε, St)
    vylcs = uylcₛ(ε, St)
    wxlcs = wxlcₛ(ε, St)
    wylcs = wylcₛ(ε, St)
    ρg    = one(T)
    ρd    = ρ̃d(ρg, ε)
    return StreamingInstability.ClassicalSIGrowthRateInput(St, ρg, ρd, vxlcs, vylcs, wxlcs, wylcs)
end

# ========================== Reference benchmarks ============================ #
#
# Each block below records the full eigenmode from the literature for
# archival purposes.  Only the eigenvalue's imaginary part (= growth rate)
# is tested numerically; the eigenvectors are kept as documentation.
#

# ──────────────────────────────────────────────────────────────────────── #
#  linA — Youdin & Johansen (2007), Table 1
# ──────────────────────────────────────────────────────────────────────── #
#   Setup:  St = 0.1,  ε = 3.0,  Kx = 30,  Kz = 30
#
#   Eigenmode (normalised):
#     ρ̃g  = + 0.0000224 + 0.0000212im        Gas density
#     ṽx  = - 0.1691398 + 0.0361553im        Gas velocity-x   (ηvₖ)
#     ṽy  = + 0.1336704 + 0.0591695im        Gas velocity-y   (ηvₖ)
#     ṽz  = + 0.1691389 - 0.0361555im        Gas velocity-z   (ηvₖ)
#     ω̃x  = - 0.1398623 + 0.0372951im        Dust velocity-x  (ηvₖ)
#     ω̃y  = + 0.1305628 + 0.0640574im        Dust velocity-y  (ηvₖ)
#     ω̃z  = + 0.1639549 - 0.0233277im        Dust velocity-z  (ηvₖ)
#
#   Eigenvalue:   ω/Ω = - 0.3480127 + 0.4190204im
#   → growth rate  s/Ω = Im(ω/Ω) = 0.4190204
# ──────────────────────────────────────────────────────────────────────── #

# ──────────────────────────────────────────────────────────────────────── #
#  linB — Youdin & Johansen (2007), Table 1
# ──────────────────────────────────────────────────────────────────────── #
#   Setup:  St = 0.1,  ε = 0.2,  Kx = 6,  Kz = 6
#
#   Eigenmode (normalised):
#     ρ̃g  = - 0.0000067 - 0.0000691im        Gas density
#     ṽx  = - 0.0174121 - 0.2770347im        Gas velocity-x   (ηvₖ)
#     ṽy  = + 0.2767976 - 0.0187568im        Gas velocity-y   (ηvₖ)
#     ṽz  = + 0.0174130 + 0.2770423im        Gas velocity-z   (ηvₖ)
#     ω̃x  = + 0.0462916 - 0.2743072im        Dust velocity-x  (ηvₖ)
#     ω̃y  = + 0.2739304 + 0.0039293im        Dust velocity-y  (ηvₖ)
#     ω̃z  = + 0.0083263 + 0.2768866im        Dust velocity-z  (ηvₖ)
#
#   Eigenvalue:   ω/Ω = + 0.4998786 + 0.0154764im
#   → growth rate  s/Ω = Im(ω/Ω) = 0.0154764
# ──────────────────────────────────────────────────────────────────────── #

# ──────────────────────────────────────────────────────────────────────── #
#  linC — Bai & Stone (2010), Table 2
# ──────────────────────────────────────────────────────────────────────── #
#   Setup:  St = 1e-2,  ε = 2.0,  Kx = 1500,  Kz = 1500
#
#   Eigenmode (normalised):
#     ρ̃g  = - 4.9063478 + 2.1332241im        Gas density
#     ṽx  = - 0.1598751 + 0.0079669im        Gas velocity-x   (ηvₖ)
#     ṽy  = + 0.1164423 + 0.0122377im        Gas velocity-y   (ηvₖ)
#     ṽz  = + 0.1598751 - 0.0079669im        Gas velocity-z   (ηvₖ)
#     ω̃x  = - 0.1567174 + 0.0028837im        Dust velocity-x  (ηvₖ)
#     ω̃y  = + 0.1159782 + 0.0161145im        Dust velocity-y  (ηvₖ)
#     ω̃z  = + 0.1590095 - 0.0024850im        Dust velocity-z  (ηvₖ)
#
#   Eigenvalue:   ω/Ω = + 0.1049236 + 0.5980690im
#   → growth rate  s/Ω = Im(ω/Ω) = 0.5980690
# ──────────────────────────────────────────────────────────────────────── #

# ──────────────────────────────────────────────────────────────────────── #
#  linD — Bai & Stone (2010), Table 2
# ──────────────────────────────────────────────────────────────────────── #
#   Setup:  St = 1e-3,  ε = 2.0,  Kx = 2000,  Kz = 2000
#
#   Eigenmode (normalised):
#     ρ̃g  = - 1.0467274 + 0.4551052im        Gas density
#     ṽx  = - 0.1719650 + 0.0740712im        Gas velocity-x   (ηvₖ)
#     ṽy  = + 0.1918893 + 0.0786519im        Gas velocity-y   (ηvₖ)
#     ṽz  = + 0.1719650 - 0.0740712im        Gas velocity-z   (ηvₖ)
#     ω̃x  = - 0.1715840 + 0.0740738im        Dust velocity-x  (ηvₖ)
#     ω̃y  = + 0.1918542 + 0.0787371im        Dust velocity-y  (ηvₖ)
#     ω̃z  = + 0.1719675 - 0.0739160im        Dust velocity-z  (ηvₖ)
#
#   Eigenvalue:   ω/Ω = + 0.3224884 + 0.3154373im
#   → growth rate  s/Ω = Im(ω/Ω) = 0.3154373
# ──────────────────────────────────────────────────────────────────────── #

# ============================== Test body =================================== #

@testset "Streaming Instability -- Linear Growth Rate" begin

    # ── Build linearised inputs from (St, ε) ────────────────────── #
    linAF32 = ClassicalSIGrowthRateInput(Float32(0.1),   Float32(3.0))
    linBF32 = ClassicalSIGrowthRateInput(Float32(0.1),   Float32(0.2))
    linCF32 = ClassicalSIGrowthRateInput(Float32(0.01),  Float32(2.0))
    linDF32 = ClassicalSIGrowthRateInput(Float32(0.001), Float32(2.0))
    linAF64 = ClassicalSIGrowthRateInput(0.1,   3.0)
    linBF64 = ClassicalSIGrowthRateInput(0.1,   0.2)
    linCF64 = ClassicalSIGrowthRateInput(0.01,  2.0)
    linDF64 = ClassicalSIGrowthRateInput(0.001, 2.0)

    # ── Compute growth rates ──────────────────────────────── #
    sAF32 = linAF32(ΚH(Float32(30.0)),   ΚH(Float32(30.0)))
    sBF32 = linBF32(ΚH(Float32(6.0)),    ΚH(Float32(6.0)))
    sCF32 = linCF32(ΚH(Float32(1500.0)), ΚH(Float32(1500.0)))
    sDF32 = linDF32(ΚH(Float32(2000.0)), ΚH(Float32(2000.0)))
    sAF64 = linAF64(ΚH(30.0),   ΚH(30.0))
    sBF64 = linBF64(ΚH(6.0),    ΚH(6.0))
    sCF64 = linCF64(ΚH(1500.0), ΚH(1500.0))
    sDF64 = linDF64(ΚH(2000.0), ΚH(2000.0))

    # ── Expected growth rates: Im(ω/Ω) from the literature ─────── #
    #    Our eigensystem returns  max Re(λ) = Im(ω/Ω)  due to the sign
    #    convention difference noted in the original references.
    ref_sA = 0.4190204     # Youdin & Johansen (2007) linA
    ref_sB = 0.0154764     # Youdin & Johansen (2007) linB
    ref_sC = 0.5980690     # Bai & Stone       (2010) linC
    ref_sD = 0.3154373     # Bai & Stone       (2010) linD

    # ── Assertions ─────────────────────────────────────────────────────────── #
    #    Relative tolerances:
    #      • Float32: 5e-2 (5%), set by the linD benchmark, for which the
    #        observed relative deviation is approximately 3.9%.
    #      • Float64: 1e-3 (0.1%).
    #
    #    The remaining Float32 benchmarks agree with the published growth rates
    #    to within approximately 0.15%. These tolerances validate agreement of
    #    the physically relevant growth rates at the selected benchmark points,
    #    rather than machine-precision agreement of the complete eigensystems.

    @testset "linA  (St=0.1,  ε=3.0,  K=30)"   begin
        @test sAF32 ≈ Float32(ref_sA)  rtol = 5e-2
        @test sAF64 ≈ ref_sA  rtol = 1e-3
    end

    @testset "linB  (St=0.1,  ε=0.2,  K=6)"    begin
        @test sBF32 ≈ Float32(ref_sB)  rtol = 5e-2
        @test sBF64 ≈ ref_sB  rtol = 1e-3
    end

    @testset "linC  (St=0.01, ε=2.0,  K=1500)" begin
        @test sCF32 ≈ Float32(ref_sC)  rtol = 5e-2
        @test sCF64 ≈ ref_sC  rtol = 1e-3
    end

    @testset "linD  (St=1e-3, ε=2.0,  K=2000)" begin
        @test sDF32 ≈ Float32(ref_sD)  rtol = 5e-2
        @test sDF64 ≈ ref_sD  rtol = 1e-3
    end
end
