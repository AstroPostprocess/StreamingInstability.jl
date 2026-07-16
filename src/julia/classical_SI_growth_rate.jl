"""
Growth rate estimation by using Chen & Lin(2020)(doi=10.3847/1538-4357/ab76ca)
    by Wei-Shan Su,
    August 5, 2025
"""

"""
# Fields
| Name    | Description                                                                 |
|---------|-----------------------------------------------------------------------------|
| `St`    | Stokes number of dust particles.                                            |
| `ρg`    | Midplane gas density.                                                       |
| `ρd`    | Midplane dust density.                                                      |
| `vxlcs` | Dimensionless gas velocity along radial (x) axis IN SOUND SPEED (vxlcs = vx_true / c_s).    |
| `vylcs` | Dimensionless gas velocity along azimuthal (y) axis IN SOUND SPEED (vylcs = vy_true / c_s). |
| `ωxlcs` | Dimensionless dust velocity along radial (x) axis IN SOUND SPEED (ωxlcs = ωx_true / c_s).   |
| `ωylcs` | Dimensionless dust velocity along azimuthal (y) axis IN SOUND SPEED (ωylcs = ωy_true / c_s).|
"""
struct ClassicalSIGrowthRateInput{T <: AbstractFloat}
    St      ::  T
    ρg      ::  T
    ρd      ::  T
    vxlcs   ::  T
    vylcs   ::  T
    ωxlcs   ::  T
    ωylcs   ::  T

    # Internal parameters
    _invSt  :: T        # inverse of St
    _εinvSt :: T        # (Midplane) Dust-to-Gas Ratio times invSt
end

function ClassicalSIGrowthRateInput(St      ::  T,
                                    ρg      ::  T,
                                    ρd      ::  T,
                                    vxlcs   ::  T,
                                    vylcs   ::  T,
                                    ωxlcs   ::  T,
                                    ωylcs   ::  T) where {T <: AbstractFloat}
    isfinite(St) && St > zero(T) || throw(ArgumentError("St must be finite and positive"))

    isfinite(ρg) && ρg > zero(T) || throw(ArgumentError("ρg must be finite and positive"))

    isfinite(ρd) && ρd >= zero(T) || throw(ArgumentError("ρd must be finite and non-negative"))

    isfinite(vxlcs) && isfinite(vylcs) && isfinite(ωxlcs) && isfinite(ωylcs) || throw(ArgumentError("All equilibrium velocities must be finite"))

    ε = ρd/ρg
    invSt = inv(St)
    εinvSt = ε * invSt
    return ClassicalSIGrowthRateInput(St, ρg, ρd, vxlcs, vylcs, ωxlcs, ωylcs, invSt, εinvSt)
end

@inline function _realλ_max(M :: MMatrix{N,N,Complex{T}}) where {N,T<:AbstractFloat}
    eigenvalues = tiny_eigvals!(M)

    λmax = T(-Inf)
    @inbounds for k = SOneTo(N)
        λ = eigenvalues[k]
        if !isfinite(λ)
            return T(NaN)
        end
        λmax = max(λmax, real(λ))
    end
    return λmax
end

"""
    (CSIGRInput :: ClassicalSIGrowthRateInput{T})(Κx :: T, Κz :: T) :: T

Compute the dimensionless linear growth rate of the classical streaming
instability formulated by Youdin & Goodman (2005), using the matrix form
given by Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

# Parameters
- `CSIGRInput :: ClassicalSIGrowthRateInput{T}`: The other input parameters for estimating growth rate.
- `Κx         :: T`: Dimensionless radial wavenumber in the shearing box   (Κx = kx × Hg).
- `Κz         :: T`: Dimensionless vertical wavenumber in the shearing box (Κz = kz × Hg).

# Return 
- `T`: Dimensionless streaming instability growth rate (s/Ω)(s = Re(σ))
"""
@inline function (CSIGRInput :: ClassicalSIGrowthRateInput{T})(Κx :: T, Κz :: T) :: T where {T <: AbstractFloat}
    ComplexT = Complex{T}

    vx = ComplexT(CSIGRInput.vxlcs)
    vy = ComplexT(CSIGRInput.vylcs)
    ωx = ComplexT(CSIGRInput.ωxlcs)
    ωy = ComplexT(CSIGRInput.ωylcs)
    invSt = ComplexT(CSIGRInput._invSt)
    εinvSt = ComplexT(CSIGRInput._εinvSt)

    Rx, Ry = εinvSt*(ωx-vx), εinvSt*(ωy-vy)
    A = -im*Κx*ωx
    B = -im*Κx*vx

    # Check all the parameters is finite
    if !isfinite(Κx) | !isfinite(Κz) | !isfinite(invSt) | !isfinite(εinvSt) | !isfinite(Rx) | !isfinite(Ry) | !isfinite(A) | !isfinite(B)
        return T(NaN)
    end
    M = zero(MMatrix{8, 8, ComplexT})
    @inbounds begin
        M[1,1] = A
        M[6,1] = Rx
        M[7,1] = Ry

        M[1,2] = -im * Κx
        M[2,2] = A - invSt
        M[3,2] = -ComplexT(0.5)
        M[6,2] = εinvSt

        M[2,3] = ComplexT(2)
        M[3,3] = A - invSt
        M[7,3] = εinvSt

        M[1,4] = -im * Κz
        M[4,4] = A - invSt
        M[8,4] = εinvSt

        M[5,5] = B
        M[6,5] = (-im * Κx) - (Rx)
        M[7,5] = -Ry
        M[8,5] = -im * Κz

        M[2,6] = invSt
        M[5,6] = -im * Κx
        M[6,6] = B - εinvSt
        M[7,6] = -ComplexT(0.5)

        M[3,7] = invSt
        M[6,7] = ComplexT(2)
        M[7,7] = B - εinvSt

        M[4,8] = invSt
        M[5,8] = -im * Κz
        M[8,8] = B - εinvSt
    end
    return _realλ_max(M)
end

"""
    (CSIGRInput :: ClassicalSIGrowthRateInput{T})(SIgrowth :: M, Κxs :: V, Κzs :: V, i :: Int) where {V <: AbstractVector{T}, M <: AbstractMatrix{T}}

Compute the dimensionless linear growth rate of the classical streaming
instability formulated by Youdin & Goodman (2005), using the matrix form
given by Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

Evaluate one mode from vectors of radial and vertical wavenumbers. The linear
index `i` follows Julia's column-major matrix ordering, so it maps to
`SIgrowth[idx, jdx]`, where `idx = mod(i - 1, length(Κxs)) + 1` and
`jdx = div(i - 1, length(Κxs)) + 1`.

# Parameters
- `CSIGRInput :: ClassicalSIGrowthRateInput{T}`: The other input parameters for estimating growth rate.
- `SIgrowth :: AbstractMatrix{T}`: Preallocated 2D array to store output growth rates. Must be of shape `(length(Κxs), length(Κzs))`.
- `Κxs :: AbstractVector{T}`: Array of dimensionless radial wavenumbers in the shearing box   (Κx = kx × Hg).
- `Κzs :: AbstractVector{T}`: Array of dimensionless vertical wavenumbers in the shearing box (Κz = kz × Hg).
- `i :: Int`: One-based linear index of the output mode to compute.

# Return
- `nothing`: `SIgrowth[i]` is updated in place.
"""
@inline function (CSIGRInput :: ClassicalSIGrowthRateInput{T})(SIgrowth :: MF, Κxs :: V, Κzs :: V, i :: Int) where {T <: AbstractFloat, V <: AbstractVector{T}, MF <: AbstractMatrix{T}}
    @boundscheck begin
        Base.require_one_based_indexing(SIgrowth, Κxs, Κzs)
        size(SIgrowth) == (length(Κxs), length(Κzs)) || throw(DimensionMismatch("SIgrowth must have size ($(length(Κxs)), $(length(Κzs))), got $(size(SIgrowth))"))
        checkbounds(SIgrowth, i)
    end

    ni = length(Κxs)
    idx = rem(i - 1, ni) + 1
    jdx = div(i - 1, ni) + 1

    ComplexT = Complex{T}

    vx = ComplexT(CSIGRInput.vxlcs)
    vy = ComplexT(CSIGRInput.vylcs)
    ωx = ComplexT(CSIGRInput.ωxlcs)
    ωy = ComplexT(CSIGRInput.ωylcs)
    invSt = ComplexT(CSIGRInput._invSt)
    εinvSt = ComplexT(CSIGRInput._εinvSt)

    Rx, Ry = εinvSt*(ωx-vx), εinvSt*(ωy-vy)
    # Check most of the parameters is finite
    if !isfinite(invSt) | !isfinite(εinvSt) | !isfinite(Rx) | !isfinite(Ry)
        @inbounds SIgrowth[i] = T(NaN)
    else
        M = zero(MMatrix{8, 8, ComplexT})
        @inbounds begin
            Κx = ComplexT(Κxs[idx])
            Κz = ComplexT(Κzs[jdx])
            A = -im*Κx*ωx
            B = -im*Κx*vx
            if !isfinite(Κx) | !isfinite(Κz) | !isfinite(A) | !isfinite(B)
                SIgrowth[i] = T(NaN)
            else
                fill!(M, zero(ComplexT))
                M[1,1] = A
                M[6,1] = Rx
                M[7,1] = Ry

                M[1,2] = -im * Κx
                M[2,2] = A - invSt
                M[3,2] = -ComplexT(0.5)
                M[6,2] = εinvSt

                M[2,3] = ComplexT(2)
                M[3,3] = A - invSt
                M[7,3] = εinvSt

                M[1,4] = -im * Κz
                M[4,4] = A - invSt
                M[8,4] = εinvSt

                M[5,5] = B
                M[6,5] = (-im * Κx) - (Rx)
                M[7,5] = -Ry
                M[8,5] = -im * Κz

                M[2,6] = invSt
                M[5,6] = -im * Κx
                M[6,6] = B - εinvSt
                M[7,6] = -ComplexT(0.5)

                M[3,7] = invSt
                M[6,7] = ComplexT(2)
                M[7,7] = B - εinvSt

                M[4,8] = invSt
                M[5,8] = -im * Κz
                M[8,8] = B - εinvSt
                SIgrowth[i] = _realλ_max(M)
            end
        end
    end
    return nothing
end

"""
    (CSIGRInput :: ClassicalSIGrowthRateInput{T})(SIgrowth :: M, Κxs :: V, Κzs :: V) where {V <: AbstractVector{T}, M <: AbstractMatrix{T}}

Compute the dimensionless linear growth rate of the classical streaming
instability formulated by Youdin & Goodman (2005), using the matrix form
given by Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

Evaluate the growth rate over vectors of radial and vertical wavenumbers. This
is the original preallocated interface; it evaluates every linear index by
calling the single-index method and preserves the output layout
`size(SIgrowth) == (length(Κxs), length(Κzs))`.

# Parameters
- `CSIGRInput :: ClassicalSIGrowthRateInput{T}`: The other input parameters for estimating growth rate.
- `SIgrowth :: AbstractMatrix{T}`: Preallocated 2D array to store output growth rates. Must be of shape `(length(Κxs), length(Κzs))`.
- `Κxs :: AbstractVector{T}`: Array of dimensionless radial wavenumbers in the shearing box   (Κx = kx × Hg).
- `Κzs :: AbstractVector{T}`: Array of dimensionless vertical wavenumbers in the shearing box (Κz = kz × Hg).

# Return
- `nothing`: Every element of `SIgrowth` is updated in place.
"""
function (CSIGRInput :: ClassicalSIGrowthRateInput{T})(SIgrowth :: MF, Κxs :: V, Κzs :: V) where {T <: AbstractFloat, V <: AbstractVector{T}, MF <: AbstractMatrix{T}}
    Base.require_one_based_indexing(SIgrowth, Κxs, Κzs)
    size(SIgrowth) == (length(Κxs), length(Κzs)) || throw(DimensionMismatch("SIgrowth must have size ($(length(Κxs)), $(length(Κzs))), got $(size(SIgrowth))"))

    @inbounds for i in 1:length(SIgrowth)
        CSIGRInput(SIgrowth, Κxs, Κzs, i)
    end
    return nothing
end

"""
    (CSIGRInput :: ClassicalSIGrowthRateInput{T})(Κxs :: V, Κzs :: V) where {V <: AbstractVector{T}}

Compute the dimensionless linear growth rate of the classical streaming
instability formulated by Youdin & Goodman (2005), using the matrix form
given by Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

Evaluate the growth rate over vectors of radial and vertical wavenumbers.

# Parameters
- `CSIGRInput :: ClassicalSIGrowthRateInput{T}`: The other input parameters for estimating growth rate.
- `Κxs :: AbstractVector{T}`: Array of dimensionless radial wavenumbers in the shearing box   (Κx = kx × Hg).
- `Κzs :: AbstractVector{T}`: Array of dimensionless vertical wavenumbers in the shearing box (Κz = kz × Hg).
"""
function (CSIGRInput :: ClassicalSIGrowthRateInput{T})(Κxs :: V, Κzs :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    SIgrowth = zeros(T, length(Κxs), length(Κzs))
    CSIGRInput(SIgrowth, Κxs, Κzs)
    return SIgrowth
end
