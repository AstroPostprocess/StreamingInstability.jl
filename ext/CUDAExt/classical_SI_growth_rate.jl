function (CSIGRInput :: StreamingInstability.ClassicalSIGrowthRateInput{T})(SIgrowth :: CuMatrix{T}, Κxs :: CuVector{T}, Κzs :: CuVector{T}, :: Val{ThreadsPerBlock} = Val(256)) where {T <: AbstractFloat, ThreadsPerBlock}
    size(SIgrowth) == (length(Κxs), length(Κzs)) || throw(DimensionMismatch("SIgrowth must have size ($(length(Κxs)), $(length(Κzs))), got $(size(SIgrowth))"))
    isempty(SIgrowth) && return nothing
    ThreadsPerBlock > 0 || throw(ArgumentError("ThreadsPerBlock must be positive"))

    @cuda always_inline=true threads=ThreadsPerBlock blocks=cld(length(SIgrowth), ThreadsPerBlock) _classical_SI_growth_rate_kernel!(CSIGRInput, SIgrowth, Κxs, Κzs)
    return nothing
end

@inline function _classical_SI_growth_rate_kernel!(CSIGRInput :: StreamingInstability.ClassicalSIGrowthRateInput{T}, SIgrowth :: MF, Κxs :: V, Κzs :: V) where {T <: AbstractFloat, V <: CuDeviceVector{T}, MF <: CuDeviceMatrix{T}}
    tid = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)

    n = length(SIgrowth)
    i = tid
    ni = length(Κxs)

    ComplexT = Complex{T}
    vx = ComplexT(CSIGRInput.vxlcs)
    vy = ComplexT(CSIGRInput.vylcs)
    ωx = ComplexT(CSIGRInput.ωxlcs)
    ωy = ComplexT(CSIGRInput.ωylcs)
    invSt = ComplexT(CSIGRInput._invSt)
    εinvSt = ComplexT(CSIGRInput._εinvSt)

    Rx, Ry = εinvSt * (ωx - vx), εinvSt * (ωy - vy)
    M = zero(MMatrix{8, 8, ComplexT})
    while i <= n
        idx = rem(i - 1, ni) + 1
        jdx = div(i - 1, ni) + 1

        Κx = Κxs[idx]
        Κz = Κzs[jdx]
        A = -im * Κx * ωx
        B = -im * Κx * vx

        if !isfinite(Κx) | !isfinite(Κz) | !isfinite(A) | !isfinite(B) | !isfinite(invSt) | !isfinite(εinvSt) | !isfinite(Rx) | !isfinite(Ry)
            @inbounds SIgrowth[idx, jdx] = T(NaN)
        else
            @inbounds begin
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
                M[6,5] = (-im * Κx) - Rx
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
            SIgrowth[idx, jdx] = StreamingInstability._realλ_max(M)
        end

        i += stride
    end
    return nothing
end
