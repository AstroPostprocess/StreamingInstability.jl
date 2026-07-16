function (CSIGRInput :: StreamingInstability.ClassicalSIGrowthRateInput{Float32})(SIgrowth :: MtlMatrix{Float32}, Κxs :: MtlVector{Float32}, Κzs :: MtlVector{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    size(SIgrowth) == (length(Κxs), length(Κzs)) || throw(DimensionMismatch("SIgrowth must have size ($(length(Κxs)), $(length(Κzs))), got $(size(SIgrowth))"))
    isempty(SIgrowth) && return nothing
    ThreadsPerGroup > 0 || throw(ArgumentError("ThreadsPerGroup must be positive"))

    @metal always_inline=true threads=(ThreadsPerGroup,) groups=(cld(length(SIgrowth), ThreadsPerGroup),) _classical_SI_growth_rate_kernel!(CSIGRInput, SIgrowth, Κxs, Κzs)
    return nothing
end

@inline function _classical_SI_growth_rate_kernel!(CSIGRInput :: StreamingInstability.ClassicalSIGrowthRateInput{Float32}, SIgrowth :: MF, Κxs :: V, Κzs :: V) where {V <: MtlDeviceVector{Float32}, MF <: MtlDeviceMatrix{Float32}}
    # Get the global thread index and stride
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    n = length(SIgrowth)
    i = tid

    ni = length(Κxs)
    vx = ComplexF32(CSIGRInput.vxlcs)
    vy = ComplexF32(CSIGRInput.vylcs)
    ωx = ComplexF32(CSIGRInput.ωxlcs)
    ωy = ComplexF32(CSIGRInput.ωylcs)
    invSt = ComplexF32(CSIGRInput._invSt)
    εinvSt = ComplexF32(CSIGRInput._εinvSt)

    Rx, Ry = εinvSt*(ωx-vx), εinvSt*(ωy-vy)
    M = zero(MMatrix{8, 8, ComplexF32})
    while i <= n
        idx = rem(i - 1, ni) + 1
        jdx = div(i - 1, ni) + 1

        Κx = Κxs[idx]
        Κz = Κzs[jdx]

        A = -im*Κx*ωx
        B = -im*Κx*vx

        if !isfinite(Κx) | !isfinite(Κz) | !isfinite(A) | !isfinite(B) | !isfinite(invSt) | !isfinite(εinvSt) | !isfinite(Rx) | !isfinite(Ry)
            @inbounds SIgrowth[i] = Float32(NaN)
        else
            @inbounds begin
                fill!(M, zero(ComplexF32))
                M[1,1] = A
                M[6,1] = Rx
                M[7,1] = Ry

                M[1,2] = -im * Κx
                M[2,2] = A - invSt
                M[3,2] = -ComplexF32(0.5)
                M[6,2] = εinvSt

                M[2,3] = ComplexF32(2)
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
                M[7,6] = -ComplexF32(0.5)

                M[3,7] = invSt
                M[6,7] = ComplexF32(2)
                M[7,7] = B - εinvSt

                M[4,8] = invSt
                M[5,8] = -im * Κz
                M[8,8] = B - εinvSt
            end
            # Match the CPU linearized interface: idx and jdx select the
            # wavenumbers, while i is the linear output index.
            @inbounds SIgrowth[i] = StreamingInstability._realλ_max(M)
        end

        i += stride
    end
    return nothing
end
