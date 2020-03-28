using IterativeSolvers
abstract type ForcingSequence end

struct FixedForceTerm{T} <: ForcingSequence
    η::T
end
η(fft::FixedForceTerm, info) = fft.η

struct DemboSteihaug <: ForcingSequence
end
η(fft::DemboSteihaug, info) = min(1/(info.k+2), info.ρFz)

struct EisenstatWalkerA{T} <: ForcingSequence
    α::T
end
function η(fft::EisenstatWalkerA, info)
    α = fft.α

    ξ = (info.ρFz - info.residual_old)/info.ρFx
    β = info.η_old^(1+sqrt(5)/2)
    η = β ≦ α ? η : max(ξ, β)
end

struct EisenstatWalkerB{T} <: ForcingSequence
    α::T
    γ::T
    ω::T
    η₀::T
end
# Default to EisenstatA-values. The paper [[[]]] suggests that γ ≥ 0.7 and
# ω ≥ (1+sqrt(5))/2
EisenstatWalkerB() = EisenstatWalkerB(0.1, 1.0, 1+sqrt(5)/2, 0.5)

function η(fft::EisenstatWalkerB, info)
    γ, ω, α = fft.γ, fft.ω, fft.α
    T = typeof(info.ρFz)

    ρ = info.ρFz/info.ρFx
    ξ = γ*ρ^ω
    β = info.η_old^(1+sqrt(5)/2)
    η = β ≦ α ? η : max(ξ, β)
end

struct ResidualKrylov{ForcingType<:ForcingSequence, Tη}
    force_seq::ForcingType
    η₀::Tη
    maxiter::Int
end
ResidualKrylov(; force_seq=FixedForceTerm(1e-4), eta0 = 1e-4, maxiter=300)=ResidualKrylov(force_seq, eta0, maxiter)
# map from method to forcing sequence
η(fft::ResidualKrylov, info) = η(fft.force_seq, info)




function nlsolve!(prob, x, method::ResidualKrylov; f_abstol=1e-8, f_reltol=1e-12)
    Tx = eltype(x)
    xp, Fx = copy(x), copy(x)

    Fx = value!(prob, Fx, x)
    ρFz = norm(Fx, 2)

    JvOp = jacvec_op(prob)

    stoptol = Tx(f_reltol)*ρFz + Tx(f_abstol)

    force_info = (k = 1, ρFz=ρFz, ρFx=nothing, η_old=nothing)

    for i = 1:method.maxiter
        # Refactor this
        if i == 1 && !isa(method.force_seq, FixedForceTerm) 
            ηₖ = method.η₀
        else 
            ηₖ = η(method, force_info)
        end

        krylov_iter = IterativeSolvers.gmres_iterable!(xp, jacvec_fn(prob), Fx; maxiter=20)
        local res
        rhs = ηₖ*norm(Fx)
        for item in krylov_iter
            res = krylov_iter.residual.current
            if res <= rhs
                break
            end
        end
        @. x = x - xp
        Fx = value!(prob, Fx, x)

        if norm(Fx, 2) < stoptol
            break
        end
        ρFx = force_info.ρFz
        η_old = ηₖ
        ρFz = norm(Fx, 2)
        force_info = (k = i, ρFz=ρFz, ρFx=ρFx, η_old=η_old, residual_old=res)
    end
    return x, Fx
end
value!(prob::OnceDiffedJv, Fx, x) = prob.R(Fx, x)
value_fn(prob) = prob.R
jacvec_fn(prob) = prob.Jv
