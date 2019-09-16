using IterativeSolvers
abstract type ForcingSequence end

struct FixedForceTerm{T} <: ForcingSequence
    η::T
end
η(fft::FixedForceTerm, info) = fft.η

struct DemboSteihaug <: ForcingSequence
end
η(fft::DemboSteihaug, info) = min(1/(info.k+2), info.normFx)

struct EisenstatWalkerA{T} <: ForcingSequence
    α::T
end
function η(fft::EisenstatWalkerA, info)
    α = fft.α

    ξ = (info.normFx - info.residual_old)/info.normFx_old
    β = info.η_old^(1+sqrt(5)/2)
    η = β ≦ α ? η : max(ξ, β)
end

struct EisenstatWalkerB{T} <: ForcingSequence
    α::T
    γ::T
    ω::T
    η₀::T
end
function η(fft::EisenstatWalkerB, info)
    γ, ω, α = fft.γ, fft.ω, fft.α
    T = typeof(info.normFx)

    ρ = info.normFx/info.normFx_old
    ξ = γ*ρ^ω
    β = info.η_old^(1+sqrt(5)/2)
    η = β ≦ α ? η : max(ξ, β)
end

struct ResidualKrylov{ForcingType<:ForcingSequence, Tη}
    force_seq::ForcingType
    η₀::Tη
    itermax::Int
end
ResidualKrylov(; force_seq=FixedForceTerm(1e-4), eta0 = 1e-4, itermax=300)=ResidualKrylov(force_seq, eta0, itermax)
# map from method to forcing sequence
η(fft::ResidualKrylov, info) = η(fft.force_seq, info)



# Default to EisenstatA-values. The paper [[[]]] suggests that γ ≥ 0.7 and
# ω ≥ (1+sqrt(5))/2
EisenstatWalkerB() = EisenstatWalkerB(0.1, 1.0, 1+sqrt(5)/2, 0.5)

struct ResidualKrylovOp{T1, T2}
    F::T1
    JacOp::T2
end
function ResidualKrylovOp(F; seed, autodiff=false)
    JacOp = JacVec(F, seed; autodiff=false)
    ResidualKrylovOp(F, JacOp)
end

function nlsolve!(kp::ResidualKrylovOp, x, method::ResidualKrylov; f_abstol=1e-8, f_reltol=1e-12)
    xp, Fx = copy(x), copy(x)

    Fx = value!(kp, Fx, x)
    normFx = norm(Fx, 2)

    stoptol = T(f_reltol)*normFx + T(f_abstol)
    force_info = (k = 1, normFx=normFx, normFx_old=nothing, η_old=nothing)

    for i = 1:method.itermax
        # Update the point of evaluation
        jacvec_fn(kp).u .= x

        # Refactor this
        ηₖ = i == 1 && !isa(method.force_seq, FixedForceTerm) ? method.η₀ : η(method, force_info)

        krylov_iter = IterativeSolvers.gmres_iterable!(xp, jacvec_fn(kp), Fx; maxiter=20)

        rhs = ηₖ*norm(Fx)
        for item in krylov_iter
            res = krylov_iter.residual.current
            if res <= rhs
                break
            end
        end
        x .= x .- xp
        Fx = value!(kp, Fx, x)

        if norm(Fx, 2) < stoptol
            break
        end
        normFx_old = force_info.normFx
        η_old = ηₖ
        normFx = norm(Fx, 2)
        force_info = (k = i, normFx=normFx, normFx_old=normFx_old, η_old=η_old, residual_old=res)
    end
    return x, Fx
end
value!(kp::ResidualKrylovOp, Fx, x) = kp.F(Fx, x)
value_fn(kp) = kp.F
jacvec_fn(kp) = kp.JacOp
