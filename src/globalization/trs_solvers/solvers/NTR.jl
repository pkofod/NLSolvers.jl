# TODO allow passing in a lambda and previous cholesky if the solution was not accepted
# As described in Algorith, 7.3.4 in [CGTBOOK]
struct NTR <: TRSPSolver
end

function initial_λs(∇f, H, Δ)
    n = length(∇f)
    T = eltype(∇f)
    Hfrob = norm(H, 2)
    Hinf  = norm(H, Inf)
    max_diagH = maximum(diag(H))

    norm_to_Δ = norm(∇f)/Δ

    sumabs_j = sum(abs, view(H, 2:n, 1))
    max_i_p = +H[1, 1] + sumabs_j
    max_i_m = -H[1, 1] + sumabs_j
    for j = 2:n
        sumabs_j = T(0)
        for i = 1:n-1
            i == j && continue
            sumabs_j += abs(H[i, j])
        end
        max_i_p = max(max_i_p, +H[j, j] + sumabs_j)
        max_i_m = max(max_i_m, -H[j, j] + sumabs_j)
    end
    λL = max(T(0), norm_to_Δ - min(max_i_p, Hfrob, Hinf), - max_diagH)
    λU = max(T(0), norm_to_Δ + min(max_i_m, Hfrob, Hinf))
    λL, λU
end
function (ms::NTR)(∇f, H, Δ::T, s, scheme, λ0=0; abstol=1e-10, maxiter=50, κeasy=T(1)/10, κhard=T(2)/10) where T
    # λ0 might not be 0 if we come from a failed TRS solve
    λ = T(λ0)
    θ = T(1)/2
    n = length(∇f)
    isg = initial_safeguards(H, ∇f, Δ)
    λ = safeguard_λ(λ, isg)
    λL, λU = isg.L, isg.U

    s₂ = T(0.0)
    diagH = diag(H)
    if !(λ ==T(0))
        for i = 1:n
            @inbounds H[i, i] = diagH[i] + λ
        end
    end

    for iter = 1:maxiter
        if !(λ ==T(0))
            for i = 1:n
                @inbounds H[i, i] = diagH[i] + λ
            end
        end
        F = cholesky(Symmetric(H); check=false)
        in𝓖 = false
        #===========================================================================
         If F is successful, then H is positive definite, and we can safely look
         at the Newton step. If this is interior, we're good.
        ===========================================================================#
        if issuccess(F)
            # H(λ) is PD, so we're in 𝓕
            linpack = false
            s .= (F\-∇f)

            s₂ = norm(s)
            if s₂ ≈ Δ
                for i = 1:n
                    H[i,i] = diagH[i]
                end
                return tr_return(; λ=λ, ∇f=∇f, H=H, s=s, interior=false, solved=true, hard_case=false, Δ=Δ)
            end
            if s₂ < Δ # in 𝓖 because we're in 𝓕, but curve below Δ
                if λ == T(0)
                    for i = 1:n
                        H[i,i] = diagH[i]
                    end
                    return tr_return(;λ=λ, ∇f=∇f, H=H, s=s, interior=true, solved=true, hard_case=false, Δ=Δ)
                else
                    in𝓖 = true
                    # we're in 𝓖 so λ is a new upper bound
                    λU = λ
                end
            else # λ ∈ 𝓛
                # in 𝓛 λ is a *lower* bound instead
               λL = λ
            end
            w = F.U'\s
            λ⁺ = λ + (s₂^2/dot(w,w))*(s₂ - Δ)/Δ
            if in𝓖
                linpack = true
                w, u = λL_with_linpack(F)
                λL = max(λL, λ - dot(u, H*u))
                pa = sum(abs2, u)
                pb = 2*dot(u, s)
                pc = sum(abs2, s)-Δ^2
                pd = sqrt(4*pb^2-pa*pc)
                α₁ = (-pb + pd)/2pa
                α₂ = (-pb - pd)/2pa

                s₁ = s + α₁*u
                m₁ = dot(∇f, s₁) + dot(s₁, H * s₁)/2
                s₂ = s + α₂*u
                m₂ = dot(∇f, s₂) + dot(s₂, H * s₂)/2
                if m₁ ≤ m₂
                    α = α₁
                else
                    α = α₂
                end

                s₂ = norm(s)
                # check hard case convergnce
                if α^2*dot(u, H*u) ≤ κhard*(dot(s, H*s)+λ*Δ^2)
                    for i = 1:n
                        H[i,i] = diagH[i]
                    end
                    println(iter)
                    return tr_return(;λ=λ, ∇f=∇f, H=H, s=s, interior=false, solved=true, hard_case=true, Δ=Δ)
                end
                # If not the hard case solution, try to factorize H(λ⁺)
                if !(λ ==T(0))
                    for i = 1:n
                        @inbounds H[i, i] = diagH[i] + λ⁺
                    end
                end
                F = cholesky(H; check=false)
                if issuccess(F)
                    # Then we're in L, great! lemma 7.3.2
                    λ = λ⁺
                else
                    λ = max(sqrt(λL*λU), λL + θ*(λU - λL))
                end 
            end
            # check for convergence
            if in𝓖 && abs(s₂ - Δ) ≤ κeasy * Δ
                for i = 1:n
                    H[i,i] = diagH[i]
                end
                return tr_return(;λ=λ, ∇f=∇f, H=H, s=s, interior=false, solved=true, hard_case=false, Δ=Δ)
            elseif abs(s₂ - Δ) ≤ κeasy * Δ # implicitly "if in 𝓕" since we're in that branch
                # u and α comes from linpack
                if linpack
                    if α^2*dot(u, H*u) ≤ κhard*(dot(sλ, H*sλ)*Δ^2)
                        s .= s .+ α*u
                        for i = 1:n
                            H[i,i] = diagH[i]
                        end
                        return tr_return(;λ=λ, ∇f=∇f, H=H, s=s, interior=false, solved=true, hard_case=false, Δ=Δ)
                    end
                end
            end
            if !in𝓖
                λ = λ⁺
            end

        else # λ ∈ 𝓝, because the factorization failed (typo in CGT)
            # Use partial factorization to find δ and v such that
            # H(λ) + δ*e*e' = 0. All we can do here is to find a better
            # lower bound, we cannot apply the newton step here.
            δ, v = λL_in_𝓝(H, F)
            #== Step 3d ==#
            λL = max(λL, λ + δ/dot(v, v))

            # No convergence steps since we're in 𝓝
            λ = max(sqrt(λL*λU), λL + θ*(λU - λL))
        end
    end
end


function λL_in_𝓝(H, F)
    T = eltype(F)
    n = first(size(F))
    δ = sum(abs2, view(F.factors, 1:(F.info - 1), F.info)) - H[F.info, F.info]
    v = zeros(T, n)
    v[F.info] = 1
    for j in (F.info - 1):-1:1
        vj = zero(T)
        for i in (j + 1):F.info
            vj += F.factors[j,i]*v[i]
        end
        v[j] = -vj/F.factors[j, j]
    end
    return δ, v
end

function λL_with_linpack(F)
    T = eltype(F)
    n = first(size(F))
    w = zeros(T, n)
    num_p1 = inv(F.factors[end, end])
    num_m1 = -inv(F.factors[end, end])
    w[end] = max(num_p1, num_m1)
    for k = n-1:-1:1
      ukk = F.factors[k, k]
      num = sum(abs2, view(F.factors, 1:(k - 1), k))
      w[k] = max((1-num)/ukk, (-1-num)/ukk)
    end
    sol = F.factors\w
    w, sol./norm(sol)
end