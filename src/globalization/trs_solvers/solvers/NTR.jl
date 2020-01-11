# TODO allow passing in a lambda and previous cholesky
# As described in Algorith, 7.3.4 in [CGTBOOK]
struct NTR <: TRSPSolver
end
function (ms::NTR)(∇f, H, Δ::T, s, scheme; abstol=1e-10, maxiter=50, κeasy=T(1)/10, κhard=T(2)/10) where T
    lambda = T(0)
    s₂ = 0.0
    fact = cholesky(H; check=false)
    in𝓕, in𝓖, in𝓛, in𝓝 = false, false, false, false

    #===========================================================================
     If fact is successful, then H is positive definite, and we can safely look
     at the Newton step. If this is interior, we're good.
    ===========================================================================#
    if issuccess(fact)
        #== Step 1a ==#
        in𝓕 = true
        # λ in 𝓕
        R = fact.U
        s .= R\(R'\-∇f) # solve R'y=-∇x and then Rx = y
        s₂ = norm(s)
        if s₂ < Δ
            in𝓖 = true
            #check convergence
        else # λ ∈ 𝓛
            in𝓛 = true
        end
    else # λ ∈ 𝓝
        in𝓝 = true
    end

    #== Step 2 ==#
    if in𝓖
        λU = λ
    else
        ΛL = λ
    end
    if in𝓕
        #== Step 3a ==#
        w = R\s
        λ⁺ = λ + (s₂^2/norm(w)^2)(s₂-Δ)/Δ
        if in𝓖
            #== Step 3b ==#
            #use linpack
            λL = max(λL, λ-dot(u,Hλ*u))
            #α = root of norm(s+αu)=Δ that makes s+αu small
            s .= s + α*u
        end
    else # λ ∉ 𝓕, that is issuccess == false
        #== Step 3c ==#
        # use partial fact to find
        # δ and v such that (Hλ + δ*e*e')v=0
        #== Step 3d ==#
        λL = max(λL, λ + δ/norm(v)^2)
    end

    #== Step 4 ==#
    if in𝓕 && abs(s₂ - Δ) ≤ κeasy * Δ
        return
    elseif iszero(λ) && in𝓖
        return
    elseif in𝓖
        # u and α comes from linpack
        if α^2*dot(u, Hλ*u) ≤ κhard*(dot(sλ, Hλ*sλ)*Δ^2)
            return s + α*u
        end
    end

    #== Step 5 ==#
    if in𝓛 # and g ≂̸ 0 (but we test outside!)
        λ = λ⁺
    elseif in𝓖
        # λ ∈ 𝓖
        fact⁺ = cholesky(Hλ⁺; check = false)
        if issuccess(fact⁺)
            #== Step 5a ==#
            # λ⁺ ∈ 𝓛
            λ = λ⁺
        else
            #== Step 5b ==#
            # λ⁺ ∈ 𝓝
            λL = max(λL, λ⁺)
            # check λL for interior convergenece
            # if interior convergence
            # else
            #     λ = max(sqrt(λL*λU), λL+θ*(λU-λL))
            # end
        end
    else
        λ = max(sqrt(λL*λU), λL+θ*(λU-λL))
    end
end
