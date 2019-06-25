struct TRSolver{T} <: NearlyExactTRSP
    abstol::T
    maxiter::Integer
end
function (ms::TRSolver)(∇f, H, Δ, p)
    T = eltype(p)
    x, info = trs(H, ∇f, Δ)
    p .= x[:,1]

    m = dot(∇f, p) + dot(p, H * p)/2
    interior = norm(p, 2) ≤ Δ
    return (p=p, mz=m, interior=interior, λ=info.λ, hard_case=info.hard_case, solved=true)
end
