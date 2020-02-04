# [SOREN] TRUST REGION MODIFICATION OF NEWTON'S METHOD
# [N&W] Numerical optimization
# [Yuan] A review of trust region algorithms for optimization
abstract type TRSPSolver end
abstract type NearlyExactTRSP <: TRSPSolver end

include("solvers/NWI.jl")
include("solvers/dogleg.jl")
include("solvers/NTR.jl")
#include("subproblemsolvers/TRS.jl") just make an example instead of relying onTRS.jl

function tr_return(;λ, ∇f, H, s, interior, solved, hard_case, Δ)
	interior = false
	solved = true
	hard_case = false

	m = dot(∇f, s) + dot(s, H * s)/2

	(p=s, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved, Δ=Δ)
end