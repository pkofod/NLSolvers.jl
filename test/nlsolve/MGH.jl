using NLSolvers, LinearAlgebra
@testset "nlsolve test suite" begin
function F!(x::Vector, fvec::Vector, fjac::Union{Nothing, Matrix}=nothing)
	if !(fvec isa Nothing)
        fvec[1] = 1 - x[1]
        fvec[2] = 10(x[2]-x[1]^2)
    end
	if !(fjac isa Nothing)
        fjac[1,1] = -1
        fjac[1,2] = 0
        fjac[2,1] = -20*x[1]
        fjac[2,2] = 10
    end
	objective_return(fvec, fjac)
end

X0 = [[-0.8, 1.0], [-0.7, 1.0], [-0.7, 1.2], [-0.5, 0.7], [0.5, 0.4]]
for x0 in X0
    initial = copy(x0)

    prob=NEqProblem(OnceDiffed(F!), nothing, NLSolvers.Euclidean(0))
    @show res = solve!(prob, copy(initial), TrustRegion(Newton(), Dogleg()), NEqOptions())
    @show res = nlsolve!(prob, copy(initial), LineSearch(Newton(), Static(0.6)))
    @test norm(res.info.best_residual, Inf) < 1e-8
    @show res = solve!(prob, copy(initial), DFSANE(), NEqOptions())
end
end