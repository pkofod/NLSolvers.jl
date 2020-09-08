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

using LinearMaps
function Jvop(x)
    function JacV(Fv, v)
        Fv[1] = -1*v[1]
        Fv[2,] = -20*x[1]*v[1] + 10*v[2]
    end
    LinearMap(JacV, length(x))
end    


X0 = [[-0.8, 1.0], [-0.7, 1.0], [-0.7, 1.2], [-0.5, 0.7], [0.5, 0.4]]
for x0 in X0
    initial = copy(x0)

    prob=NEqProblem(F!, nothing, nothing, NLSolvers.Euclidean(0))
    @show res = solve!(prob, copy(initial), TrustRegion(Newton(), Dogleg()), NEqOptions())
    
    @show res = solve!(prob, copy(initial), LineSearch(Newton(), Static(0.6)), NEqOptions())
    @test norm(res.info.best_residual, Inf) < 1e-8
    @show res = solve!(prob, copy(initial), DFSANE(), NEqOptions())
    probjv = NEqProblem(F!, Jvop, nothing, NLSolvers.Euclidean(0))
    @show res = solve!(probjv, copy(initial), InexactNewton(), NEqOptions())
end

function F!(x::Vector, fvec::Vector, fjac::Union{Nothing, Matrix}=nothing)
    if !(fvec isa Nothing)
        fvec[1] = x[1] + 10x[2]
        fvec[2] = sqrt(5)*(x[3] - x[4])
        fvec[3] = (x[2] - 2x[3])^2
        fvec[4] = sqrt(10)*(x[1] - x[4])^2
    end
    if !(fjac isa Nothing)
        fill!(fjac, 0)
        fjac[1,1] = 1
        fjac[1,2] = 10
        fjac[2,3] = sqrt(5)
        fjac[2,4] = -fjac[2,3]
        fjac[3,2] = 2(x[2] - 2x[3])
        fjac[3,3] = -2fjac[3,2]
        fjac[4,1] = 2sqrt(10)*(x[1] - x[4])
        fjac[4,4] = -fjac[4,1]
    end
    objective_return(fvec, fjac)
end

function Jvop(x)
    function JacV(Fv, v)
        Fv[1] = v[1]+10*v[2]
        Fv[2] = (v[3]-v[4])*sqrt(5)
        xx23 = 2*x[2]-4*x[3]
        Fv[3] = v[2]*xx23-v[3]*xx23*2
        xx41 = 2*sqrt(10)*(x[1]-x[4])
        Fv[4] = v[1]*xx41 - v[4]*xx41
    end
    LinearMap(JacV, length(x))
end    


prob=NEqProblem(F!, nothing, nothing, NLSolvers.Euclidean(0))

probjv1 = NEqProblem(F!, Jvop, nothing, NLSolvers.Euclidean(0))


end