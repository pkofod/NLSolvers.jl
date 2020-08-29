using NLSolvers


function himmelblau_twicediff(x, ∇f, ∇²f)
    if !(∇²f == nothing)
        ∇²f11 = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
        ∇²f12 = 4.0 * x[1] + 4.0 * x[2]
        ∇²f21 = 4.0 * x[1] + 4.0 * x[2]
        ∇²f22 = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
        ∇²f = [∇²f11 ∇²f12; ∇²f21 ∇²f22]
    end

    if !(∇f == nothing)
        ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        ∇f = [∇f1, ∇f2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

    objective_return(fx, ∇f, ∇²f)
end
himmelblau = TwiceDiffed(himmelblau_twicediff, true)
start = [2.0,2.0]

prob=MinProblem(; obj = himmelblau, bounds = ([0.0,0.0],[2.5,3.0]))

res_unc = solve(prob, start, LineSearch(Newton(), Backtracking()), MinOptions())
res_con = solve(prob, start, NLSolvers.ProjectedNewton(), MinOptions())


























@time @testset "Active Set Projected Newton" begin
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 
bad_rosenbrock(x) = rand() <= 0.01 : (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 : Inf
function f(F, X)
    i = 1
    for x in X
        F[i] = rosenbrock(x)
        i+=1
    end
end
f(x) = rosenbrock(x)
bounds = (fill(-4.0, 2), fill(4.0, 2))
nd = NonDiffed(rosenbrock)
problem = MinProblem(; obj=nd,
	                   bounds=bounds)
nd_batch = NonDiffed(f)
problem_batch = MinProblem(; obj=NLSolvers.Batched(nd_batch),
	                   bounds=bounds)

@show minimize!(problem, zeros(2), APSO(), MinOptions())
@show minimize!(problem_batch, zeros(2), APSO(), MinOptions())


function himmelblau(x)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end
nd = NonDiffed(himmelblau)

problem = MinProblem(; obj=nd,
	                   bounds=bounds)

@show minimize!(problem, zeros(2), APSO(), MinOptions(maxiter=2000))

end