using NLSolvers
@time @testset "Adaptive Particle Swarm" begin
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
# ParaNonDiffed
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
#nd_batch = NonDiffed(f)
#problem_batch = MinProblem(; obj=NLSolvers.Batched(nd_batch),
#	                   bounds=bounds)

@show minimize!(problem, zeros(2), APSO(), MinOptions(maxiter=2000))
#show minimize!(problem_batch, zeros(2), APSO(), MinOptions())


end