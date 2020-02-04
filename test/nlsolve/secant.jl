using NLSolvers, Test

solve(NEqProblem(x->x^2), 0.001, Secant(;factor=0.99, shift=1e-1))
solve(NEqProblem(x->sin(x)), 0.001, Secant(;factor=0.99, shift=1e-1))
solve(NEqProblem(x->cos(x)), 0.001, Secant(;factor=0.99, shift=1e-1))

# @test_throws ArgumentError solve(NEqProblem(x->x^2), [0.001,3], Secant(;factor=0.99, shift=1e-8))
