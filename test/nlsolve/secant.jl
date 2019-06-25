using NLSolvers, Test

solve(NLEProblem(x->x^2), 0.001, Secant(;factor=0.99, shift=1e-1))
solve(NLEProblem(x->sin(x)), 0.001, Secant(;factor=0.99, shift=1e-1))
solve(NLEProblem(x->cos(x)), 0.001, Secant(;factor=0.99, shift=1e-1))

# @test_throws ArgumentError solve(NLEProblem(x->x^2), [0.001,3], Secant(;factor=0.99, shift=1e-8))
