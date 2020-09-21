using Revise
using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using ForwardDiff

# TODO
# Make a more efficient MeritObjective that returns something that acts as the actual thing if requested (mostly for debug)
# but can also be efficiently used to get cauchy and newton
#
# Make DOGLEG work also with BFGS (why is convergence so slow?)
# # Look into what caches are created
# # Why does SR1 give inf above?
#
# Mutated first r last (easy to make fallback for nonmutating)
# NelderMead
# time limit not enforced in @show solve(NelderMead)
# no convergence crit either
#
# APSO
# no @show solve
# wrong return type (no ConvergenceInfo)
#
# PureRandom search.
#
# Does not have a ! method, this should be documented. Maybe add it for consistency?
# If the sampler is empty and there are bounds, draw uniformly there in stead of specifying lb and ub in PureRandomSearch
#
# really need a QNmodel for model vars that creates nothing or don't populate fields of a named tuple for Newton for example
# LineObjective and  LineObjective! should just dispatch on the caceh being nothing or not
#
# ConjgtaeGraduent with HZAW fails because it overwrites Py into P∇fz which seems to alias P∇fz. That alias needs to be checked
# and a CGModelVars type should allocate Py where appropriate - could y be overwitten with Py and then recalcualte y afterwards?
#
#
# ADAM needs @show solve! and AdaMax
# 
# TODO: LineObjetive doesn't need ! when we have problem in there and mstyle

#### OPTIMIZATION
f = OPT_PROBS["himmelblau"]["array"]["mutating"]
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
prob = OptimizationProblem(f)
prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))

@show res = solve!(prob, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve(prob, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,1.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob_bounds, x0, APSO(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"]).+1
@show res = solve(prob_on_bounds, x0, ActiveBox(), MinOptions())
@test_broken all(x0 .== [3.0,1.0])
xbounds = [ 3.5, 1.6165968467447174]
@test res.info.minimum == NLSolvers.value(prob_on_bounds, xbounds)

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob_bounds, x0, SimulatedAnnealing(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-1

#x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
#@show solve(prob_bounds, x0, SIMAN(), MinOptions())
#@test all(x0 .== [3.0,1.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob, x0, LineSearch(), MinOptions())
#@test all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob, x0, LineSearch(SR1()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob, x0, LineSearch(DFP()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob, x0, LineSearch(BFGS()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob, x0, LineSearch(LBFGS()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob, x0, LineSearch(LBFGS(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show res = solve!(prob, x0, LineSearch(LBFGS(), Backtracking()), MinOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve(prob, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

# Stalls at [3, 1] with default @show solve
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, LineSearch(Newton()), MinOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, LineSearch(Newton(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12
#@test all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, TrustRegion(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, TrustRegion(SR1(), NTR()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, TrustRegion(Newton(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
@show solve(prob, x0, Adam(), MinOptions(maxiter=20000))
@test res.info.minimum < 1e-16

## Notice that prob is only used for value so this should be extremely generic! It does need a comparison though.
@show solve(prob, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]), MinOptions())

f = OPT_PROBS["exponential"]["array"]["mutating"]
x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
prob = OptimizationProblem(f)
prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
@show res = solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
@show res = solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
@show res = solve!(prob, x0, TrustRegion(Newton(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
@show res = solve!(prob, x0, TrustRegion(Newton(), NTR()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
@show res = solve!(prob, x0, TrustRegion(Newton(), NWI()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
@show res = solve!(prob, x0, TrustRegion(SR1(), NTR()), MinOptions())
@test_broken res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
@show res = solve!(prob, x0, TrustRegion(SR1(Inverse()), NTR()), MinOptions())
@test res.info.minimum == 2.0
