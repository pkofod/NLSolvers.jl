using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using ForwardDiff
using Test
@testset "optimization interface" begin
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

res = solve!(prob, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,1.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob_bounds, x0, APSO(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"]).+1
res = solve(prob_on_bounds, x0, ActiveBox(), MinOptions())
@test_broken all(x0 .== [3.0,1.0])
xbounds = [ 3.5, 1.6165968467447174]
@test res.info.minimum == NLSolvers.value(prob_on_bounds, xbounds)

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob_bounds, x0, SimulatedAnnealing(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-1

#x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
#@show solve(prob_bounds, x0, SIMAN(), MinOptions())
#@test all(x0 .== [3.0,1.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(), MinOptions())
#@test all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(SR1()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(DFP()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(BFGS()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(LBFGS()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(LBFGS(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(LBFGS(), Backtracking()), MinOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

# Stalls at [3, 1] with default @show solve
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(Newton()), MinOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, LineSearch(Newton(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12
#@test all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(SR1(), NTR()), MinOptions())
@test res.info.minimum < 1e-16

# not PSD
#x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
#res = solve!(prob, x0, TrustRegion(Newton(), Dogleg()), MinOptions())
#@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, Adam(), MinOptions(maxiter=20000))
@test res.info.minimum < 1e-16

## Notice that prob is only used for value so this should be extremely generic! It does need a comparison though.
res = solve(prob, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]), MinOptions())

f = OPT_PROBS["exponential"]["array"]["mutating"]
x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
prob = OptimizationProblem(f)
prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(Newton(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(Newton(), NTR()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(Newton(), NWI()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(SR1(), NTR()), MinOptions())
@test_broken res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve!(prob, x0, TrustRegion(SR1(Inverse()), NTR()), MinOptions())
@test res.info.minimum == 2.0
end

const statictest_s0 = OPT_PROBS["himmelblau"]["staticarray"]["state0"]
const statictest_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["staticarray"]["static"]; inplace=false)
@testset "staticopt" begin
    res = solve(statictest_prob, statictest_s0, LineSearch(Newton()), MinOptions())
    @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton()), MinOptions())
    alloc = @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton()), MinOptions())
    @test alloc == 0

    _res = solve(statictest_prob, statictest_s0, LineSearch(Newton()), MinOptions())
    _alloc = @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton()), MinOptions())
    @test _alloc == 0
    @test norm(_res.info.∇fz, Inf) < 1e-8

    _res = solve(statictest_prob, statictest_s0, LineSearch(Newton(), Backtracking()), MinOptions())
    _alloc = @allocated solve(statictest_prob, statictest_s0, LineSearch(Newton(), Backtracking()), MinOptions())
    @test _alloc == 0
    @test norm(_res.info.∇fz, Inf) < 1e-8
end

@testset "newton" begin
    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=true)
    res = solve!(test_prob, copy(test_x0), LineSearch(Newton()), MinOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8

    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
    res = solve(test_prob, test_x0, LineSearch(Newton()), MinOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8
end
@testset "Newton linsolve" begin
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=true)
    res_lu = solve!(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, lu(B), g))), MinOptions())
    @test norm(res_lu.info.∇fz, Inf) < 1e-8
    res_qr = solve!(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, qr(B), g))), MinOptions())
    @test norm(res_qr.info.∇fz, Inf) < 1e-8
  
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
    res_qr = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(B, g)->qr(B)\g)), MinOptions())
    @test norm(res_qr.info.∇fz, Inf) < 1e-8
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=true)
    res_lu = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(B, g)->lu(B)\g)), MinOptions())
    @test norm(res_lu.info.∇fz, Inf) < 1e-8
end















const static_x0 = OPT_PROBS["fletcher_powell"]["staticarray"]["x0"][1]
const static_prob_qn = OPT_PROBS["fletcher_powell"]["staticarray"]["static"]
@testset "no alloc static" begin

    @testset "no alloc" begin
        @allocated solve(static_prob_qn, static_x0, LineSearch(BFGS(Inverse()), Backtracking()), MinOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(BFGS(Inverse()), Backtracking()), MinOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(BFGS(Direct()), Backtracking()), MinOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(BFGS(Direct()), Backtracking()), MinOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(DFP(Inverse()), Backtracking()), MinOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(DFP(Inverse()), Backtracking()), MinOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(DFP(Direct()), Backtracking()), MinOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(DFP(Direct()), Backtracking()), MinOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(SR1(Inverse()), Backtracking()), MinOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(SR1(Inverse()), Backtracking()), MinOptions())
        @test _alloc == 0

        solve(static_prob_qn, static_x0, LineSearch(SR1(Direct()), Backtracking()), MinOptions())
        _alloc = @allocated solve(static_prob_qn, static_x0, LineSearch(SR1(Direct()), Backtracking()), MinOptions())
        @test _alloc == 0
    end
end

Random.seed!(4568532)
solve(static_prob_qn, rand(3), Adam(), MinOptions(maxiter=1000))
solve(static_prob_qn, rand(3), AdaMax(), MinOptions(maxiter=1000))




@testset "bound newton" begin
# using NLSolvers


# function himmelblau_twicediff(x, ∇f, ∇²f)
#     if !(∇²f == nothing)
#         ∇²f11 = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
#         ∇²f12 = 4.0 * x[1] + 4.0 * x[2]
#         ∇²f21 = 4.0 * x[1] + 4.0 * x[2]
#         ∇²f22 = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
#         ∇²f = [∇²f11 ∇²f12; ∇²f21 ∇²f22]
#     end

#     if !(∇f == nothing)
#         ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
#         44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
#         ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
#         4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
#         ∇f = [∇f1, ∇f2]
#     end

#     fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

#     objective_return(fx, ∇f, ∇²f)
# end
# himmelblau = TwiceDiffed(himmelblau_twicediff, true)
# start = [2.0,2.0]

# prob=MinProblem(; obj = himmelblau, bounds = ([0.0,0.0],[2.5,3.0]))

# res_unc = solve(prob, start, LineSearch(Newton(), Backtracking()), MinOptions())
# res_con = solve(prob, start, NLSolvers.ProjectedNewton(), MinOptions())














# @time @testset "Active Set Projected Newton" begin
# rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 
# bad_rosenbrock(x) = rand() <= 0.01 : (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 : Inf
# function f(F, X)
#     i = 1
#     for x in X
#         F[i] = rosenbrock(x)
#         i+=1
#     end
# end
# f(x) = rosenbrock(x)
# bounds = (fill(-4.0, 2), fill(4.0, 2))
# nd = NonDiffed(rosenbrock)
# problem = MinProblem(; obj=nd,
# 	                   bounds=bounds)
# nd_batch = NonDiffed(f)
# problem_batch = MinProblem(; obj=NLSolvers.Batched(nd_batch),
# 	                   bounds=bounds)

# @show minimize!(problem, zeros(2), APSO(), MinOptions())
# @show minimize!(problem_batch, zeros(2), APSO(), MinOptions())


# function himmelblau(x)
#     fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
# end
# nd = NonDiffed(himmelblau)

# problem = MinProblem(; obj=nd,
# 	                   bounds=bounds)

# @show minimize!(problem, zeros(2), APSO(), MinOptions(maxiter=2000))

# end
end