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
# ADAM needs @show solve and AdaMax
# 
# TODO: LineObjetive doesn't need ! when we have problem in there and mstyle

#### OPTIMIZATION
f = OPT_PROBS["himmelblau"]["array"]["mutating"]
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
prob = OptimizationProblem(f)
prob_oop = OptimizationProblem(f; inplace=false)
prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
prob_bounds_oop = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]); inplace=false)
prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))
prob_on_bounds_oop = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]); inplace=false)

res = solve(prob, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_oop, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,1.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_bounds, x0, APSO(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum == 0.0

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"]).+1
res = solve(prob_on_bounds_oop, x0, ActiveBox(), MinOptions())
@test_broken all(x0 .== [3.0,1.0])
xbounds = [ 3.5, 1.6165968467447174]
@test res.info.minimum == NLSolvers.value(prob_on_bounds, xbounds)

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_bounds, x0, SimulatedAnnealing(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-1

#x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
#@show solve(prob_bounds, x0, SIMAN(), MinOptions())
#@test all(x0 .== [3.0,1.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(), MinOptions())
#@test all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(SR1()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(DFP()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(BFGS()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(LBFGS()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(LBFGS(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(LBFGS(), Backtracking()), MinOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_oop, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob_oop, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())
@test res.info.minimum < 1e-12

# Stalls at [3, 1] with default @show solve
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(Newton()), MinOptions())
@test res.info.minimum < 1e-12

#@test all(x0 .== [3.0,2.0])
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, LineSearch(Newton(), HZAW()), MinOptions())
@test res.info.minimum < 1e-12
#@test all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(SR1(), NTR()), MinOptions())
@test res.info.minimum < 1e-16

# not PSD
#x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
#res = solve(prob, x0, TrustRegion(Newton(), Dogleg()), MinOptions())
#@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum < 1e-16

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
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
res = solve(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(Newton(), Dogleg()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(Newton(), NTR()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(Newton(), NWI()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(SR1(), NTR()), MinOptions())
@test res.info.minimum == 2.0

x0 = copy(OPT_PROBS["exponential"]["array"]["x0"])
res = solve(prob, x0, TrustRegion(SR1(Inverse()), NTR()), MinOptions())
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
    res = solve(test_prob, copy(test_x0), LineSearch(Newton()), MinOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8

    test_x0 = [2.0, 2.0]
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
    res = solve(test_prob, test_x0, LineSearch(Newton()), MinOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8
end
@testset "Newton linsolve" begin
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=true)
    res_lu = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, lu(B), g))), MinOptions())
    @test norm(res_lu.info.∇fz, Inf) < 1e-8
    res_qr = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, qr(B), g))), MinOptions())
    @test norm(res_qr.info.∇fz, Inf) < 1e-8
  
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
    res_qr = solve(test_prob, (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(B, g)->qr(B)\g)), MinOptions())
    @test norm(res_qr.info.∇fz, Inf) < 1e-8
    test_prob = OptimizationProblem(OPT_PROBS["himmelblau"]["array"]["mutating"]; inplace=false)
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
    f = OPT_PROBS["himmelblau"]["array"]["mutating"]
    x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
    prob = OptimizationProblem(f)
    prob_oop = OptimizationProblem(f; inplace=false)
    prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
    prob_bounds_oop = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]); inplace=false)
    prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))
    prob_on_bounds_oop = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]); inplace=false)

    start = [3.7,2.0]

    res_unc = solve(prob_bounds, copy(start), LineSearch(Newton(), Backtracking()), MinOptions())
    @test res_unc.info.minimizer ≈ [3.0, 2.0]
    res_con = solve(prob_bounds, copy(start), ActiveBox(), MinOptions())
    @test res_con.info.minimizer ≈ [3.0, 2.0]
    res_unc = solve(prob_on_bounds, copy(start), LineSearch(Newton(), Backtracking()), MinOptions())
    @test res_unc.info.minimizer ≈ [3.0, 2.0]
    res_con = solve(prob_on_bounds, copy(start), ActiveBox(), MinOptions())
    @test res_con.info.minimizer ≈ [3.5, 1.6165968467448326]
end

function fourth_f(x)
    fx = x^4 + sin(x)
    return fx
end
function fourth_fg(∇f, x)
    ∇f = 4x^3 + cos(x)

    fx = x^4 + sin(x)
    return fx, ∇f
end

function fourth_fgh(∇f, ∇²fx, x)
    ∇²f = 12x^2 - sin(x)
    ∇f = 4x^3 + cos(x)

    fx = x^4 + sin(x)
    return fx, ∇f, ∇²f
end

const scalar_prob_oop = OptimizationProblem(ScalarObjective(fourth_f, nothing, fourth_fg, fourth_fgh, nothing, nothing, nothing, nothing); inplace=false)
@testset "scalar no-alloc" begin
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(SR1(Direct())), MinOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(SR1(Direct())), MinOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(BFGS(Direct())), MinOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(BFGS(Direct())), MinOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(DFP(Direct())), MinOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(DFP(Direct())), MinOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), MinOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), MinOptions())
    @test _alloc == 0

    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), MinOptions())
    _alloc = @allocated solve(scalar_prob_oop, 4.0, LineSearch(Newton()), MinOptions())
    @test _alloc == 0
end


using DoubleFloats
@testset "Test double floats" begin
	function fdouble(x)
	    fx = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
	    return fx
	end
	function fgdouble(G, x)
	    fx = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

        G1 = -2 * (1 - x[1]) - 400 * (x[2] - x[1]^2) * x[1]
        G2 = 200 * (x[2] - x[1]^2)
        G = [G1,G2]
	    return fx, G
	end
	f_obj = OptimizationProblem(ScalarObjective(fdouble, nothing, fgdouble, nothing, nothing, nothing, nothing, nothing))
	res = res = solve(f_obj, Double64.([1,2]), LineSearch(GradientDescent(Inverse())), MinOptions(;g_abstol=1e-32, maxiter=100000))
    @test res.info.minimum < 1e-45
    res = res = solve(f_obj, Double64.([1,2]), LineSearch(BFGS(Inverse())), MinOptions(;g_abstol=1e-32))
    @test res.info.minimum < 1e-45
	res = res = solve(f_obj, Double64.([1,2]), LineSearch(DFP(Inverse())), MinOptions(;g_abstol=1e-32))
    @test res.info.minimum < 1e-45
	res = res = solve(f_obj, Double64.([1,2]), LineSearch(SR1(Inverse())), MinOptions(;g_abstol=1e-32))
    @test res.info.minimum < 1e-45
end


function myfun(x::T) where T
    fx = T(x^4 + sin(x))
    return fx
end
function myfun(∇f, x::T) where T
    ∇f = T(4*x^3 + cos(x))
    fx = myfun(x)
    fx, ∇f
end
function myfun(∇f, ∇²f, x::T) where T<:Real
    ∇²f = T(12*x^2 - sin(x))
    fx, ∇f = myfun(∇f, x)
    T(fx), ∇f, ∇²f
end
@testset "scalar return types" begin
    for T in (Float16, Float32, Float64, Rational{BigInt}, Double32, Double64)
        if T == Rational{BigInt}
            options = MinOptions()
        else
            options = MinOptions(g_abstol=eps(T), g_reltol=T(0))
        end
        for M in (SR1, BFGS, DFP, Newton)
            if M == Newton
                obj = OptimizationProblem(ScalarObjective(myfun, nothing, myfun, myfun, nothing, nothing, nothing, nothing); inplace=false)
                res = solve(obj, T(3.1), LineSearch(M()), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
            else
                obj = OptimizationProblem(ScalarObjective(myfun, nothing, myfun, myfun, nothing, nothing, nothing, nothing); inplace=false)
                res = solve(obj, T(3.1), LineSearch(M(Direct())), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
                res = solve(obj, T(3.1), LineSearch(M(Inverse())), options)
                @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
            end
        end
    end
end



using GeometryTypes
@testset "GeometryTypes" begin
    function fu(G, x)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

        if !(G == nothing)
            G1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
            G2 = 200.0 * (x[2] - x[1]^2)
            gx = Point(G1,G2)

            return fx, gx
        else
            return fx
        end
    end
    fu(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    f_obj = OptimizationProblem(ScalarObjective(fu, nothing, fu, nothing, nothing, nothing, nothing, nothing); inplace=false)
    res = solve(f_obj, Point(1.3,1.3), LineSearch(GradientDescent(Inverse())), MinOptions())
    @test res.info.minimum < 1e-9
    res = solve(f_obj, Point(1.3,1.3), LineSearch(BFGS(Inverse())), MinOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(DFP(Inverse())), MinOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(SR1(Inverse())), MinOptions())
    @test res.info.minimum < 1e-10
    
    res = solve(f_obj, Point(1.3,1.3), LineSearch(GradientDescent(Direct())), MinOptions())
    @test res.info.minimum < 1e-9
    res = solve(f_obj, Point(1.3,1.3), LineSearch(BFGS(Direct())), MinOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(DFP(Direct())), MinOptions())
    @test res.info.minimum < 1e-10
    res = solve(f_obj, Point(1.3,1.3), LineSearch(SR1(Direct())), MinOptions())
    @test res.info.minimum < 1e-10
end