using Revise
using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using ForwardDiff

function theta(x)
   if x[1] > 0
       return atan(x[2] / x[1]) / (2.0 * pi)
   else
       return (pi + atan(x[2] / x[1])) / (2.0 * pi)
   end
end

function F_fletcher_powell!(x, Fx)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        end
        Fx[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
        Fx[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
        Fx[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
    Fx
end

function F_jacobian_fletcher_powell!(x, Fx, Jx)
    ForwardDiff.jacobian!(Jx, (y,x)->F_fletcher_powell!(x,y), Fx, x)
    Fx, Jx
end



# import NLSolvers: OnceDiffedJv
# function OnceDiffedJv(F; seed, autodiff=false)
#     JacOp = JacVec(F, seed; autodiff=false)
#     OnceDiffedJv(F, JacOp)
# end

# jv = JacVec((y,x)->F_powell!(x,y), rand(3); autodiff=false)
# function jvop(x)
#     jv.u .= x
#     jv
# end
# prob_obj = NLSolvers.NEqObjective(F_powell!, nothing, F_jacobian_powell!, jvop)


jv = JacVec((y,x)->F_fletcher_powell!(x,y), rand(3); autodiff=false)
function jvop(x)
    jv.u .= x
    jv
end
prob_obj = NLSolvers.NEqObjective(F_fletcher_powell!, nothing, F_jacobian_fletcher_powell!, jvop)

prob = NEqProblem(prob_obj)

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, LineSearch(Newton(), Backtracking()))

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, Anderson(), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, TrustRegion(Newton()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, TrustRegion(DBFGS()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, TrustRegion(BFGS()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, TrustRegion(SR1()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
state = (z=copy(x0), d=copy(x0), Fx=copy(x0), Jx=zeros(3,3))
solve!(prob, x0, LineSearch(Newton(), Backtracking()), NEqOptions(), state)

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, InexactNewton(FixedForceTerm(1e-3), 1e-3, 300), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, InexactNewton(EisenstatWalkerA(), 1e-8, 300), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, InexactNewton(EisenstatWalkerB(), 1e-8, 300), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
solve!(prob, x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())



function dfsane_exponential(x, Fx)
  Fx[1] = exp(x[1]-1)-1
  for i = 2:length(Fx)
  	Fx[i] = i*(exp(x[i]-1)-x[i])
  end
  Fx
end

function FJ_dfsane_exponential!(x, Fx, Jx)
  ForwardDiff.jacobian!(Jx, (y,x)->dfsane_exponential(x,y), Fx, x)
  Fx, Jx
end
prob_obj = NLSolvers.NEqObjective(dfsane_exponential, nothing, FJ_dfsane_exponential!, nothing)

n = 10000
x0 = fill(n/(n-1), n)
solve!(NEqProblem(prob_obj), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())

n = 1000
x0 = fill(n/(n-1), n)
solve!(NEqProblem(prob_obj), x0, LineSearch(NLSolvers.Newton()), NLSolvers.NEqOptions())

n = 10000
x0 = fill(n/(n-1), n)
solve!(NEqProblem(prob_obj), x0, TrustRegion(NLSolvers.Newton()), NLSolvers.NEqOptions())

n = 1000
x0 = fill(n/(n-1), n)
solve!(NEqProblem(prob_obj), x0, TrustRegion(NLSolvers.BFGS()), NLSolvers.NEqOptions())











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
# time limit not enforced in solve(NelderMead)
# no convergence crit either
#
# APSO
# no solve
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
# ADAM needs solve! and AdaMax

#### OPTIMIZATION
f = OPT_PROBS["himmelblau"]["array"]["mutating"]
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
prob = OptimizationProblem(f)
prob_bounds = OptimizationProblem(obj=f, bounds=([-5.0,-9.0],[13.0,4.0]))
prob_on_bounds = OptimizationProblem(obj=f, bounds=([3.5,-9.0],[13.0,4.0]))

solve!(prob, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve(prob, x0, NelderMead(), MinOptions())
@test all(x0 .== [3.0,1.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob_bounds, x0, APSO(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"]).+1
solve(prob_on_bounds, x0, ActiveBox(), MinOptions())
@test_broken all(x0 .== [3.0,1.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob_bounds, x0, SimulatedAnnealing(), MinOptions())
@test_broken all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve(prob_bounds, x0, SIMAN(), MinOptions())
@test all(x0 .== [3.0,1.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(), MinOptions())
#@test all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(SR1()), MinOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(DFP()), MinOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(BFGS()), MinOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(LBFGS()), MinOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(LBFGS(), HZAW()), MinOptions())

#@test all(x0 .== [3.0,2.0])
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve(prob, x0, LineSearch(ConjugateGradient(), HZAW()), MinOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve(prob, x0, LineSearch(ConjugateGradient(update=HS()), HZAW()), MinOptions())

# Stalls at [3, 1] with default solve
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(Newton()), MinOptions())
#@test all(x0 .== [3.0,2.0])
x0 = 1.0.+copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, LineSearch(Newton(), HZAW()), MinOptions())
#@test all(x0 .== [3.0,2.0])

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, TrustRegion(), MinOptions())
@test all(x0 .== [3.0,2.0])
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, TrustRegion(SR1(), NTR()), MinOptions())

x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, TrustRegion(Newton(), Dogleg()), MinOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, TrustRegion(BFGS(), Dogleg()), MinOptions())
x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve!(prob, x0, TrustRegion(DBFGS(), Dogleg()), MinOptions())


x0 = copy(OPT_PROBS["himmelblau"]["array"]["x0"])
solve(prob, x0, Adam(), MinOptions(maxiter=20000))



## Notice that prob is only used for value so this should be extremely generic! It does need a comparison though.
solve(prob, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]), MinOptions())
