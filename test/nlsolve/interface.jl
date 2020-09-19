using Revise
using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays
using IterativeSolvers
using ForwardDiff
using Test


#
# Have Anderosn use FixedPointProblem
#
#



function theta(x)
   if x[1] > 0
       return atan(x[2] / x[1]) / (2.0 * pi)
   else
       return (pi + atan(x[2] / x[1])) / (2.0 * pi)
   end
end

function F_fletcher_powell!(Fx, x)
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

function F_jacobian_fletcher_powell!(Fx, Jx, x)
    ForwardDiff.jacobian!(Jx, F_fletcher_powell!, Fx, x)
    Fx, Jx
end

jv = JacVec(F_fletcher_powell!, rand(3); autodiff=false)
function jvop(x)
    jv.u .= x
    jv
end
prob_obj = NLSolvers.NEqObjective(F_fletcher_powell!, nothing, F_jacobian_fletcher_powell!, jvop)

prob = NEqProblem(prob_obj)

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, LineSearch(Newton(), Backtracking()))

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, Anderson(), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, TrustRegion(Newton()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, TrustRegion(DBFGS()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, TrustRegion(BFGS()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, TrustRegion(SR1()), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
state = (z=copy(x0), d=copy(x0), Fx=copy(x0), Jx=zeros(3,3))
@show solve!(prob, x0, LineSearch(Newton(), Backtracking()), NEqOptions(), state)

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, InexactNewton(FixedForceTerm(1e-3), 1e-3, 300), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, InexactNewton(EisenstatWalkerA(), 1e-8, 300), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, InexactNewton(EisenstatWalkerB(), 1e-8, 300), NEqOptions())

x0 = [-1.0, 0.0, 0.0]
@show solve!(prob, x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions(maxiter=1000))



function dfsane_exponential(Fx, x)
  Fx[1] = exp(x[1]-1)-1
  for i = 2:length(Fx)
  	Fx[i] = i*(exp(x[i]-1)-x[i])
  end
  Fx
end

function FJ_dfsane_exponential!(Fx, Jx, x)
  ForwardDiff.jacobian!(Jx, dfsane_exponential, Fx, x)
  Fx, Jx
end
prob_obj = NLSolvers.NEqObjective(dfsane_exponential, nothing, FJ_dfsane_exponential!, nothing)

n = 10000
x0 = fill(n/(n-1), n)
@show res = solve!(NEqProblem(prob_obj), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions(maxiter=1000))
@test norm(res.info.best_residual, Inf)<1e-5

n = 1000
x0 = fill(n/(n-1), n)
@show res = solve!(NEqProblem(prob_obj), x0, LineSearch(NLSolvers.Newton()), NLSolvers.NEqOptions())
@test norm(res.info.best_residual, Inf)<1e-8

n = 10000
x0 = fill(n/(n-1), n)
@show res = solve!(NEqProblem(prob_obj), x0, TrustRegion(NLSolvers.Newton()), NLSolvers.NEqOptions())
@test norm(res.info.best_residual, Inf)<1e-8

#n = 1000
#x0 = fill(n/(n-1), n)
#@show solve!(NEqProblem(prob_obj), x0, TrustRegion(NLSolvers.BFGS()), NLSolvers.NEqOptions())
