"""
  NEqProblem(residuals)
  NEqProblem(residuals, options)

An NEqProblem (Non-linear system of Equations Problem), is used to represent the
mathematical problem of finding zeros in the residual function of square systems
of equations. The problem is defined by `residuals` which is an appropriate objective
type (for example `NonDiffed`, `OnceDiffed`, ...) for the types of algorithm to be used.

Options are stored in `options` and are of the `NEqOptions` type. See more information
about options using `?NEqOptions`.

The package NLSolversAD.jl adds automatic conversion of problems to match algorithms
that require higher order derivates than provided by the user. It also adds AD
constructors for a target number of derivatives.
"""
struct NEqProblem{R<:ObjWrapper}
  residuals::R
end
function value(nleq::NEqProblem, x)
    nleq.F(x)
end

"""
  NEqOptions(; ...)

NEqOptions are used to control the behavior of solvers for non-linear systems of
equations. Current options are:

  - `maxiter` [= 10000]: number of major iterations where appropriate
"""
struct NEqOptions{T, Tmi}
  f_abstol::T
  f_reltol::T
  maxiter::Tmi
end
NLEOptions(; f_abstol=1e-8, f_reltol=1e-8, maxiter=10^4) = NLEOptions(f_abstol, f_reltol, maxiter)

"""
KrylovNEqProblem(res)
KrylovNEqProblem(res, opt)

A KrylovNEqProblem (Non-linear system of Equations Problem for Krylov solvers),
is used to represent the mathematical problem of finding zeros in the residual
function of square systems of equations using only the residual function and a
function that calculates Jacobian-vector products. The problem is defined by
`res` which is a `OnceDiffedJv` instance.

Options are stored in `opt` and are of the `KrylovNEqOptions` type. See more information
about options using `?NEqOptions`.

The package NLSolversAD.jl adds constructors for methods that provide residual
function evaluation and Jacobians only (for example `NonDiffed`, `OnceDiffed`, ...).
"""
struct KrylovNEqProblem{R<:OnceDiffedJv}
  krylovres::R
end

"""
  KrylovNEqOptions(; ...)

KrylovNEqOptions are used to control the behavior of solvers for non-linear systems of
equations. Current options are:

  - `maxiter` [= 10000]: number of major iterations where appropriate
"""
struct KrylovNEqOptions{Tmi}
 maxiter::Tmi
end

