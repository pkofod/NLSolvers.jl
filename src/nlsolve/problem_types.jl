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
struct NEqProblem{R, Tb, Tm<:Manifold}
  residuals::R
  bounds::Tb
  manifold::Tm
end
NEqProblem(residuals) = NEqProblem(residuals, nothing, Euclidean(0))
_manifold(prob::NEqProblem) = prob.manifold

function value(nleq::NEqProblem, x)
    nleq.residuals(x)
end
function value(nleq::NEqProblem{<:NonDiffed, <:Any, <:Any}, x, F)
    nleq.residuals(x, F)
end
function value(nleq::NEqProblem{<:Any, <:Any, <:Any}, x, F)
    nleq.residuals(x, F, nothing)
end

"""
  NEqOptions(; ...)

NEqOptions are used to control the behavior of solvers for non-linear systems of
equations. Current options are:

  - `maxiter` [= 10000]: number of major iterations where appropriate
"""
struct NEqOptions{T, Tmi}
  f_limit::T
  f_abstol::T
  f_reltol::T
  maxiter::Tmi
end
NEqOptions(; f_limit=0.0, f_abstol=1e-8, f_reltol=1e-8, maxiter=10^4) = NEqOptions(f_limit, f_abstol, f_reltol, maxiter)

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

function Base.show(io::IO, ci::ConvergenceInfo{<:Any, <:Any, <:NEqOptions})
  opt = ci.options
  info = ci.info

  println(io, "Results of solving non-linear equations\n")
  println(io, "* Algorithm:")
  println(io, "  $(summary(ci.solver))")
  println(io)
  println(io, "* Candidate solution:")
  println(io, "  Final residual norm:      $(@sprintf("%.2e", norm(ci.info.solution, Inf)))")
  if haskey(info, :temperature)
    println(io, "  Final temperature:        $(@sprintf("%.2e", ci.info.temperature))")
  end
  println(io)
  println(io, "  Initial residual norm:    $(@sprintf("%.2e", info.ρF0))")
  println(io)
  println(io, "* Convergence measures")
  if true
#    println(io, "  |x - x'|              = $(@sprintf("%.2e", info.ρs)) <= $(@sprintf("%.2e", opt.x_abstol)) ($(info.ρs<=opt.x_abstol))")
#    println(io, "  |x - x'|/|x|          = $(@sprintf("%.2e", info.ρs/info.ρx)) <= $(@sprintf("%.2e", opt.x_reltol)) ($(info.ρs/info.ρx <= opt.x_reltol))")
    if isfinite(opt.f_limit)
      ρF = norm(info.solution, Inf)
      println(io, "  |F(x')|               = $(@sprintf("%.2e", ρF)) <= $(@sprintf("%.2e", opt.f_limit)) ($(ρF<=opt.f_limit))")
    end
    if haskey(info, :fx)
      Δf = abs(info.fx-info.minimum)
      println(io, "  |f(x) - f(x')|        = $(@sprintf("%.2e", Δf)) <= $(@sprintf("%.2e", opt.f_abstol)) ($(Δf<=opt.f_abstol))")
      println(io, "  |f(x) - f(x')|/|f(x)| = $(@sprintf("%.2e", Δf/abs(info.fx))) <= $(@sprintf("%.2e", opt.f_reltol)) ($(Δf/abs(info.fx)<=opt.f_reltol))")
    end
    if haskey(info, :∇fz)
      ρ∇f = opt.g_norm(info.∇fz)
      println(io, "  |g(x)|                = $(@sprintf("%.2e", ρ∇f)) <= $(@sprintf("%.2e", opt.g_abstol)) ($(ρ∇f<=opt.g_abstol))")
      println(io, "  |g(x)|/|g(x₀)|        = $(@sprintf("%.2e", ρ∇f/info.∇f0)) <= $(@sprintf("%.2e", opt.g_reltol)) ($(ρ∇f/info.∇f0<=opt.g_reltol))")
    end
  end
  println(io)
  println(io, "* Work counters")
  println(io, "  Seconds run:   $(@sprintf("%.2e", info.time))")
  println(io, "  Iterations:    $(info.iter)")
end
