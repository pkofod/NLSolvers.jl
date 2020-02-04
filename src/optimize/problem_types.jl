"""
  MinProblem(...)
A MinProblem (Minimization Problem), is used to represent the mathematical problem
of finding local minima of the given objective function. The problem is defined by `objective`
which is an appropriate objective type (for example `NonDiffed`, `OnceDiffed`, ...)
for the types of algorithm to be used. The constraints of the problem are encoded
in `constraints`. See the documentation for supported types of constraints
including convex sets, and more. It is possible to explicitly state that there
are bounds constraints and manifold constraints on the inputs.

Options are stored in `options` and must be an appropriate options type. See more information
about options using `?MinOptions`.
"""
struct MinProblem{O, B, M, C} <: AbstractProblem
    objective::O
    bounds::B
    manifold::M
    constraints::C
end
MinProblem(; obj=nothing, bounds=nothing, manifold=nothing, constraints=nothing) =
  MinProblem(obj, bounds, manifold, constraints)

# These are conveniences that should be in optim
solve(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, x0, solver=BFGS()) =
  minimize(prob.objective, (x0, I), solver, MinOptions())
solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, x0, solver=BFGS()) =
  minimize!(prob.objective, (x0, I), solver, MinOptions())
solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, s0::Tuple) =
  minimize!(prob.objective, s0, BFGS(), MinOptions())

struct ConvergenceInfo{Ts, T}
  solver::Ts
  info::T
end
function Base.show(io::IO, ::ConvergenceInfo)
  if isa(mr.method, NelderMead)
      println(io, "    √(Σ(yᵢ-ȳ)²)/n $(nm_converged(r) ? "≤" : "≰") $(nm_tol(r))")
  else
      println(io, "    |x - x'|               = $(x_abschange(r)) $(x_abschange(r)<=x_abstol(r) ? "≤" : "≰") $(x_abstol(r))")
      println(io, "    |x - x'|/|x'|          = $(x_relchange(r)) $(x_relchange(r)<=x_reltol(r) ? "≤" : "≰") $(x_reltol(r))")
      println(io, "    |f(x) - f(x')|         = $(f_abschange(r)) $(f_abschange(r)<=f_abstol(r) ? "≤" : "≰") $(f_abstol(r))")
      println(io, "    |f(x) - f(x')|/|f(x')| = $(f_relchange(r)) $(f_relchange(r)<=f_reltol(r) ? "≤" : "≰") $(f_reltol(r))")
      println(io, "    |g(x)|                 = $(g_residual(r)) $(g_residual(r)<=g_tol(r) ?  "≤" : "≰") $(g_tol(r))")
  end
end

struct MinOptions{T1, T2}
    g_abstol::T1
    g0_reltol::T1
    maxiter::T2
    show_trace::Bool
end

MinOptions(; g_abstol=1e-8, g0_reltol=0.0, maxiter=10000, show_trace=false) =
  MinOptions(g_abstol, g0_reltol, maxiter, show_trace)

struct MinResults{Tr, Tc<:ConvergenceInfo, Th, Ts, To}
  res::Tr
  conv::Tc
  history::Th
  solver::Ts
  options::To
end
convinfo(mr::MinResults) = mr.conv
function converged(MinResults)
end 
function Base.show(io::IO, mr::MinResults)
  println(io, "* Status: $(converged(mr))")
  println(io)
  println(io, "* Candidate solution")
  println(io, "  Minimizer: ", minimizer(mr))
  println(io, "  Minimum:   ", minimum(mr))
  println(io)
  println(io, "* Found with")
  println(io, "  Algorithm: ", summary(mr.solver))
  println(io, "  Initial point: ", initial_point(mr))
  println(io, "  Initial value: ", initial_value(mr))

  println(" * Trace stored: ", has_history(mr))


  println(io)
  println(io, " * Convergence measures\n")
  show(io, convinfo(mr))
end

#=
* Convergence measures
  |x - x'|               = 3.47e-07 ≰ 0.0e+00
  |x - x'|/|x'|          = 3.47e-07 ≰ 0.0e+00
  |f(x) - f(x')|         = 6.59e-14 ≰ 0.0e+00
  |f(x) - f(x')|/|f(x')| = 1.20e+03 ≰ 0.0e+00
  |g(x)|                 = 2.33e-09 ≤ 1.0e-08

* Work counters
  Seconds run:   0  (vs limit Inf)
  Iterations:    16
  f(x) calls:    53
  ∇f(x) calls:   53
=#
