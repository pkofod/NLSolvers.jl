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

struct ConvergenceInfo{Ts, T, O}
  solver::Ts
  info::T
  options::O
end
function Base.show(io::IO, ci::ConvergenceInfo)
  opt = ci.options
  info = ci.info

  println(io, "Results of minimization\n")
  println(io, "* Algorithm:")
  println(io, "  $(summary(ci.solver))")
  println(io)
  println(io, "* Candidate solution:")
  println(io, "  Final objective value:    $(@sprintf("%.2e", ci.info.minimum))")
  if haskey(info, :∇fz)
    println(io, "  Final gradient norm:      $(@sprintf("%.2e", opt.g_norm(info.∇fz)))")
  end
  println(io)
  println(io, "  Initial objective value:  $(@sprintf("%.2e", ci.info.f0))")
  if haskey(info, :∇f0)
    println(io, "  Initial gradient norm:    $(@sprintf("%.2e", info.∇f0))")
  end
  println(io)
  println(io, "* Convergence measures")
  if isa(ci.solver, NelderMead)
      nm_converged(r) = 0.0
      println(io, "  √(Σ(yᵢ-ȳ)²)/n         = $(@sprintf("%.2e", info.nm_obj)) <= $(@sprintf("%.2e", opt.nm_tol)) ($(info.nm_obj<=opt.nm_tol))")
  else
      println(io, "  |x - x'|              = $(@sprintf("%.2e", info.Δx)) <= $(@sprintf("%.2e", opt.x_abstol)) ($(info.Δx<=opt.x_abstol))")
      println(io, "  |x - x'|/|x|          = $(@sprintf("%.2e", info.Δx/info[2])) <= $(@sprintf("%.2e", opt.x_reltol)) ($(info.Δx/info[2] <= opt.x_reltol))")
      if isfinite(opt.f_limit)
        println(io, "  |f(x')|               = $(@sprintf("%.2e", info.fz)) <= $(@sprintf("%.2e", opt.f_limit)) ($(info.fz<=opt.f_limit))")
      end
      if haskey(info, :fx)
        Δf = abs(info.fx-info.fz)
        println(io, "  |f(x) - f(x')|        = $(@sprintf("%.2e", Δf)) <= $(@sprintf("%.2e", opt.f_abstol)) ($(Δf<=opt.f_abstol))")
        println(io, "  |f(x) - f(x')|/|f(x)| = $(@sprintf("%.2e", Δf/info.fx)) <= $(@sprintf("%.2e", opt.f_reltol)) ($(Δf/info.fx<=opt.f_reltol))")
      end
      if haskey(info, :∇fx) && haskey(info, :∇fz)
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

struct MinOptions{T1, T2, T3, T4, Txn, Tgn}
  x_abstol::T1
  x_reltol::T1
  x_norm::Txn
  g_abstol::T2
  g_reltol::T2
  g_norm::Tgn
  f_limit::T3
  f_abstol::T3
  f_reltol::T3
  nm_tol::T3
  maxiter::T4
  show_trace::Bool
end

MinOptions(; x_abstol=0.0, x_reltol=0.0, x_norm=x->norm(x, Inf),
             g_abstol=1e-8, g_reltol=0.0, g_norm=x->norm(x, Inf),
             f_limit=-Inf, f_abstol=0.0, f_reltol=0.0,
             nm_tol=1e-8, maxiter=10000, show_trace=false) =
  MinOptions(x_abstol, x_reltol, x_norm,
             g_abstol, g_reltol, g_norm,
             f_limit, f_abstol, f_reltol,
             nm_tol,
             maxiter, show_trace)

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
  println(io, "* Status: $(any(converged(mr)))")
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

function prepare_variables(objective, approach, x0, ∇fz, B)
    z = x0
    x = copy(z)

    if isa(B, Nothing)  # didn't provide a B
        if isa(modelscheme(approach), GradientDescent)
            # We don't need to maintain a dense matrix for Gradient Descent
            B = I
        else
            # Construct a matrix on the correct form and of the correct type
            # with the content of I_{n,n}
            B = I + abs.(0*x*x')
        end
    end
    # first evaluation
    if isa(modelscheme(approach), Newton)
        fz, ∇fz, B = objective(x, ∇fz, B)
    else
        fz, ∇fz = objective(x, ∇fz)
    end
    fx = copy(fz)
    ∇fx = copy(∇fz)
    return x, fx, ∇fx, z, fz, ∇fz, B
end

function g_converged(∇fz, ∇f0, options)
  g_converged = options.g_norm(∇fz) ≤ options.g_abstol
  g_converged = g_converged || options.g_norm(∇fz) ≤ ∇f0*options.g_reltol
  return g_converged
end

function x_converged(x, z, options)
  y = x .- z
  ynorm = options.x_norm(y)
  x_converged = ynorm ≤ options.x_abstol
  x_converged = x_converged || ynorm ≤ options.x_norm(x)*options.x_reltol
  x_converged = x_converged || any(isnan.(z))
  return x_converged
end
function f_converged(fx, fz, options)
  y = fx - fz
  ynorm = abs(y)
  f_converged = ynorm ≤ options.f_abstol
  f_converged = f_converged || ynorm ≤ abs(fx)*options.f_reltol
  f_converged = f_converged || fz ≤ options.f_limit
  f_converged = f_converged || isnan(fz)
  return f_converged
end
function converged(approach, x, z, ∇fz, ∇f0, fx, fz, options, skip=nothing)
  if approach isa TrustRegion && skip == true
    # special logic for region reduction here
    false
  else
    x_converged(x, z, options), g_converged(∇fz, ∇f0, options), f_converged(fx, fz, options)
  end
end