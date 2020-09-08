function solve!(prob::NEqProblem, x, method::LineSearch=LineSearch(Newton(), Static(1)), options=NEqOptions())
  t0 = time()
  F = prob.R
  scheme, linesearch = modelscheme(method), algorithm(method)

  z, d, Fx, Jx = copy(x), copy(x), copy(x), x*x'

  Fx, Jx = F(x, Fx, Jx)
  ρF0 = norm(Fx, Inf)
  ρ2F0 = norm(Fx, 2)
  T = typeof(ρF0)
  stoptol = T(options.f_reltol)*ρF0 + T(options.f_abstol)
  if ρF0 < stoptol
    return x, Fx, 0
  end
  ρs = ρF0
  iter = 1
  while iter ≤ options.maxiter
    x .= z
    # Update the point of evaluation
    d = scheme.linsolve(d, Jx, -Fx)

    merit = MeritObjective(prob, F, Fx, Jx, d)
    # φ = LineObjective!(F, ∇fz, z, x, d, fx, dot(∇fx, d))
    # Need to restrict to static and backtracking here
    φ = LineObjective(prob, merit, nothing, z, x, d, (ρF0^2)/2, -ρF0^2)

    # Perform line search along d
    α, f_α, ls_success = find_steplength(InPlace(), linesearch, φ, T(1.0))

    z .= x .+ α.*d
    Fx, Jx = F(x, Fx, Jx)
    ρF = norm(Fx, 2)
    ρs = norm(x.-z, Inf)
    if ρF < stoptol #|| ρs <= 1e-12
        break
    end
    iter += 1
  end
  ConvergenceInfo(method, (solution=x, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=ρs, iter=iter, time=time()-t0), options)
end
init(::NEqProblem, ::LineSearch, x) = (z=copy(x), d=copy(x), Fx=copy(x), Jx=x*x')
function solve(prob::NEqProblem, x, method::LineSearch=LineSearch(Newton(), Static(1)), options=NEqOptions(), cache=init(prob, method, x))
  t0 = time()
  F = prob.R
  scheme, linesearch = modelscheme(method), algorithm(method)

  z, d, Fx, Jx = cache

  Fx, Jx = F(x, Fx, Jx)
  ρF0 = norm(Fx, Inf)
  ρ2F0 = norm(Fx, 2)
  T = typeof(ρF0)
  stoptol = T(options.f_reltol)*ρF0 + T(options.f_abstol)
  if ρF0 < stoptol
    return x, Fx, 0
  end
  ρs = ρF0
  iter = 1
  while iter ≤ options.maxiter
    x .= z
    # Update the point of evaluation
    d = scheme.linsolve(d, Jx, -Fx)

    merit = MeritObjective(prob, F, Fx, Jx, d)
    # φ = LineObjective!(F, ∇fz, z, x, d, fx, dot(∇fx, d))
    # Need to restrict to static and backtracking here
    φ = LineObjective(prob, merit, nothing, z, x, d, (ρF0^2)/2, -ρF0^2)

    # Perform line search along d
    α, f_α, ls_success = find_steplength(InPlace(), linesearch, φ, T(1.0))

    z .= x .+ α.*d
    Fx, Jx = F(x, Fx, Jx)
    ρF = norm(Fx, 2)
    ρs = norm(x.-z, Inf)
    if ρF < stoptol #|| ρs <= 1e-12
        break
    end
    iter += 1
  end
  ConvergenceInfo(method, (solution=x, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=ρs, iter=iter, time=time()-t0), options)
end
