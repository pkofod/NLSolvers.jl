function nlsolve!(prob::NEqProblem{<:OnceDiffed}, x, method::LineSearch=LineSearch(Newton(), Static(1)), options=NEqOptions(); maxiter=2000, f_abstol=1e-8, f_reltol=1e-12)
  t0 = time()
  F = prob.residuals
  scheme, linesearch = modelscheme(method), algorithm(method)

  z, d, Fx, Jx = copy(x), copy(x), copy(x), x*x'

  Fx, Jx = F(x, Fx, Jx)
  ρF0 = norm(Fx, Inf)
  T = typeof(ρF0)
  stoptol = T(f_reltol)*ρF0 + T(f_abstol)
  if ρF0 < stoptol
    return x, Fx, 0
  end
  ρs = ρF0
  iter = 1
  while iter ≤ maxiter
    x .= z
    # Update the point of evaluation
    scheme.linsolve(d, Jx, -Fx)

    merit = MeritObjective(prob, F, Fx, Jx, d)
    # φ = LineObjective!(F, ∇fz, z, x, d, fx, dot(∇fx, d))
    # Need to restrict to static and backtracking here
    φ = LineObjective(prob, merit, nothing, z, x, d, (ρF0^2)/2, -ρF0^2)

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, T(1.0))

    z .= x .+ α.*d
    Fx, Jx = F(x, Fx, Jx)
    ρF = norm(Fx, 2)
    ρs = norm(x.-z, Inf)
    if ρF < stoptol #|| ρs <= 1e-12
        break
    end
    iter += 1
  end
  ConvergenceInfo(method, (solution=x, best_residual=Fx, ρF0=ρF0, ρs=ρs, iter=iter, time=time()-t0), options)
end
