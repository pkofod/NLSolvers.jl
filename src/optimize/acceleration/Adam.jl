struct Adam{T}
  α::T
  β₁::T
  β₂::T
  ϵ::T
end
Adam(;alpha=0.0001, beta_mean=0.9, beta_var=0.999, epsilon=1e-8) = Adam(alpha, beta_mean, beta_var, epsilon)
struct AdaMax{T}
  α::T
  β₁::T
  β₂::T
end
AdaMax(;alpha=0.002, beta_mean=0.9, beta_var=0.999) = AdaMax(alpha, beta_mean, beta_var)


function minimize(mp::MinProblem, x0, adam::Adam, options)
  α, β₁, β₂, ϵ = adam.α, adam.β₁, adam.β₂, adam.ϵ
  t0 = time()

  fz, ∇fz = mp.objective(x0, copy(x0))
  f0, ∇f0 = fz, norm(∇fz, Inf)

  z = copy(x0)
  m = copy(∇fz)
  v = fill(zero(∇fz[1]^2), length(m))
  a = 1 - β₁
  b = 1 - β₂

  iter = 0
  is_converged = false

  while !is_converged
    iter = iter + 1
    m = β₁.*m .+ a.*∇fz
    v = β₂.*v .+ b.*∇fz.^2
    # m̂ = m./(1-β₁^t)
    # v̂ = v./(1-β₂^t)
    # x = x .- α*m̂/(sqrt.(v̂+ϵ))
    αₜ = α * sqrt(1 - β₂^iter)/(1 - β₁^iter)
    z = z .- αₜ .* m ./ (sqrt.(v) .+ ϵ)

    # ∇fz = gradient!(mp, x, ∇fz)
    fz, ∇fz = mp.objective(z, ∇fz)
    is_converged = iter >= options.maxiter
  end
  ConvergenceInfo(adam, (minimizer=z, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end
function minimize(mp::MinProblem, x0, adam::AdaMax, options)
  α, β₁, β₂ = adam.α, adam.β₁, adam.β₂
  t0 = time()

  fz, ∇fz = mp.objective(x0, copy(x0))
  f0, ∇f0 = fz, norm(∇fz, Inf) 
 
  z = copy(x0)
  m = copy(∇fz)
  u = fill(zero(∇fz[1]^2), length(m))
  a = 1 - β₁

  iter = 0
  is_converged = false

  while !is_converged
    iter = iter+1
    m = β₁.*m .+ a.*∇fz
    u = max.(β₂.*u, abs.(∇fz))
    z = z .- (α ./ (1 - β₁^iter)) .* m ./ u

    fz, ∇fz = mp.objective(z, ∇fz)
    is_converged = iter >= options.maxiter
  end
  ConvergenceInfo(adam, (minimizer=z, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end