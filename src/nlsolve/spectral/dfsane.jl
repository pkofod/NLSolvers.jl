#===============================================================================
 Robust NonMonotone Line search
 From DF-Sane paper https://www.ams.org/journals/mcom/2006-75-255/S0025-5718-06-01840-0/S0025-5718-06-01840-0.pdf
 This line search doesn't fit the rest of the API so I need to come up with an
 interface for nonmonotone line searches (it has to accept an fbar (or Q in Hager
 Zhang's notion) and a forcing term)
===============================================================================#
struct RNMS end

function find_steplength(rnms::RNMS, φ, φ0, fbar, ηk::T, τmin, τmax) where T
  α₊ = T(1)
  α₋ = -α₊
  γ = T(1)/4
  for k = 1:100
    φα₊ = φ(α₊)
    if φα₊ ≤ fbar + ηk - γ*α₊^2*φ0
      return α₊, φα₊
    end
    φα₋ = φ(α₋)
    if φα₋ ≤ fbar + ηk - γ*α₋^2*φ0
      return α₋, φα₋
    end
    αt = α₊^2*φ0/(φα₊ + (2*α₊-1)*φ0)
    if αt < τmin*α₊
      α₊ = τmin*α₊
    elseif αt > τmax*α₊
      α₊ = τmax*α₊
    else
      α₊ = αt
    end

    # This needs to be safe guarded!
    αt = α₋^2*φ0/(φα₋ + (2*α₋-1)*φ0)
    if αt < τmin*α₋
      α₋ = τmin*α₋
    elseif αt > τmax*α₋
      α₋ = τmax*α₋
    else
      α₋ = αt
    end
  end
  throw(ErrorException("Line search failed."))
end

function safeguard_σ(σ::T, σmin, σmax, F) where T
  if abs(σ) < σmin || abs(σ) > σmax
    normF = norm(F)
    if normF > T(1)
      σ = T(1)
    elseif T(1)/10^5 ≤ normF ≤ T(1)
      σ = inv(normF)
    elseif normF < T(1)/10^5
      σ = T(10)^5
    end
  end
  return σ
end

struct DFSANE end
function nlsolve!(prob::NEqProblem, x0, scheme::DFSANE, options::NEqOptions)
  F = prob.residuals

  T = eltype(x0)

  σmin, σmax = 1e-10, 1e10
  τmin, τmax = T(1)/10, T(1)/2

  nexp = 2

  fvals = @SVector(T[])

  x = copy(x0)
  Fx = copy(x0)
  Fx = value(prob, x, Fx)
  y = copy(Fx)
  normFx0 = norm(Fx)
  fx = normFx0^nexp
  fvals = push(fvals, fx)
  σ₀ = T(1)
  σ = safeguard_σ(σ₀, σmin, σmax, Fx)
  for j = 1:10^5
    z=copy(x) # FIXME
    fbar = maximum(x->fvals[x], max(1, length(fvals)-5+1):length(fvals))
    d = -σ*Fx
    ηk = normFx0/(1+j)^2
#    φ = LineObjective(prob, OnceDiffed(, Fx, z, x, d, fx, 0*fx)
    φ(α) = norm(F(x.+α.*d, Fx, nothing))^nexp
    φ0 = φ(0)
    α, φα = find_steplength(RNMS(), φ, φ0, fbar, ηk, τmin, τmax)
    # LineObjective(x) should return z as well!
    s = α*d
    x .= x.+s
    y .= Fx
    Fx = value(prob, x, Fx)
    y .-= Fx
    fx = norm(Fx)^nexp
    fvals = push(fvals, fx)
    σ = norm(s)^2/dot(s, y)
    σ = safeguard_σ(σ, σmin, σmax, Fx)
    if sqrt(φα) < 1e-8
      break
    end
  end
  x, Fx
end
