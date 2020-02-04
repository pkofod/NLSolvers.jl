#===============================================================================
  Anderson acceleration is a fixed point iteration acceleration method. A fixed
  point problem is one of finding a solution to G(x) = x or alternatively to
  find a solution to F(x) = G(x) - x = 0. To allow for easy switching between
  methods, we write the code in the F(x) = 0 form, but it really is of no impor-
  tance. This makes it a bit simpler to keep a consistent naming scheme as well,
  since convergence is measured in F(x) not G(x) directly.
===============================================================================#
struct Anderson{T}
  delay::T
  memory::T
end
Anderson() = Anderson(1)
function vv_shift!(G)
  for i = 1:length(G)-1
    G[i] = G[i+1]
  end
end

                  # args
function nlsolve!(G::NonDiffed, x,
                  anderson::Anderson;
                  # kwargs
                  Gx = similar(x),
                  Fx = similar(x),
                  f_abstol=sqrt(eltype(x)))
  #==============================================================
    Do initial function iterations; default is a delay of 0,
    but we always do at least one evaluation of G, so set up AA.

    Notation: AA solves G(x) - x = 0 or F(x) = 0
  ==============================================================#  
  Gx = G(x, Gx)
  Fx .= Gx .- x
  x .= Gx
  for i = 1:anderson.delay
    Gx = G(x, Gx)
	Fx .= Gx .- x
    x .= Gx
    finite_check = isfinite(x)
    if norm(Fx) < f_abstol || !finite_check
      return (x=x, Fx=Fx, acc_iter=0, finite=finite_check)
    end
  end

  #==============================================================
    If we got this far, then the delay was not enough to con-
    verge. However, we now hope to have moved to a region where
    everything is well-behaved, and we start the acceleration.
  ==============================================================#

  n = length(x)
  memory = min(n, anderson.memory)

  Q = x*x[1:memory]'
  R = x[1:memory]*x[1:memory]'

  x_cache = similar(x)

  #==============================================================
    Start Anderson Acceleration. We use QR updates to add new
    successive changes in G to the system, and once the memory
    is exhausted, we use QR downdates to forget the oldest chan-
    ges we have stored.
  ==============================================================#
  effective_memory = 0

  G = [copy(x) for i=1:memory)]
  Δg = copy(Gx)
  Δf = copy(Fx)

  Gold = similar(x)
  for k = 1:itermax
    Gx = G(x, Gx)
	Fx .= Gx .- x
    x .= Gx

    # is this actually needed? I think we can avoid these
    @. Δg = Gx - Gold
    @. Δf = Fx - Fold

    Gold .= Gx
    Fold .= Fx

    effective_memory += 1

    # if we've exhausted the memory, downdate
    if effective_memory > m
      vv_shift!(G)
      qrdelete!(Q, R, m)
      effective_memory -= 1
    end

    # Add the latest change to G
    G[effective_memory] .= Δg

    # QR update
	qradd!(Q, R, vec(Δf), effective_memory)

	# Create views for the system depending on the effective memory counter
	Qv = view(Q, :, 1:effective_memory)
	Rv = UpperTriangular(view(R, 1:effective_memory, 1:effective_memory))

	# check condition number
    if !isa(droptol, Nothing)
        while cond(Rv) > droptol && effective_memory > 1
            qrdelete!(Q, R, effective_memory)

            effective_memory -= 1
            Qv = view(Q, :, 1:effective_memory)
            Rv = UpperTriangular(view(R, 1:effective_memory, 1:effective_memory))
        end
    end
 
    # solve least squares problem
    γv = view(cache.γv, 1:m_eff)
    ldiv!(Rv, mul!(γv, Qv', vec(Fx)))

    # update next iterate
    for i in 1:effective_memory
        @. x -= γv[i] * cache.G[i]
    end

    if !isa(beta, Nothing)
        x .= x .- (1 .- beta).*(Fx .- Qv*Rv*γv))) # this is suboptimal!
    end
  end
end

    if !isa(beta, Nothing)
        x .= x .- (1 .- beta).*(fcur .- mul!(x_cache, Qv[m_eff], mul!(m_eff_cache[m_eff], Rv[m_eff], γ[m_eff])))
    end
end
