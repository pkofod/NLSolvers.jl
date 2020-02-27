# Nocedal (1980) based on 
# Specialized S and Y matrices are constructed and updated
struct TwoLoop end
struct CompactLimited end
struct LBFGS{TA, F, T, TP}
  approx::TA
	type::F
  memory::T
  P::TP
end
hasprecon(::LBFGS{<:Inverse, <:Any, <:Any, <:Nothing}) = NoPrecon()
hasprecon(::LBFGS{<:Inverse, <:Any, <:Any, <:Any}) = HasPrecon()

LBFGS() = LBFGS(Inverse(), TwoLoop(), 5, nothing)
LBFGS(approx) = LBFGS(approx, TwoLoop(), 5, nothing)
"""
	q holds gradient at current state
	history is a named tuple of S and Y
	memory is the number of past (s,y) pairs we have
"""
function find_direction!(scheme::LBFGS{<:Inverse, <:TwoLoop}, q, 
                  qnvars,
                  memory,
                  scaling,
                  precon)
    d = qnvars.d
    T = eltype(d)
    S, Y = qnvars.S, qnvars.Y
	  α, ρ = qnvars.α, qnvars.ρ
    # Backward pass
    @inbounds for i in memory:-1:1
        α[i] = ρ[i] * real(dot(S[i], q))
        q .-= α[i] .* Y[i]
    end

    # Copy scaled or preconditioned q into s for forward pass
    # Actually, could we do scaling each iteration? Like in BFGS
    if memory > 0
      if scaling isa InitialScaling{<:ShannoPhua} # we need a pair to scale
          k = scaling(S[memory], Y[memory])
          @. d = k*q
      elseif !(precon isa Nothing)
          ldiv!(d, precon, q)
      end
    end
    # Forward pass
    @inbounds for i in 1:memory
        β = ρ[i] * real(dot(Y[i], d))
        d .+= S[i] .* (α[i] - β)
    end
    # Negate search direction
    negate!(d)
end
function update_obj!(objective, qnvars, α, x, ∇fx, z, ∇fz, current_memory, scheme::LBFGS{<:Inverse, <:TwoLoop}, scale=nothing)
    # Calculate final step vector and update the state
    fz, ∇fz = objective(z, ∇fz)
    # add Project gradient

    # Quasi-Newton update
    qnvars = update!(scheme, qnvars, ∇fx, ∇fz, current_memory)

    return fz, ∇fz, qnvars
end
@inbounds function update!(scheme::LBFGS{<:Inverse, <:TwoLoop}, qnvars, ∇fx, ∇fz, current_memory)
  S, Y, ρ = qnvars.S, qnvars.Y, qnvars.ρ
  n = length(S)
  m = min(n, 1+current_memory)
  if current_memory == n
    s1, y1 = S[1], Y[1] # hoisting, no allocation
    for i = 2:n
      S[i-1] = S[i]
      Y[i-1] = Y[i]
      ρ[i-1] = ρ[i]
    end
    S[end] = s1
    Y[end] = y1
  end
  @. Y[m] = ∇fz - ∇fx
  ρ[m] = 1/dot(S[m], Y[m])
  TwoLoopVars(qnvars.d, S, Y, qnvars.α, ρ) 
end