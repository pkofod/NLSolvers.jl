"""
# ProjectedNewton
## Constructor
```julia
    ProjectedNewton(; epsilon=1e-8)
```

`epsilon` determines the threshold for whether a bound is approximately active or not, see eqn. (32) in [1].

## Description
ProjectedNewton second order for bound constrained optimization. It's an active set and allows for rapid exploration of the constraint face. It employs a modified Armijo-line search that takes the active set into account. Details can be found in [1].

## References
[1] http://www.mit.edu/~dimitrib/ProjectedNewton.pdf
"""
struct ProjectedNewton{T}
    ϵ::T
end
ProjectedNewton(; epsilon=1e-12) = ProjectedNewton(epsilon)

modelscheme(::ProjectedNewton) = Newton()
"""
    diagrestrict(x, c, i)

Returns the correct element of the Hessian according to the active set and the diagonal matrix described in [1].

[1] http://www.mit.edu/~dimitrib/ProjectedNewton.pdf
"""
function diagrestrict(x::T, c, i) where T
    if !c
        # If not binding, then return the value
        return x
    else
        # If binding, then return 1 if the diagonal or 0 otherwise
        T(i)
    end
end

function is_ϵ_active(x, lower, upper, ∇fx, ϵ∇f=eltype(x)(0))
    # it is requied that ϵ ⩽ min(U_i - L_i)/2 to uniquely choose
    # an underestimate of the inactive set or else there would be
    # two ways of defining 𝓐^ϵ.
    lowerbinding = x <= lower + ϵ∇f
    upperbinding = x >= upper - ϵ∇f

    pointing_down = ∇fx >= 0
    pointing_up   = ∇fx <= 0

    lower_active = lowerbinding && pointing_down
    upper_active = upperbinding && pointing_up

    lower_active || upper_active
end

function solve(prob::MinProblem, x0, scheme::ProjectedNewton, options::MinOptions)
    t0 = time()

    x0, B0 = x0, [1.0 0.0; 0.0 1.0]
    lower, upper = bounds(prob)
    ϵ∇f = scheme.ϵ

    !any(clamp.(x0, lower, upper) .!= x0) || error("Initial guess not in the feasible region")

    linesearch = ArmijoBertsekas()
    mstyle = OutOfPlace()

    objvars = prepare_variables(prob, scheme, x0, copy(x0), B0)
    f0, ∇f0 = objvars.fz, norm(objvars.∇fz, Inf) # use user norm
    fz, ∇fz = objvars.fz, objvars.∇fz # use user norm
    fx, ∇fx = fz, copy(∇fz)
    B = B0
    x = copy(x0)
    z = copy(x0)
    Tf = typeof(fz)
    is_first=false
    Ix = Diagonal(z.*0 .+ 1)
    for i = 1:options.maxiter
        x = copy(z)
        fx = copy(fz)
        ∇fx = copy(∇fz)

        activeset = is_ϵ_active.(x, lower, upper, ∇fx, ϵ∇f)
        incactiveset = .!(activeset)

        # The hessian needs to be adapted to ignore the active region
        binding = .!(incactiveset*incactiveset')
        Hhat = diagrestrict.(inv(B), binding, Ix)

        # Update current gradient and calculate the search direction
        d = clamp.(x.-Hhat*∇fx, lower, upper).-x # solve Bd = -∇fx
        φ = _lineobjective(mstyle, prob, prob.objective, ∇fz, z, x, d, fz, dot(∇fz, d))

        # Perform line search along d
        # Also returns final step vector and update the state
        α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1), ∇fz, activeset, lower, upper, x, d, ∇fx, activeset)
        # # Calculate final step vector and update the state
        s = @. α * d
        z = @. x + s
        s = clamp.(z, lower, upper) - x
        z = x + s
        
        # Update approximation
        fz, ∇fz, B, s, y = update_obj(prob.objective, s, ∇fx, z, ∇fz, B, Newton(), is_first)
        if norm(x.-clamp.(x.-∇fx, lower, upper), Inf) < 1e-6
            return z, fz, i
        end
    end
    @show z.-min.(upper, max.(z.-∇fz, lower))
  z, fz, options.maxiter
end

"""
# ArmijoBertsekas
## Constructor
```julia
    ArmijoBertsekas()
```
## Description
ArmijoBertsekas is the modified Armijo backtracking line search described in [1]. It takes into account whether an element of the gradient is active or not.

## References
[1] http://www.mit.edu/~dimitrib/ProjectedNewton.pdf
"""
struct ArmijoBertsekas{T1, T2, T3, TR} <: LineSearcher
    ratio::T1
    decrease::T1
    maxiter::T2
    interp::T3
    steprange::TR
    verbose::Bool
end
ArmijoBertsekas(; ratio=0.5, decrease=1e-4, maxiter=50,
               steprange=(0.0, Inf), interp=FixedInterp(),
               verbose=false) =
 ArmijoBertsekas(ratio, decrease, maxiter, interp, steprange, verbose)

function find_steplength(mstyle, ls::ArmijoBertsekas, φ::T, λ, ∇fx, Ibool, lower, upper, x, p, g, activeset) where T
    #== unpack ==#
    φ0, dφ0 = φ.φ0, φ.dφ0
    Tf = typeof(φ0)
    ratio, decrease, maxiter, verbose = Tf(ls.ratio), Tf(ls.decrease), ls.maxiter, ls.verbose

    #== factor in Armijo condition ==#
    t0 = decrease*dφ0 # dphi0 should take into account the active set
    iter, α, β = 0, λ, λ # iteration variables
    f_α = φ(α)   # initial function value
    x⁺ = retract.(lower, upper, x, p, α)

    if verbose
        println("Entering line search with step size: ", λ)
        println("Initial value: ", φ0)
        println("Value at first step: ", f_α)
    end
    is_solved = isfinite(f_α) && f_α <= φ0 - decrease*sum(bertsekas_R.(x, x⁺, g, p, α, activeset))
    while !is_solved && iter <= maxiter
        iter += 1
        β, α, f_α = interpolate(ls.interp, φ, φ0, dφ0, α, f_α, ratio)
        x⁺ = retract.(lower, upper, x, p, α)
        is_solved = isfinite(f_α) && f_α <= φ0 - decrease*sum(bertsekas_R.(x, x⁺, g, p, α, activeset))
    end

    ls_success = iter >= maxiter ? false : true

    if verbose
        !ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success
end

bertsekas_R(x, x⁺, g, p, α, i) = i ? g*(x-x⁺) : α*p*g
# defined univariately
retract(lower, upper, x, p, α) = min(upper, max(lower, x-α*p))