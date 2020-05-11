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
ProjectedNewton(; epsilon=1e-8)

"""
    diagrestrict(x, c, i)

Returns the correct element of the Hessian according to the active set and the diagonal matrix described in [1].

[1] http://www.mit.edu/~dimitrib/ProjectedNewton.pdf
"""
function diagrestrict(x::T, c, i)
    if !c
        # If not binding, then return the value
        return x
    else
        # If binding, then return 1 if the diagonal or 0 otherwise
        T(i)
    end
end

function isactive(x, lower, upper, ∇fx, ϵ=eltype(x)(0))
    lowerbinding  = x   <= lower + ϵ∇f
    pointing_down = ∇fx >= 0
    lower_active = lowerbinding && pointing_down

    upperbinding = x   >= upper + ϵ∇f
    pointing_up  = ∇fx <= 0
    upper_active = upperbinding && pointing_up

    lower_active || upper_active
end
function minimize(prob::MinProblem, s0, scheme::ProjectedNewton, options::MinOptions)
    t0 = time()

    x0, H0 = s0
    lower, upper = bounds(prob)
    ϵ∇f = approach

    clamp.(x, lower, upper) == x || error("Initial guess not in the feasible region")

    linesearch = ArmijoBertsekas()
    mstyle = OutOfPlace()

    for i = 1:options.maxiter
        activeset = is_active.(x, lower, upper, ∇fx, ϵ∇f)

        Ix = Diagonal(copy(x).*0 .+ 1)

        # The hessian needs to be adapted to ignore the active region
        binding = Diagonal(activeset)
        Hhat = diagrestrict.(B, binding, Ix)

        # set gradient for search direction to 0 if the element is active
        ∇fxc = ∇fx .* sl

        # Update current gradient and calculate the search direction
        d = find_direction(Hhat, ∇fxc, scheme) # solve Bd = -∇fx
        φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

        # Perform line search along d
        # Also returns final step vector and update the state
        α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1))
        # # Perform line search along d

        # Should the line search project at each step?
        α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

        # # Calculate final step vector and update the state
        s = @. α * d
        z = @. x + s
        z = clamp.(z, lower, upper)
        s = @. z - x

        # Update approximation
        fz, ∇fz, B = update_obj(objective, d, s, ∇fx, z, ∇fz, B, scheme, is_first)
        ∇fzc = ∇fz .* sl
        # Check for gradient convergence
        is_converged = converged(z, ∇fzc, options.g_tol)
        #@show sl ∇fz z s
    end

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
function find_steplength(mstyle, ls::ArmijoBertsekas, φ::T, λ) where T
    #== unpack ==#
    φ0, dφ0 = φ.φ0, φ.dφ0, 
    Tf = typeof(φ0)
    ratio, decrease, maxiter, verbose = Tf(ls.ratio), Tf(ls.decrease), ls.maxiter, ls.verbose

    #== factor in Armijo condition ==#
    t = -decrease*dφ0

    iter, α, β = 0, λ, λ # iteration variables
    f_α = φ(α)   # initial function value

    if verbose
        println("Entering line search with step size: ", λ)
        println("Initial value: ", φ0)
        println("Value at first step: ", f_α)
    end
    is_solved = isfinite(f_α) && f_α <= φ0 + α*t
    while !is_solved && iter <= maxiter
        iter += 1
        β, α, f_α = interpolate(ls.interp, φ, φ0, dφ0, α, f_α, ratio)
        is_solved = isfinite(f_α) && f_α <= φ0 + α*t
    end

    ls_success = iter >= maxiter ? false : true

    if verbose
        !ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success
end

  