# Notation:
# λ is the initial step length
# α current trial step length
# β current trial step length
# d is the search direction
# x is the current iterate
# f is the objective
# ϕ is the line search objective and is a function of the step length only

# TODO
# Is ∇f_x needed when we have dϕ_0? Maybe in some cases.


# For non-linear systems of equations equations we generally choose the sum-of-
# squares merit function. Some useful things to remember is:
#
# f(y) = 1/2*|| F(y) ||^2 =>
# ∇_df = -d'*J(x)'*F(x)
#
# where we remember the notation x means the current iterate here. This means
# that if we step in the Newton direction such that d is defined by
#
# J(x)*d = -F(x) => -d'*J(x)' = F(x)' =>
# ∇_df = -F(x)'*F(x) = -f(x)
#


# This file contains several implementation of what we might call "Backtracking".
# The AbstractBacktracking line searches try to satisfy the Amijo(-Goldstein)
# condition:
#     |f(x + α*d)| < (1-c_1*α)*|f(x)|
# That is: the function should

# As per [Nocedal & Wright, pp. 37] we don't have to think about the curvature
# condition as long as we use backtracking.

abstract type AbstractBacktracking end

"""
    _safe_α(α_cand, α_curr, c, ratio)
Returns the safeguarded value of α in a Amijo
backtracking line search.

σ restriction 0 < c < ratio < 1
"""
function _safe_α(α_cand, α_curr, c=0.1, ratio=0.5)
    α_cand < c*α_curr && return c*α_curr
    α_cand > ratio*α_curr && return ratio*α_curr

	α_cand # if the candidate is in the interval, just return it
end


struct Backtracking{T1, T2} <: LineSearch
    ratio::T1
	c::T1
	maxiter::T2
	verbose::Bool
end
Backtracking(; ratio=0.5, c=1e-4, maxiter=50, verbose=false) = Backtracking(ratio, c, maxiter, verbose)

function _solve_pol(ls::Backtracking, ϕ, ϕ_0, dϕ_0, α, c, ratio)
	β = α
	α = ratio*α
	ϕ_α = ϕ(α)
	β, α, ϕ_α
end

# clean this up!!
function _solve_pol!(ls::Backtracking, obj, ϕ, ν, x, d, ϕ_0, dϕ_0, α, c, ratio)
	β = α
	α = ratio*α
	@. ν = x + α*d
	ϕ_α = obj(ν) # update function value
	β, α, ϕ_α
end

struct TwoPointQuadratic{T1, T2} <: LineSearch
    ratio::T1
	c::T1
	maxiter::T2
	verbose::Bool
end

## Polynomial line search
# f(α) = ||F(xₙ+αdₙ)||²₂
# f(0) = ||F(xₙ)||²₂
# f'(0) = 2*(F'(xₙ)'dₙ)'F(xₙ) = 2*F(xₙ)'*(F'(xₙ)*dₙ) < 0
# if f'(0) >= 0, then dₙ does is not a descent direction
# for the merit function. This can happen for broyden

# two-point parabolic
# at α = 0 we know f and f' from F and J as written above
# at αc define
TwoPointQuadratic(; ratio=0.5, c=1e-4, maxiter=50, verbose=false) = TwoPointQuadratic(ratio, c, maxiter, verbose)
function twopoint(f, f_0, df_0, α, f_α, ratio)
	ρ_lo, ρ_hi = 0.1, ratio
    # get the minimum (requires df0 < 0)
	c = (f_α - f_0 - df_0*α)/α^2

	# p(α) = f0 + df_0*α + c*α^2 is the function
	# we have df_0 < 0. Then if  f_α > f(0) then c > 0
	# by the expression above, and p is convex. Then,
	# we have a minimum between 0 and α at

	γ = -df_0/(2*c) # > 0 by df0 < 0 and c > 0
    # safeguard α
    return max(min(γ, α*ρ_hi), α*ρ_lo) # σs
end

function _solve_pol(ls::TwoPointQuadratic, ϕ, ϕ_0, dϕ_0, α, f_α, ratio)
	β = α
	α = twopoint(ϕ, ϕ_0, dϕ_0, α, f_α, ratio)
	ϕ_α = ϕ(α)
	β, α, ϕ_α
end

# clean this up!!
function _solve_pol!(ls::TwoPointQuadratic, obj, ϕ, ν, x, d, ϕ_0, dϕ_0, α, f_α, ratio)
	β = α
	α = twopoint(ϕ, ϕ_0, dϕ_0, α, f_α, ratio)
	@. ν = x + α*d
	ϕ_α = obj(ν) # update function value
	β, α, ϕ_α
end


"""
    find_steplength(---)

Returns a step length, (merit) function value at step length and succes flag.
"""
function find_steplength(ls::Union{Backtracking, TwoPointQuadratic}, obj::T, d, x,
	                     λ, #
						 ϕ_0::Tf,
						 ∇f_x,
						 dϕ_0=dot(d, ∇f_x),
						 ) where {T, Tf}
	ratio, c, maxiter, verbose = Tf(ls.ratio), Tf(ls.c), ls.maxiter, ls.verbose

    if verbose
        println("Entering line search with step size: ", λ)
        println("Initial value: ", ϕ_0)
        println("Value at first step: ", obj(x+λ*d))
    end

    t = -c*dϕ_0

    iter, α, β = 0, λ, λ # iteration variables
    f_α = obj(x + α*d)   # initial function value

	is_solved = isfinite(f_α) && f_α <= ϕ_0 + c*α*t

    while !is_solved && iter <= maxiter
        iter += 1
        β, α, f_α = _solve_pol(ls, α->obj(x+α*d), ϕ_0, dϕ_0, α, f_α, ratio)
		is_solved = isfinite(f_α) && f_α <= ϕ_0 + c*α*t
    end

	ls_success = iter >= maxiter ? false : true

    if verbose
		!ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success
end
function find_steplength!(ls::Union{Backtracking, TwoPointQuadratic}, obj::T, d, x, λ, ϕ_0, ∇f_x, dϕ_0=dot(d, ∇f_x)) where T
	ratio, c, maxiter, verbose = ls.ratio, ls.c, ls.maxiter, ls.verbose

    if verbose
        println("Entering line search with step size: ", λ)
        println("Initial value: ", ϕ_0)
        println("Value at first step: ", obj(x+λ*d))
    end

    t = -c*dϕ_0

    α, β = λ, λ

    iter = 0

	# the "nu" state
	ν = x .+ α .* d

    f_α = obj(ν) # initial function value

	is_solved = isfinite(f_α) && f_α <= ϕ_0 + c*α*t
    while !is_solved && iter <= maxiter
        iter += 1
		β, α, f_α = _solve_pol!(ls, obj, α->obj(x+α*d), ν, x, d, ϕ_0, dϕ_0, α, f_α, ratio)
		is_solved = isfinite(f_α) && f_α <= ϕ_0 + c*α*t
    end

	ls_success = iter >= maxiter ? false : true

    if verbose
		!ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success
end


# The
struct ThreePointQuadratic{T1, T2} <: LineSearch
    ratio::T1
	c::T1
	maxiter::T2
	verbose::Bool
end
ThreePointQuadratic(; ratio=0.5, c=1e-4, maxiter=50, verbose=false) = ThreePointQuadratic(ratio, c, maxiter, verbose)
