"""
# PureRandomSearch
## Constructor
```julia
PureRandomSearch(; draw, lb, ub)
```

Defines a pure random search method sampling from `draw`. It is up to the caller
to either supply a `draw` that takes zero input arguments and outputs an iterative
consistent with the objective function. If no `draw` method is supplied, the user
must provide lower bounds `lb` and upper bounds `ub` consistent with the objective
function.

## Example

"""

struct PureRandomSearch{T, P}
    draw::T
    parstyle::P
end

function PureRandomSearch(; draw=nothing, lb=nothing, ub=nothing)
    if isa(draw, Nothing)
        if isa(lb, Nothing) || isa(ub, Nothing)
            throw(ArgumentError("If you do not provide a draw, you need to provide lower and upper bounds. See ?PureRandomSearch for more details."))
        end
        width = ub .- lb
        T = eltype(width)
        N = length(width)
        draw = () -> width .* rand(T, N) .+ lb
    end
    PureRandomSearch(draw)
end
function minimize(objective, prs::PureRandomSearch; maxiter = 100)
    xbest = prs.draw()
    fbest = objective(nothing, xbest)
    for i = 1:maxiter
        xcandidate = prs.draw()
        fcandidate = objective(nothing, xcandidate)
        if fcandidate ≤ fbest
            fbest = fcandidate
            xbest = copy(xcandidate)
        end
    end
    return (fbest = fbest, xbest = xbest)
end
