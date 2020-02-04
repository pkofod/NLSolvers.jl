function prepare_variables(objective, approach, x0, ∇fz, B)
    z = x0
    x = copy(z)

    if isa(B, Nothing)  # didn't provide a B
        if isa(first(approach), GradientDescent)
            # We don't need to maintain a dense matrix for Gradient Descent
            B = I
        else
            # Construct a matrix on the correct form and of the correct type
            # with the content of I_{n,n}
            B = I + abs.(0*x*x')
        end
    end
    # first evaluation
    if isa(first(approach), Newton)
        fz, ∇fz, B = objective(x, ∇fz, B)
    else
        fz, ∇fz = objective(x, ∇fz)
    end
    fx = copy(fz)
    ∇fx = copy(∇fz)
    return x, fx, ∇fx, z, fz, ∇fz, B
end

function converged(z, ∇fz, ∇f0, options)
    g_converged = norm(∇fz) ≤ options.g_abstol || norm(∇fz) ≤ ∇f0*options.g0_reltol
    return g_converged || any(isnan.(z))
end

include("inplace_loop.jl")
include("outofplace_loop.jl")
