function minimize!(objective::ObjWrapper, x0, scheme::QuasiNewton, B0=nothing, options=OptOptions(), cache=preallocate_qn_caches_inplace(x0))
    minimize!(objective, x0, (scheme, Backtracking()), B0, options, cache )
end
function minimize!(objective::ObjWrapper, x0, approach::Tuple{<:Any, <:LineSearch}, B0=nothing,
                   options::OptOptions=OptOptions(), # options
                   cache = preallocate_qn_caches_inplace(x0), # preallocate arrays for QN
                   )

    T = eltype(x0)
    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(objective, approach, x0, copy(x0), B0)

    # first iteration
    x, fx, ∇fx, z, fz, ∇fz, B, is_converged = iterate!(cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options)

    iter = 0
    while iter <= options.maxiter && !is_converged
        iter += 1
        # take a step and update approximation
        x, fx, ∇fx, z, fz, ∇fz, B, is_converged = iterate!(cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options, false)
        if norm(x.-z, Inf) == T(0)
            break
        end
    end
    return z, fz, ∇fz, iter
end

function iterate!(cache, x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach::Tuple{<:Any, <:LineSearch}, objective, options, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach

    y, d, s = cache.y, cache.d, cache.s

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, ∇fx, scheme) # solve Bd = -∇fx

    # Perform line search along d
    α, f_α, ls_success = find_steplength!(linesearch, objective, d, x, Tf(1.0), fx, ∇fx)

    # Calculate final step vector and update the state
    @. s = α * d
    @. z = x + s

    # Update approximation
    fz, ∇fz, B = update_obj!(objective, d, s, y, ∇fx, z, ∇fz, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(z, ∇fz, options.g_tol)

    return x, fx, ∇fx, z, fz, ∇fz, B, is_converged
end
