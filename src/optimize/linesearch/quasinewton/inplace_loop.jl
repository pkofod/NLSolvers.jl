function minimize!(objective::ObjWrapper, x0,
                   scheme::QuasiNewton, options=OptOptions(),
                   cache=preallocate_qn_caches_inplace(x0))

    minimize!(objective, (x0, nothing), (scheme, Backtracking()), options, cache)
end
function minimize!(objective::ObjWrapper, x0,
                   approach::Tuple, options=OptOptions(),
                   cache=preallocate_qn_caches_inplace(x0))

    minimize!(objective, (x0, nothing), approach, options, cache)
end
function minimize!(objective::ObjWrapper, state0::Tuple,
                   scheme::QuasiNewton, options=OptOptions(),
                   cache=preallocate_qn_caches_inplace(first(state0)))

    minimize!(objective, state0, (scheme, Backtracking()), options, cache)
end
function minimize!(objective::ObjWrapper, state0::Tuple, approach::Tuple{<:Any, <:LineSearch},
                   options::OptOptions=OptOptions(), # options
                   cache = preallocate_qn_caches_inplace(x0), # preallocate arrays for QN
                   )

    #==============
         Setup
    ==============#
    x0, B0 = state0
    T = eltype(x0)
    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(objective, approach, x0, copy(x0), B0)

    #========================
         First iteration
    ========================#
    x, fx, ∇fx, z, fz, ∇fz, B, is_converged = iterate!(cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options)

    iter = 0
    while iter <= options.maxiter && !is_converged
        iter += 1
        # take a step and update approximation
        x, fx, ∇fx, z, fz, ∇fz, B, is_converged = iterate!(cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options, false)
        if norm(x.-z, Inf) ≤ eps(T)
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
    φ = LineObjective!(objective, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1.0))

    # Calculate final step vector and update the state
    @. s = α * d
    @. z = x + s

    # Update approximation
    fz, ∇fz, B = update_obj!(objective, d, s, y, ∇fx, z, ∇fz, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(z, ∇fz, options.g_tol)

    return x, fx, ∇fx, z, fz, ∇fz, B, is_converged
end
