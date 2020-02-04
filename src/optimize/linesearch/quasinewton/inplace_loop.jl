function minimize!(obj::ObjWrapper, x0,
                   scheme::Union{GradientDescent{<:Direct}, GradientDescent{<:Inverse}, QuasiNewton}, options=MinOptions(),
                   cache=preallocate_qn_caches_inplace(x0))

    minimize!(obj, (x0, nothing), (scheme, Backtracking()), options, cache)
end
function minimize!(obj::ObjWrapper, x0,
                   approach::Tuple{<:QuasiNewton, <:LineSearch}, options=MinOptions(),
                   cache=preallocate_qn_caches_inplace(x0))

    minimize!(obj, (x0, nothing), approach, options, cache)
end
function minimize!(obj::ObjWrapper, state0::Tuple,
                   scheme::QuasiNewton, options=MinOptions(),
                   cache=preallocate_qn_caches_inplace(first(state0)))

    minimize!(obj, state0, (scheme, Backtracking()), options, cache)
end
function minimize!(obj::ObjWrapper, state0::Tuple, approach::Tuple{<:QuasiNewton, <:LineSearch},
                   options::MinOptions=MinOptions(), # options
                   cache = preallocate_qn_caches_inplace(x0), # preallocate arrays for QN
                   )

    #==============
         Setup
    ==============#
    x0, B0 = state0
    T = eltype(x0)
    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(obj, approach, x0, copy(x0), B0)
    ∇f0 = norm(∇fz, Inf) 

    #========================
         First iteration
    ========================#
    x, fx, ∇fx, z, fz, ∇fz, B, is_converged = iterate!(cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, obj, options, ∇f0)

    iter = 1
    while iter <= options.maxiter && !is_converged
        iter += 1
        # take a step and update approximation
        x, fx, ∇fx, z, fz, ∇fz, B, is_converged = iterate!(cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, obj, options, ∇f0, false)
        if norm(x.-z, Inf) ≤ eps(T)
            break
        end
    end
    return z, fz, ∇fz, iter
end

function iterate!(cache, x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach::Tuple{<:Any, <:LineSearch}, obj, options, ∇f0, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach
    mstyle = InPlace()
    y, d, s = cache.y, cache.d, cache.s

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1.0))

    # Calculate final step vector and update the state
    @. s = α * d
    @. z = x + s

    # Update approximation
    fz, ∇fz, B = update_obj!(obj, d, s, y, ∇fx, z, ∇fz, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(z, ∇fz, ∇f0, options)

    return x, fx, ∇fx, z, fz, ∇fz, B, is_converged
end
