struct ASA
end
function _minimize(mstyle, obj::ObjWrapper, s0::Tuple, approach::ASA, options::MinOptions, cache)
    t0 = time()
    #==============
         Setup
    ==============#
    x0, B0 = s0
    T = eltype(x0)
    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(obj, approach, x0, copy(x0), B0)
    ∇f0 = norm(∇fz, Inf) 
    f0 = fz
    
    #========================
         First iteration
    ========================#
    x, fx, ∇fx, z, fz, ∇fz, B = iterate(mstyle, cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, obj, options)
    iter = 1
    # Check for gradient convergence
    is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options)
    while iter <= options.maxiter && !any(is_converged)
        iter += 1
        # take a step and update approximation
        x, fx, ∇fx, z, fz, ∇fz, B = iterate(mstyle, cache, x, fx, ∇fx, z, fz, ∇fz, B, approach, obj, options, false)
        # Check for gradient convergence
        is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options)
    end

    return ConvergenceInfo(approach, (B=B, Δx=norm(x.-z), ρx=norm(x), z=z, fx=fx, fz=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end
function iterate(mstyle::InPlace, cache, x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach::ASA, obj, options, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = modelscheme(approach), algorithm(approach)

    y, d, s = cache.y, cache.d, cache.s

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # Calculate final step vector and update the state
    @. s = α * d
    @. z = x + s

    # Update approximation
    fz, ∇fz, B = update_obj!(obj, d, s, y, ∇fx, z, ∇fz, B, scheme, is_first)
    return x, fx, ∇fx, z, fz, ∇fz, B
end

function iterate(mstyle::OutOfPlace, cache, x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach::ASA, obj, options, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Update current gradient and calculate the search direction
    d = find_direction(B, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s

    # Update approximation
    fz, ∇fz, B = update_obj(obj, d, s, ∇fx, z, ∇fz, B, scheme, is_first)

    return x, fx, ∇fx, z, fz, ∇fz, B
end
