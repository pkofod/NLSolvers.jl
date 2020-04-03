struct QNVars{T, Ty}
    d::T # search direction
    s::T # change in x
    y::Ty # change in successive gradients
end
function QNVars(x, g)
    QNVars(copy(g), copy(x), copy(x))
end
function preallocate_qn_caches_inplace(x0)
    # Maintain gradient and state pairs in QNVars
    cache = QNVars(x0, x0)
    return cache
end

function minimize(obj::ObjWrapper, x0, approach::LineSearch, options::MinOptions)
    _minimize(OutOfPlace(), MinProblem(;obj=obj), (x0, nothing), approach, options, nothing)
end
function minimize(obj::ObjWrapper, s0::Tuple, approach::LineSearch, options::MinOptions)
    _minimize(OutOfPlace(), MinProblem(obj), s0, approach, options, nothing)
end

function minimize!(obj::ObjWrapper, x0, approach::LineSearch, options::MinOptions, cache=QNVars(x0, x0))
    _minimize(InPlace(), MinProblem(;obj=obj), (x0, nothing), approach, options, cache)
end
function minimize!(obj::ObjWrapper, s0::Tuple, approach::LineSearch, options::MinOptions, cache=QNVars(first(s0), first(s0)))
    _minimize(InPlace(), MinProblem(obj), s0, approach, options, cache)
end

function _minimize(mstyle, prob::MinProblem, s0::Tuple, approach::LineSearch, options::MinOptions, cache)
    t0 = time()

    obj = prob.objective
    #==============
         Setup
    ==============#
    x0, B0 = s0
    T = eltype(x0)
    
    objvars = prepare_variables(prob, approach, x0, copy(x0), B0)
    P = initial_preconditioner(approach, x0)
    f0, ∇f0 = objvars.fz, norm(objvars.∇fz, Inf) # use user norm

    if any(initial_converged(approach, objvars, ∇f0, options))
        x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
        return ConvergenceInfo(approach, (P=P, B=B, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=0, time=time()-t0), options)
    end
    qnvars = QNVars(copy(objvars.∇fz), copy(objvars.∇fz), copy(objvars.∇fz))

    #==============================
             First iteration
    ==============================#
    objvars, P, qnvars = iterate(mstyle, qnvars, objvars, P, approach, prob, obj, options)
    iter = 1
    # Check for gradient convergence
    is_converged = converged(approach, objvars, ∇f0, options)
    while iter < options.maxiter && !any(is_converged)
        iter += 1
        #==============================
                     iterate
        ==============================#
        objvars, P, qnvars = iterate(mstyle, qnvars, objvars, P, approach, prob, obj, options, false)
        #==============================
                check convergence
        ==============================#
        is_converged = converged(approach, objvars, ∇f0, options)
    end
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    return ConvergenceInfo(approach, (P=P, B=B, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end
function iterate(mstyle::InPlace, cache, objvars, P, approach::LineSearch, prob::MinProblem, obj::ObjWrapper, options::MinOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars


    Tf = typeof(fx)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    y, d, s = cache.y, cache.d, cache.s

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update preconditioner
    P = update_preconditioner(approach, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, P, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    # Also returns final step vector and update the state
    α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1))

    @. s = α * d
    @. z = x + s

    # Update approximation
    fz, ∇fz, B, s, y = update_obj!(obj, s, y, ∇fx, z, ∇fz, B, scheme, is_first)
    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=B, Pg=Pg), P, QNVars(d, s, y)
end

function iterate(mstyle::OutOfPlace, cache, objvars, P, approach::LineSearch, prob::MinProblem, obj::ObjWrapper, options::MinOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tf = typeof(fx)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Update preconditioner
    P = update_preconditioner(approach, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction(B, P, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # # Perform line search along d
    α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s

    # Update approximation
    fz, ∇fz, B, s, y = update_obj(obj, s, ∇fx, z, ∇fz, B, scheme, is_first)

    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=B, Pg=Pg), P, QNVars(d, s, y)
end
