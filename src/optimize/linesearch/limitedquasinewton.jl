struct TwoLoopVars{T, TR, M}
    d::T # search direction
    S::TR # change in x
    Y::TR # change in successive gradients
    α::M
    ρ::M
end
function TwoLoopVars(x, memory)
    d = copy(x)

    S = [copy(x) for i = 1:memory]
    Y = [copy(x) for i = 1:memory]
    α = similar(x, memory)
    ρ = similar(x, memory)
    TwoLoopVars(d, S, Y, α, ρ)
end
lbfgs_vars(method, x) = TwoLoopVars(x, method.memory)
function minimize(obj::ObjWrapper, x0, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::MinOptions)
    minimize(obj, (x0, nothing), approach, options)
end
function minimize(obj::ObjWrapper, s0::Tuple, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::MinOptions)
    _minimize(OutOfPlace(), obj, s0, approach, options, nothing)
end

function minimize!(obj::ObjWrapper, x0, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::MinOptions, cache=preallocate_qn_caches_inplace(x0))
    minimize!(obj, (x0, nothing), approach, options, cache)
end
function minimize!(obj::ObjWrapper, s0::Tuple, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::MinOptions, cache=preallocate_qn_caches_inplace(first(s0)))
    _minimize(InPlace(), obj, s0, approach, options, cache)
end

function _minimize(mstyle, obj::ObjWrapper, s0::Tuple, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, options::MinOptions, cache)
    t0 = time()
    #==============
         Setup
    ==============#
    x0, B0 = s0
    T = eltype(x0)
    # Remove B from all of this
    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(obj, approach, x0, copy(x0), B0)

    qnvars = lbfgs_vars(modelscheme(approach), x)
    ∇f0 = norm(∇fz, Inf) 
    f0 = fz
    P = initial_preconditioner(approach, x0)

    #========================
         First iteration
    ========================#
    x, fx, ∇fx, z, fz, ∇fz, qnvars, P = iterate(mstyle, 1, qnvars, x, fx, ∇fx, z, fz, ∇fz, P, approach, obj, options)
    iter = 1

    # Check for gradient convergence
    is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options)
    while iter <= options.maxiter && !any(is_converged)
        iter += 1
        # take a step and update approximation
        x, fx, ∇fx, z, fz, ∇fz, qnvars, P = iterate(mstyle, iter, qnvars, x, fx, ∇fx, z, fz, ∇fz, P, approach, obj, options, false)
        # Check for gradient convergence
        is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options)
    end

    return ConvergenceInfo(approach, (P=P,B=B, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end
function iterate(mstyle::InPlace, iter::Integer, qnvars::TwoLoopVars, x, fx::Tf, ∇fx, z, fz, ∇fz, P, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, obj, options, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch, K = modelscheme(approach), algorithm(approach), approach.scaling # make grabber and get a better name than K
    current_memory = min(iter-1, scheme.memory)
    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update preconditioner
    P = update_preconditioner(approach, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction!(scheme, copy(∇fz), qnvars, current_memory, K, P) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    s = current_memory == length(qnvars.S) ? qnvars.S[1] : qnvars.S[1+current_memory]
    @. s = α * d
    @. z = x + s
    # Update approximation
    fz, ∇fz, qnvars = update_obj!(obj, qnvars, α, x, ∇fx, z, ∇fz, current_memory, scheme)
    return x, fx, ∇fx, z, fz, ∇fz, qnvars, P
end

function iterate(mstyle::OutOfPlace, cache, x, fx::Tf, ∇fx, z, fz, ∇fz, B, P, approach::LineSearch{<:LBFGS{<:Inverse, <:TwoLoop}, <:Any, <:QNScaling}, obj, options, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Update preconditioner
    P = update_preconditioner(approach, x, P)
    # Update current gradient and calculate the search direction
    d = find_direction(B, P, ∇fx, scheme) # solve Bd = -∇fx
    φ = _lineobjective(mstyle, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s

    # Update approximation
    fz, ∇fz, B = update_obj(obj, s, ∇fx, z, ∇fz, B, scheme, is_first)

    return x, fx, ∇fx, z, fz, ∇fz, B, P
end
