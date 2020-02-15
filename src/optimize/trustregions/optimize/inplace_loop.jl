function minimize!(objective::ObjWrapper, s0::Tuple, approach::TrustRegion, options::MinOptions)
    t0 = time()
    x0, B0 = s0
    T = eltype(x0)
    Δmin = sqrt(eps(T))

    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(objective, approach, x0, copy(x0), B0)
    p = copy(x)

    ∇f0 = norm(∇fz, Inf)
    f0 = copy(fz)


    Δk = T(20.0)
    x, fx, ∇fx, z, fz, ∇fz, B, Δkp1, accept = iterate!(p, x, fx, ∇fx, z, fz, ∇fz, B, Δk, approach, objective, options)

    iter = 1
    # Check for gradient convergence
    is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options, accept)
    while iter <= options.maxiter && !any(is_converged)
        iter += 1
        x, fx, ∇fx, z, fz, ∇fz, B, Δkp1, accept = iterate!(p, x, fx, ∇fx, z, fz, ∇fz, B, Δkp1, approach, objective, options)
        # Check for gradient convergence
        is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options, accept)
    end
    return ConvergenceInfo(approach, (Δx=norm(x.-z), ρx=norm(x), z=z, fx=fx, fz=fz, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=iter, time=time()-t0), options)
end


function iterate!(p, x, fx, ∇fx, z, fz, ∇fz, Bx, Δk, approach::TrustRegion, objective, options, scale=false)
    T = eltype(x)
    scheme, subproblemsolver = modelscheme(approach), algorithm(approach)

    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)
    spr = subproblemsolver(∇fx, Bx, Δk, p, scheme; abstol=1e-10)
    Δm = -spr.mz
    # Grab the model value, m. If m is zero, the solution, z, does not improve
    # the model value over x. If the model is not converged, but the optimal
    # step is inside the trust region and gives a zero improvement in the objec-
    # tive value, we may conclude that "something" is wrong. We might be at a
    # ridge (positive-indefinite case) for example, or the scaling of the model
    # is such that we cannot satisfy ||∇f|| < tol.
    if abs(spr.mz) < eps(T)
        # set flag to check for problems
    end

    # add Retract
    z .= x + p

    # Update before acceptance, to keep adding information about the hessian
    # even when the step is not "good" enough.
    fz, ∇fz, B = update_obj(objective, spr.p, ∇fx, z, ∇fz, Bx, scheme, scale)

    # Δf is often called ared or Ared for actual reduction. I prefer "change in"
    # f, or Delta f.
    Δf = fx - fz

    # Calculate the ratio of actual improvement over predicted improvement.
    R = Δf/Δm

    Δkp1, reject_step = update_trust_region(spr, R, p)

    if reject_step
        z .= x
        fz = fx
        ∇fz .= ∇fx
    end

    return x, fx, ∇fx, z, fz, ∇fz, B, Δkp1, reject_step
end

function update_trust_region(spr, R, p)
    T = eltype(p)
    # Chosing a parameter > 0 might be preferable here. See p. 4 of Yuans survey
    # We want to avoid cycles, but we also need something that takes very small
    # steps when convergence is hard to achieve.
    α = T(0)/7 # acceptance ratio
    t2 = T(1)/4
    t3 = t2 # could differ!
    t4 = T(1)/2
    λ34 = T(0)/2
    γ = T(2.5) # gamma for grow
    λγ = T(0)/2 # distance along growing interval ∈ (0, 1]
    Δmax = T(10)^5 # restrict the largest step
    σ = T(1)/4

    Δk = spr.Δ
    # We accept all steps larger than α ∈ [0, 1/4). See p. 415 of [SOREN] and
    # p.79 as well as  Theorem 4.5 and 4.6 of [N&W]. A α = 0 might cycle,
    # see p. 4 of [YUAN].
    if !(α <= R)
        if spr.interior
            # If you reject an interior solution, make sure that the next
            # delta is smaller than the current step. Otherwise you waste
            # steps reducing Δk by constant factors while each solution
            # will be the same.
            Δkp1 = σ * norm(p)
        else
            Δkp1 = λ34*norm(p, 2)*t3 + (1-λ34)*Δk*t4
        end
        reject_step = true
    else
        # While we accept also the steps in the case that α <= Δf < t2, we do not
        # trust it too much. As a result, we restrict the trust region radius. The
        # new trust region radius should be set to a radius Δkp1 ∈ [t3*||d||, t4*Δk].
        # We use the number λ34 ∈ [0, 1] to move along the interval. [N&W] sets
        # λ34 _= 1 and t4 = 1/4, see Algorithm 4.1 on p. 69.
        if R < t2
            Δkp1 = λ34*norm(p, 2)*t3 + (1-λ34)*Δk*t4
        else
            Δkp1 = min(λγ*Δk+(1-λγ)*Δk*γ, Δmax)
        end
        reject_step = false
    end
    return Δkp1, reject_step
end
