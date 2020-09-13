# make this keyworded?
init(::NEqProblem, ::LineSearch, x) = (z=copy(x), d=copy(x), Fx=copy(x), Jx=x*x')
# the bang is just potentially inplace x and state. nonbang copies these
function solve!(prob::NEqProblem, x, method::LineSearch=LineSearch(Newton(), Static(1)), options=NEqOptions(), state=init(prob, method, x))
    t0 = time()
    iip = is_inplace(prob)
    # Unpack important objectives
    F = prob.R.F
    FJ = prob.R.FJ

    # Unpack method
    scheme, linesearch = modelscheme(method), algorithm(method)

    z, d, Fx, Jx = state
    T = eltype(Fx)

    # Set up MeritObjective. This defines the least squares
    # objective for the line search.
    merit = MeritObjective(prob, F, FJ, Fx, Jx, d)
    meritproblem = OptimizationProblem(merit)
    # Evaluate the residual and Jacobian
    Fx, Jx = FJ(x, Fx, Jx)
    ρF0, ρ2F0 = norm(Fx, Inf),  norm(Fx, 2)

    stoptol = T(options.f_reltol)*ρF0 + T(options.f_abstol)
    if ρF0 < stoptol
        return x, Fx, 0
    end

    # Create variable for norms but keep the first ones for printing purposes.
    ρs, ρ2F = ρF0, ρ2F0

    iter = 1
    while iter ≤ options.maxiter
        # Shift z into x
        x .= z

        # Update the search direction
        d = scheme.linsolve(d, Jx, -Fx)

        # Need to restrict to static and backtracking here because we don't allow
        # for methods that calculate the gradient of the line objective.
        #
        # For non-linear systems of equations we choose the sum-of-
        # squares merit function. Some useful things to remember is:
        #
        # f(y) = 1/2*|| F(y) ||^2 =>
        # ∇_df = -d'*J(x)'*F(x)
        #
        # where we remember the notation x means the current iterate and y is any
        # proposal. This means that if we step in the Newton direction such that d
        # is defined by
        #
        # J(x)*d = -F(x) => -d'*J(x)' = F(x)' =>
        # ∇_df = -F(x)'*F(x) = -f(x)*2
        #
        # φ = LineObjective!(F, ∇fz, z, x, d, fx, dot(∇fx, d))
        φ = LineObjective(meritproblem, z, z, x, d, (ρ2F^2)/2, -ρ2F^2)

        # Perform line search along d
        α, f_α, ls_success = find_steplength(InPlace(), linesearch, φ, T(1.0))

        # Step in the direction α*d
        z .= x .+ α.*d

        # Update residual and jacobian
        Fx, Jx = FJ(z, Fx, Jx)

        # Update 2-norm for line search conditions: ϕ(0) and ϕ'(0)
        ρ2F = norm(Fx, 2)

        # Update the largest successive change in the iterate
        ρs = mapreduce(x->abs(x[1]-x[2]), max, zip(x,z)) # norm(x.-z, Inf)

        if ρ2F < stoptol || ρs <= 1e-12
            break
        end
        iter += 1
    end
    return ConvergenceInfo(method, (solution=x, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=ρs, iter=iter, time=time()-t0), options)
end
