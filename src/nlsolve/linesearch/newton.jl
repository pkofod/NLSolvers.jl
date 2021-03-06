# make this keyworded?
init(::NEqProblem, ::LineSearch, x) = (z=copy(x), d=copy(x), Fx=copy(x), Jx=x*x')
# the bang is just potentially inplace x and state. nonbang copies these
function solve(problem::NEqProblem, x, method::LineSearch=LineSearch(Newton(), Static(1)), options=NEqOptions(), state=init(problem, method, x))
    t0 = time()

    # Unpack
    scheme, linesearch = modelscheme(method), algorithm(method)
    # Unpack important objectives
    F = problem.R.F
    FJ = problem.R.FJ
    # Unpack state
    z, d, Fx, Jx = state
    T = eltype(Fx)


    # Set up MeritObjective. This defines the least squares
    # objective for the line search.
    merit = MeritObjective(problem, F, FJ, Fx, Jx, d)
    meritproblem = OptimizationProblem(merit, nothing, Euclidean(0), nothing, mstyle(problem), nothing)
    # Evaluate the residual and Jacobian
    Fx, Jx = FJ(Fx, Jx, x)
    ρF0, ρ2F0 = norm(Fx, Inf),  norm(Fx, 2)

    stoptol = T(options.f_reltol)*ρF0 + T(options.f_abstol)
    if ρF0 < stoptol
        return ConvergenceInfo(method, (solution=x, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=T(NaN), iter=0, time=time()-t0), options)
    end

    # Create variable for norms but keep the first ones for printing purposes.
    ρs, ρ2F = ρF0, ρ2F0

    iter = 1
    while iter ≤ options.maxiter
        # Shift z into x
        if mstyle isa InPlace
            x .= z
        else
            x = copy(z)
        end
        # Update the search direction
        if mstyle isa InPlace
            d = scheme.linsolve(d, Jx, -Fx)
        else
            d = scheme.linsolve(Jx, -Fx)
        end
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
        α, f_α, ls_success = find_steplength(mstyle, linesearch, φ, T(1))

        # # Calculate final step vector and update the state

        # Step in the direction α*d
        z = retract(problem, z, x, d, α)

        # Update residual and jacobian
        Fx, Jx = FJ(Fx, Jx, z)

        # Update 2-norm for line search conditions: ϕ(0) and ϕ'(0)
        ρ2F = norm(Fx, 2)

        # Update the largest successive change in the iterate
        ρs = mapreduce(x->abs(x[1]-x[2]), max, zip(x,z)) # norm(x.-z, Inf)

        if ρ2F < stoptol || ρs <= 1e-12
            break
        end
        iter += 1
    end
    return ConvergenceInfo(method, (solution=z, best_residual=Fx, ρF0=ρF0, ρ2F0=ρ2F0, ρs=ρs, iter=iter, time=time()-t0), options)
end
