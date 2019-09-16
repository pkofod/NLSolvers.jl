struct Secant{T, Line}
    factor::T
    shift::T
    ls::Line
end
"""
    Secant(;factor, shift)

Construct a method instance of `Secant` using the two required keywords `factor`
and `shift`. The two numbers are used to construct a pertubed iterate `x'` such
that:

    x' = x*factor + shift

where x is the initial iterate. See section 1.3 in [1] for an explanation.

References
---
[1] Kelley, C. T. (2003). Solving nonlinear equations with Newton's method (Vol. 1). Siam.
"""
Secant(; factor, shift, linesearch=Static()) = Secant(factor, shift, linesearch)


function solve(prob::NLEProblem, x0, method=Secant(0.99, 1e-1), options=NLEOptions())
    if length(x0) > 1
        throw(ArgumentError("You cannot solve your problem with the Secant method, since it is a univariate method, and `length(x0)>1` where `x0` is the initial iterate."))
    end

    T = eltype(x0)
    f_abstol, f_reltol, maxiter = options.f_abstol, options.f_reltol, options.maxiter

    # Evaluate
    xnext = x0
    Fnext = value(prob, xnext)

    # define the stopping tolerance
    f_stoptol = T(f_abstol) + T(f_reltol)*Fnext

    # Check for initial convergence
    if norm(Fnext) ≤ f_stoptol
        return (F=Fnext, x=xnext, code=:success, iter=0)
    end

    # construct the shifted iterate and evaluate the value of the residual
    xcurr = method.factor*x0+method.shift
    Fcurr = value(prob, xcurr)

    for i = 1:maxiter
        # Compute the difference
        ΔF = (Fnext-Fcurr)

        # Try to rescue the approximation
        if ΔF < eps(T)
            # Try to rescue
            xcurr = xnext
            Fcurr = Fnext

            xnext = xcurr*T(method.factor) + T(method.shift)
            Fnext = value(prob, xnext)

            ΔF = Fnext-Fcurr

            if ΔF < eps(T)
                # Failed to rescue
                return (F=Fnext, x=xnext, code=:failure_ΔF, iter=i)
            end
        end

        # calculate change in x between past two iterates
        s = - Fcurr*(xnext - xcurr)/ΔF
        Δx = s

        xcurr = xnext
        Fcurr = Fnext

        xnext = xcurr + Δx
        Fnext = value(prob, xnext)

        if norm(Fnext) ≤ f_stoptol
            return (F=Fnext, x=xnext, code=:success, iter=i)
        end
    end
    (F=Fnext, x=xnext, code=:failure_itermax, iter=options.maxiter)
end
