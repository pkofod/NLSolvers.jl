function nlsolve!(F::OnceDiffed, x, approach::Tuple=(Newton(), Static(1)); maxiter=200, f_abstol=1e-8, f_reltol=1e-12)
    scheme, linesearch = approach
    xp, d, Fx, Jx = copy(x), copy(x), copy(x), x*x'

    Fx, Jx = F(x, Fx, Jx)
    normFx = norm(Fx, 2)
    T = typeof(normFx)
    stoptol = T(f_reltol)*normFx + T(f_abstol)
    if normFx < stoptol
        return x, Fx, 0
    end
    iter = 1
    while iter ≤ maxiter
        # Update the point of evaluation
        d .= -(Jx\Fx)

        merit = MeritObjective(F, Fx, Jx, d)
        # φ = LineObjective!(F, ∇fz, z, x, d, fx, dot(∇fx, d))
        φ = LineObjective(merit, x, d, (normFx^2)/2, -normFx^2)
        # Perform line search along d
        α, f_α, ls_success = find_steplength(linesearch, φ, T(1.0))

        x .= x .+ α.*d
        Fx, Jx = F(x, Fx, Jx)
        normFx = norm(Fx, 2)
        if normFx < stoptol
            break
        end
        iter += 1
    end
    return x, Fx, iter
end
