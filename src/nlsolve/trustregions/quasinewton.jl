function nlsolve2!(F::OnceDiffed, x, approach::Tuple{<:Newton, <:Dogleg}; maxiter=200, f_abstol=1e-8, f_reltol=1e-12)
    scheme, linesearch = approach

    function normed_residual(Jx, Fx, x)
        Fx, Jx = F(x, Fx, Jx)
        f = (norm(Fx)^2)/2
        if Jx isa Nothing
            if Fx isa Nothing
                throw(ErrorException("Error happened in trust region nlsolve"))
            end
            return f, Jx'*Fx
        else
            return f, Jx'*Fx, Jx'*Jx
        end
    end
    td = TwiceDiffed(normed_residual)
    res = minimize!(td, x, approach)
    return res[1], res[3], res[4]
end
