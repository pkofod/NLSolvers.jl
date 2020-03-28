# Dogleg is appropriate here because Jx'*Jx is positive definite, and in that
# case we only need to calculate one newton step. In the secular equation version
# we need repeaed factorizations, and that is not as easy to exploit (but maybe
# there's a good shifted one out there?)
function nlsolve!(F::OnceDiffed, x, approach::TrustRegion{<:Newton, <:Any, <:Any}; maxiter=200, f_abstol=1e-8, f_reltol=1e-12)

    function normed_residual(x, Fx, Jx)
        Fx, Jx = F(x, Fx, Jx)
        f = (norm(Fx)^2)/2
        if Jx isa Nothing
            return f, Jx'*Fx
        else
            # As you may notice, this can be expensive... Because the gradient
            # is going to be very simple. May want to create a
            # special type or way to hook into trust regions here. WWe can exploit
            # that we only need the cauchy and the newton steps, not any shifted
            # systems
            return f, Jx'*Fx, Jx'*Jx
        end
    end
    td = TwiceDiffed(normed_residual)
    res = minimize!(td, x, approach, MinOptions())
    return res
end
