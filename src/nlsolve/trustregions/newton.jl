# Dogleg is appropriate here because Jx'*Jx is positive definite, and in that
# case we only need to calculate one newton step. In the secular equation version
# we need repeaed factorizations, and that is not as easy to exploit (but maybe
# there's a good shifted one out there?)
nlsolve!(F::OnceDiffed, x, approach::TrustRegion{<:Newton, <:Any, <:Any}, options::NEqOptions) = 
  solve!(NEqProblem(F), x, approach, options)
struct NormedResiduals{Tx, Tfx, Tf}
  x::Tx
  Fx::Tfx
  F::Tf
end
function (nr::NormedResiduals)(x, Fx, Jx)
  Fx, Jx = nr.F(x, Fx, Jx)
  if nr.x !== nothing && !all(nr.x .== x)
    nr.x .= x
    nr.Fx .= Fx
  end
  f = (norm(Fx)^2)/2
  if Jx isa Nothing
    return f, Jx'*Fx
  else
    # As you may notice, this can be expensive... Because the gradient
    # is going to be very simple. May want to create a
    # special type or way to hook into trust regions here. We can exploit
    # that we only need the cauchy and the newton steps, not any shifted
    # systems
    return f, Jx'*Fx, Jx'*Jx
  end
end

function solve!(prob::NEqProblem, x, approach::TrustRegion{<:Newton, <:Any, <:Any}, options::NEqOptions)
    F = prob.R
    # should we wrap a Fx here so we can log F0 info here?
    # and so we can extract it at the end as well?
    # xcache = copy(x).-1
    Fx_outer = similar(x)
    x_outer = prevfloat.(x)
    normed_residual = NormedResiduals(x_outer, Fx_outer, F)
    normed_residual(x_outer, Fx_outer, nothing)
    ρF0 = norm(normed_residual.Fx, Inf)
    ρ2F0 = norm(normed_residual.Fx, 2)
    td = TwiceDiffed(normed_residual)
    res = minimize!(td, x, approach, MinOptions())
    # normed_residual to get Fx
    # f0*2 is wrong because it's 2norm
    newinfo = (best_residual=Fx_outer, ρF0=ρF0, ρ2F0=ρ2F0, time=res.info.time, iter=res.info.iter)
    return ConvergenceInfo(approach, newinfo, options)
end
