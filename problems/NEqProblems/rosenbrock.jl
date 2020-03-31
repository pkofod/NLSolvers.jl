function f!(fvec, x)
    fvec[1] = 1 - x[1]
    fvec[2] = 10(x[2]-x[1]^2)
end
function j!(fjac, x)
    fjac[1,1] = -1
    fjac[1,2] = 0
    fjac[2,1] = -20x[1]
    fjac[2,2] = 10
end

function nlsolvers_f(x, Fx, Jx)
	if Fx !== nothing
		f!(Fx, x)
	end
	if Jx !== nothing
		j!(Jx, x)
	end
	objective_return(Fx, Jx)
end
x0 = [-1.2, 1.0]


nlsolve!(OnceDiffed(nlsolvers_f), copy(x0), TrustRegion(NLSolvers.Newton()))
nlsolve!(OnceDiffed(nlsolvers_f), copy(x0), TrustRegion(NLSolvers.Newton(), Dogleg()))
nlsolve!(OnceDiffed(nlsolvers_f), copy(x0), TrustRegion(NLSolvers.Newton(), NTR()))
nlsolve!(OnceDiffed(nlsolvers_f), copy(x0), TrustRegion(NLSolvers.Newton(), NWI()))
# initial convergence
nlsolve!(OnceDiffed(nlsolvers_f), fill(1.0, 2), TrustRegion(NLSolvers.Newton()))
nlsolve!(OnceDiffed(nlsolvers_f), fill(1.0, 2), TrustRegion(NLSolvers.Newton(), Dogleg()))
nlsolve!(OnceDiffed(nlsolvers_f), fill(1.0, 2), TrustRegion(NLSolvers.Newton(), NTR()))
nlsolve!(OnceDiffed(nlsolvers_f), fill(1.0, 2), TrustRegion(NLSolvers.Newton(), NWI()))