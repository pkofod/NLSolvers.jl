function f_2by2!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function g_2by2!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

function nlsolvers_f(x, F, J)
	if F !== nothing
		f_2by2!(F, x)
	end
	if J !== nothing
		g_2by2!(J, x)
	end
	objective_return(F, J)
end

nlsolve!(OnceDiffed(nlsolvers_f), [ -0.5; 1.4], TrustRegion(NLSolvers.Newton(), Dogleg()))
nlsolve!(OnceDiffed(nlsolvers_f), [ -0.5; 1.4], TrustRegion(NLSolvers.Newton()))
nlsolve!(OnceDiffed(nlsolvers_f), [ -0.5; 1.4], TrustRegion(NLSolvers.Newton(), NWI()))
# initial convergence
nlsolve!(OnceDiffed(nlsolvers_f), [ 0.0; 1.0], TrustRegion(NLSolvers.Newton(), Dogleg()))
nlsolve!(OnceDiffed(nlsolvers_f), [ 0.0; 1.0], TrustRegion(NLSolvers.Newton()))
nlsolve!(OnceDiffed(nlsolvers_f), [ 0.0; 1.0], TrustRegion(NLSolvers.Newton(), NWI()))