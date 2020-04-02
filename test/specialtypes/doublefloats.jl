using DoubleFloats
@testset "Test double floats" begin
	function f(x, G)
	    fx = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

	    if !(G == nothing)
	        G1 = -2 * (1 - x[1]) - 400 * (x[2] - x[1]^2) * x[1]
	        G2 = 200 * (x[2] - x[1]^2)
	        G = [G1,G2]
	    end
	    objective_return(fx, G)
	end
	f_obj = OnceDiffed(f)
	@show minimize(f_obj, Double64.([1,2]), LineSearch(GradientDescent(Inverse())), MinOptions(;g_abstol=1e-32, maxiter=100000))
	@show minimize(f_obj, Double64.([1,2]), LineSearch(BFGS(Inverse())), MinOptions(;g_abstol=1e-32))
	@show minimize(f_obj, Double64.([1,2]), LineSearch(DFP(Inverse())), MinOptions(;g_abstol=1e-32))
	@show minimize(f_obj, Double64.([1,2]), LineSearch(SR1(Inverse())), MinOptions(;g_abstol=1e-32))
end