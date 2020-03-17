using DoubleFloats
@testset "Test double floats" begin
	function f(x, G)
		@show x
@show	    fx = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

	    if !(G == nothing)
	        G1 = -2 * (1 - x[1]) - 400 * (x[2] - x[1]^2) * x[1]
	        G2 = 200 * (x[2] - x[1]^2)
	 @show       gx = Double64.([G1,G2])

	        return fx, gx
	    else
	        return fx
	    end
	end
	f_obj = OnceDiffed(f)
	minimize(f_obj, Double64.([1,2]), LineSearch(GradientDescent(Inverse())), MinOptions(;g_abstol=1e-32))
	minimize(f_obj, Double64.([1,2]), LineSearch(BFGS(Inverse())), MinOptions(;g_abstol=1e-32))
	minimize(f_obj, Double64.([1,2]), LineSearch(DFP(Inverse())), MinOptions(;g_abstol=1e-32))
	minimize(f_obj, Double64.([1,2]), LineSearch(SR1(Inverse())), MinOptions(;g_abstol=1e-32))

	minimize(f_obj, Double64.([1,2]), LineSearch(GradientDescent(Direct())), MinOptions(;g_abstol=1e-32))
	minimize(f_obj, Double64.([1,2]), LineSearch(BFGS(Direct())), MinOptions(;g_abstol=1e-32))
	minimize(f_obj, Double64.([1,2]), LineSearch(DFP(Direct())), MinOptions(;g_abstol=1e-32))
	minimize(f_obj, Double64.([1,2]), LineSearch(SR1(Direct())), MinOptions(;g_abstol=1e-32))

end