using NLSolvers, Test, DoubleFloats
function myfun(∇f, x)
    myfun(nothing, ∇f, x)
end
function myfun(∇²f, ∇f, x::T) where T<:Real
    if !(∇²f == nothing)
        ∇²f = T(12*x^2 - sin(x))
    end
    if !(∇f == nothing)
        ∇f = T(4*x^3 + cos(x))
    end
    fx = T(x^4 + sin(x))
    objective_return(T(fx), ∇f, ∇²f)
end
@testset "scalar return types" begin
for T = (Float16, Float32, Float64, Rational{BigInt}, Double32, Double64)
    if T == Rational{BigInt}
        options = MinOptions()
    else
        options = MinOptions(g_abstol=eps(T), g_reltol=T(0))
    end
    for M in (SR1, BFGS, DFP, Newton)
        if M == Newton
            obj = TwiceDiffed(myfun)
            res = minimize(obj, T(3.1), LineSearch(M()), options)
            @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
        else
            obj = OnceDiffed(myfun)
            res = minimize(obj, T(3.1), LineSearch(M(Direct())), options)
            @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
            res = minimize(obj, T(3.1), LineSearch(M(Inverse())), options)
            @test all(isa.([res.info.minimum, res.info.∇fz, res.info.minimizer], T))
        end
    end
end
end
