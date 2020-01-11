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
    for M in (SR1, BFGS, DFP, Newton)
        if M == Newton
            obj = TwiceDiffed(myfun)
            res = minimize(obj, T(4), M())
            @test all(isa.(res[1:3], T))
        else
            obj = OnceDiffed(myfun)
            res = minimize(obj, T(4), M(Direct()))
            @test all(isa.(res[1:3], T))
            res = minimize(obj, T(4), M(Inverse()))
            @test all(isa.(res[1:3], T))
        end
    end
end
end
