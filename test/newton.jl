using NLSolvers, StaticArrays, Test

function himmelblau(x, ∇f)
    if !(∇f == nothing)
        ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        ∇f = @SVector([∇f1, ∇f2])
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f)
end

function himmelblau!(x, ∇f)
    if !(∇f == nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    return ∇f == nothing ? fx : (fx, ∇f)
end

function himmelblau!(x, ∇f, ∇²f)
    if !(∇²f == nothing)
        ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
        ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 1] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
    end


    if ∇f == nothing && ∇²f == nothing
        fx = himmelblau!(x, ∇f)
        return fx
    elseif ∇²f == nothing
        return himmelblau!(x, ∇f)
    else
        fx, ∇f = himmelblau!(x, ∇f)
        return fx, ∇f, ∇²f
    end
end
himmelblau_nonmut(x, ∇f) = himmelblau!(x, ∇f)
himmelblau_nonmut(x, ∇f, ∇²f) = himmelblau!(x, ∇f, ∇²f)
@testset "Newton" begin
    function himmelblau_twicediff(x, ∇f, ∇²f)
        if !(∇²f == nothing)
            ∇²f11 = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
            ∇²f12 = 4.0 * x[1] + 4.0 * x[2]
            ∇²f21 = 4.0 * x[1] + 4.0 * x[2]
            ∇²f22 = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
            ∇²f = @SMatrix([∇²f11 ∇²f12; ∇²f21 ∇²f22])
        end

        if !(∇f == nothing)
            ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
            ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
            ∇f = @SVector([∇f1, ∇f2])
        end

        fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

        if ∇f == nothing && ∇²f == nothing
            return fx
        elseif ∇²f == nothing
            return fx, ∇f
        else
            return fx, ∇f, ∇²f
        end
    end
    inferredhimmelblau = TwiceDiffed(himmelblau_twicediff; infer=true)

    res = minimize!(TwiceDiffed(himmelblau!), (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton()), MinOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8
    res = minimize(TwiceDiffed(himmelblau_nonmut), (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton()), MinOptions())
    @test norm(res.info.∇fz, Inf) < 1e-8
    @testset "Newton linsolve" begin
        res_qr = minimize!(TwiceDiffed(himmelblau_nonmut), (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, qr(B), g))), MinOptions())
        @test norm(res_qr.info.∇fz, Inf) < 1e-8
        res_qr = minimize(TwiceDiffed(himmelblau_nonmut), (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(B, g)->qr(B)\g)), MinOptions())
        @test norm(res_qr.info.∇fz, Inf) < 1e-8
        res_lu = minimize!(TwiceDiffed(himmelblau_nonmut), (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(d, B, g)->ldiv!(d, lu(B), g))), MinOptions())
        @test norm(res_lu.info.∇fz, Inf) < 1e-8
        res_lu = minimize(TwiceDiffed(himmelblau_nonmut), (copy([2.0,2.0]), [0.0 0.0;0.0 0.0]), LineSearch(Newton(; linsolve=(B, g)->lu(B)\g)), MinOptions())
        @test norm(res_lu.info.∇fz, Inf) < 1e-8
    end
    @testset "newton static" begin
        sl = @SVector([0.0,0.0])
        state0 = (@SVector([2.0,2.0]), I+sl*sl')
        res = minimize(inferredhimmelblau, state0, LineSearch(Newton()), MinOptions())
        _alloc = @allocated minimize(inferredhimmelblau, state0, LineSearch(Newton()), MinOptions())
        @test _alloc == 0
        @test norm(res.info.∇fz, Inf) < 1e-8
        _res = minimize(inferredhimmelblau, state0, LineSearch(Newton(), Backtracking()), MinOptions())
        _alloc = @allocated minimize(inferredhimmelblau, state0, LineSearch(Newton(), Backtracking()), MinOptions())
        @test _alloc == 0
        _alloc = @allocated minimize(inferredhimmelblau, state0, LineSearch(Newton(), Backtracking()), MinOptions())
        @test _alloc == 0
        @test norm(res.info.∇fz, Inf) < 1e-8
    end

end
