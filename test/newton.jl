using NLSolvers, StaticArrays, Test

function himmelblau(∇f, x)
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

function himmelblau!(∇f, x)
    if !(∇f == nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    return ∇f == nothing ? fx : (fx, ∇f)
end

function himmelblau!(∇²f, ∇f, x)
    if !(∇²f == nothing)
        ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
        ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 1] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
    end


    if ∇f == nothing && ∇²f == nothing
        fx = himmelblau!(∇f, x)
        return fx
    elseif ∇²f == nothing
        return himmelblau!(∇f, x)
    else
        fx, ∇f = himmelblau!(∇f, x)
        return fx, ∇f, ∇²f
    end
end
himmelblau_nonmut(∇f, x) = himmelblau!(∇f, x)
himmelblau_nonmut(∇²f, ∇f, x) = himmelblau!(∇²f, ∇f, x)
@testset "Newton" begin
    function himmelblau_twicediff(∇²f, ∇f, x)
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
