@testset "no-alloc static" begin
    function theta(x)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    function fletcher_powell_fg_static(∇f, x)
        T = eltype(x)
        theta_x = theta(x)

        if !(∇f==nothing)
            if x[1]^2 + x[2]^2 == 0
                dtdx1 = T(0)
                dtdx2 = T(0)
            else
                dtdx1 = - x[2] / ( T(2) * pi * ( x[1]^2 + x[2]^2 ) )
                dtdx2 =   x[1] / ( T(2) * pi * ( x[1]^2 + x[2]^2 ) )
            end
            ∇f1 = -2000.0*(x[3]-10.0*theta_x)*dtdx1 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
            ∇f2 = -2000.0*(x[3]-10.0*theta_x)*dtdx2 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
            ∇f3 =  200.0*(x[3]-10.0*theta_x) + 2.0*x[3];
            ∇f = @SVector[∇f1, ∇f2, ∇f3]
        end

        fx = 100.0 * ((x[3] - 10.0 * theta_x)^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

        if ∇f == nothing
            return fx
        else
            return fx, ∇f
        end
    end
    sv3 = @SVector[0.0,0.0,0.0]

    fg_static = OnceDiffed(fletcher_powell_fg_static; infer=true)

    state0 = (@SVector[-0.5, 0.0, 0.0], I+sv3*sv3')

    @allocated minimize(fg_static, state0, LineSearch(BFGS(Inverse()), Backtracking()), MinOptions())
    _alloc = @allocated minimize(fg_static, state0, LineSearch(BFGS(Inverse()), Backtracking()), MinOptions())
    @test _alloc == 0

    minimize(fg_static, state0, LineSearch(BFGS(Direct()), Backtracking()), MinOptions())
    _alloc = @allocated minimize(fg_static, state0, LineSearch(BFGS(Direct()), Backtracking()), MinOptions())
    @test _alloc == 0

    minimize(fg_static, state0, LineSearch(DFP(Inverse()), Backtracking()), MinOptions())
    _alloc = @allocated minimize(fg_static, state0, LineSearch(DFP(Inverse()), Backtracking()), MinOptions())
    @test _alloc == 0

    minimize(fg_static, state0, LineSearch(DFP(Direct()), Backtracking()), MinOptions())
    _alloc = @allocated minimize(fg_static, state0, LineSearch(DFP(Direct()), Backtracking()), MinOptions())
    @test _alloc == 0

    minimize(fg_static, state0, LineSearch(SR1(Inverse()), Backtracking()), MinOptions())
    _alloc = @allocated minimize(fg_static, state0, LineSearch(SR1(Inverse()), Backtracking()), MinOptions())
    @test _alloc == 0

    minimize(fg_static, state0, LineSearch(SR1(Direct()), Backtracking()), MinOptions())
    _alloc = @allocated minimize(fg_static, state0, LineSearch(SR1(Direct()), Backtracking()), MinOptions())
    @test _alloc == 0

end
