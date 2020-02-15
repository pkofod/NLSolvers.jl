function exponential!(H, g, x)
    fx = exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)

    if is_something(g)
        g[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
        g[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    end
    if is_somthing(H)
        H[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 2.0 * exp((3.0 - x[2])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
    end
    objective_return(fx, g, H)
end
exponential_prb = Dict()
exponential_prb["twicediffed!"] = TwiceDiffed(exponential!)
exponential_prb["oncediffed!"] = OnceDiffed(exponential!)
exponential_prb["initial_x"] = [0.0, 0.0]
exponential_prb["minimizer"] = [2.0, 3.0]
exponential_prb["minimum"] = exponential!(nothing, nothing, [2.0, 3.0])
problems["unconstrained"]["exponential"] = exponential_prb

function fletcher_powell!(∇f, x)
    function theta(x)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    if ( x[1]^2 + x[2]^2 == zero(eltype(x)) )
        dtdx1 = zero(eltype(x))
        dtdx2 = zero(eltype(x))
    else
        dtdx1 = - x[2] / ( 2.0 * pi * ( x[1]^2 + x[2]^2 ) )
        dtdx2 =   x[1] / ( 2.0 * pi * ( x[1]^2 + x[2]^2 ) )
    end

    fx = 100.0 * ((x[3] - 10.0 * theta(x))^2 +
        (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

    if is_something(∇f)
        ∇f[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1.0)*x[1]/sqrt( x[1]^2+x[2]^2 )
        ∇f[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1.0)*x[2]/sqrt( x[1]^2+x[2]^2 )
        ∇f[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3]
        return fx, ∇f
    end
    return fx
end

fletcher_powell_prb = Dict()
fletcher_powell_prb["twicediffed!"] = TwiceDiffed(fletcher_powell!)
fletcher_powell_prb["oncediffed!"] = OnceDiffed(fletcher_powell!)
fletcher_powell_prb["initial_x"] = [-1.0,0.0,0.0]
fletcher_powell_prb["minimizer"] = [1.0,0.0,0.0]
fletcher_powell_prb["minimum"] = fletcher_powell!(nothing, nothing, [1.0,0.0,0.0])
problems["unconstrained"]["fletcher_powell"] = fletcher_powell_prb


function himmelblau!(x, g, H)
  fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

  if !isnothing(g)
    g[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
    44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    g[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
    4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
  end
  if !isnothing(H)
    H[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
    H[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    H[2, 1] = 4.0 * x[1] + 4.0 * x[2]
    H[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
  end
  objective_return(fx, g, H)
end

himmelblau_prb = Dict()
himmelblau_prb["twicediffed!"] = TwiceDiffed(himmelblau!)
himmelblau_prb["oncediffed!"] = OnceDiffed(himmelblau!)
himmelblau_prb["initial_x"] = [2.0, 2.0]
himmelblau_prb["minimizer"] = [3.0, 2.0]
himmelblau_prb["minimum"] = himmelblau_!(nothing, nothing, [3.0, 2.0])
problems["unconstrained"]["himmelblau"] = himmelblau_prb


function hosaki!(H, ∇f, x)
    a = (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4)
    fx = a * x[2]^2 * exp(-x[2])

    if !(∇f isa Nothing)
    ∇f[1] = (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8)* x[2]^2 * exp(-x[2])
    ∇f[2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) - (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
    end
    if !(H isa Nothing)
    H[1, 1] = (3.0 * x[1]^2 - 14.0 * x[1] + 14.0) * x[2]^2 * exp(-x[2])
    H[1, 2] = 2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    H[2, 1] =  2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    H[2, 2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * exp(-x[2]) - 4.0 * ( 1.0 - 8.0 * x[1] + 7.0 *  x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) + (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
    end
    objective_return(fx, ∇f, H)
end
td_obj = TwiceDiffed(hosaki!)

hosaki_prb = Dict()
hosaki_prb["twicediffed!"] = TwiceDiffed(hosaki!)
hosaki_prb["oncediffed!"] = OnceDiffed(hosaki!)
hosaki_prb["initial_x"] = [3.6, 1.9]
hosaki_prb["minimizer"] = [4.0, 2.0]
hosaki_prb["minimum"] = hosaki!(nothing, nothing, [4.0, 2.0])
problems["unconstrained"]["hosaki"] = hosaki_prb



function large_polynomial!(H, ∇f, x)
    fx = zero(x[1])
    for i in 1:250
        fx += (i - x[i])^2
    end
    if !(∇f isa Nothing)
        for i in 1:250
            ∇f[i] = -2.0 * (i - x[i])
        end
    end
    if !(H isa Nothing)
        for i in 1:250
            for j in i:250
                if i == j
                    H[i, j] = 2.0
                else
                    H[i, j] = 0.0
                    H[j, i] = 0.0
                end
            end
        end
    end
    objective_return(fx, ∇f, H)
end

td_obj = TwiceDiffed(large_polynomial!)
large_polynomial = Dict()
large_polynomial["twicediffed!"] = TwiceDiffed(large_polynomial!)
large_polynomial["oncediffed!"] = OnceDiffed(large_polynomial!)
large_polynomial["initial_x"] = zeros(250)
large_polynomial["minimizer"] = collect(float(1:250))
large_polynomial["minimum"] = large_polynomial!(nothing, nothing, collect(float(1:250)))
problems["unconstrained"]["large_polynomial"] = large_polynomial

function parabola!(H, ∇f, x)
    fx =  (1.0 - x[1])^2 + (2.0 - x[2])^2 + (3.0 - x[3])^2 +
        (5.0 - x[4])^2 + (8.0 - x[5])^2

    if !(∇f isa Nothing)
        ∇f[1] = -2.0 * (1.0 - x[1])
        ∇f[2] = -2.0 * (2.0 - x[2])
        ∇f[3] = -2.0 * (3.0 - x[3])
        ∇f[4] = -2.0 * (5.0 - x[4])
        ∇f[5] = -2.0 * (8.0 - x[5])
    end
    if !(H isa Nothing)
        for i in 1:5
            for j in 1:5
                if i == j
                    H[i, j] = 2.0
                else
                    H[i, j] = 0.0
                end
            end
        end
    end
    objective_return(fx, ∇f, H)
end

parabola = Dict()
parabola["twicediffed!"] = TwiceDiffed(parabola!)
parabola["oncediffed!"] = OnceDiffed(parabola!)
parabola["initial_x"] = zeros(5)
parabola["minimizer"] = [1.0, 2.0, 3.0, 5.0, 8.0]
parabola["minimum"] = parabola!(nothing, nothing, [1.0, 2.0, 3.0, 5.0, 8.0])
problems["unconstrained"]["parabola"] = parabola


function polynomial!(H, ∇f, x)
    fx  = (10.0 - x[1])^2 + (7.0 - x[2])^4 + (108.0 - x[3])^4
    if !(∇f isa Nothing)
        ∇f[1] = -2.0 * (10.0 - x[1])
        ∇f[2] = -4.0 * (7.0 - x[2])^3
        ∇f[3] = -4.0 * (108.0 - x[3])^3
    end
    if !(H isa Nothing)
        H[1, 1] = 2.0
        H[1, 2] = 0.0
        H[1, 3] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 12.0 * (7.0 - x[2])^2
        H[2, 3] = 0.0
        H[3, 1] = 0.0
        H[3, 2] = 0.0
        H[3, 3] = 12.0 * (108.0 - x[3])^2
    end
    objective_return(fx, ∇f, H)
end

polynomial = Dict()
polynomial["twicediffed!"] = TwiceDiffed(polynomial!)
polynomial["oncediffed!"] = OnceDiffed(polynomial!)
polynomial["initial_x"] = zeros(3)
polynomial["minimizer"] = [10.0, 7.0, 108.0]
polynomial["minimum"] = polynomial!(nothing, nothing, [10.0, 7.0, 108.0])
problems["unconstrained"]["polynomial"] = polynomial

function powell!(H, ∇f, x)
    fx = (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
    (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4
    if !(∇f isa Nothing)
        ∇f[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
        ∇f[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
        ∇f[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
        ∇f[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
    end
    if !(H isa Nothing)
        H[1, 1] = 2.0 + 120.0 * (x[1] - x[4])^2
        H[1, 2] = 20.0
        H[1, 3] = 0.0
        H[1, 4] = -120.0 * (x[1] - x[4])^2
        H[2, 1] = 20.0
        H[2, 2] = 200.0 + 12.0 * (x[2] - 2.0 * x[3])^2
        H[2, 3] = -24.0 * (x[2] - 2.0 * x[3])^2
        H[2, 4] = 0.0
        H[3, 1] = 0.0
        H[3, 2] = -24.0 * (x[2] - 2.0 * x[3])^2
        H[3, 3] = 10.0 + 48.0 * (x[2] - 2.0 * x[3])^2
        H[3, 4] = -10.0
        H[4, 1] = -120.0 * (x[1] - x[4])^2
        H[4, 2] = 0.0
        H[4, 3] = -10.0
        H[4, 4] = 10.0 + 120.0 * (x[1] - x[4])^2
    end
    objective_return(fx, ∇f, H)
end

powell = Dict()
powell["twicediffed!"] = TwiceDiffed(powell!)
powell["oncediffed!"] = OnceDiffed(powell!)
powell["initial_x"] = [-3.0, -1.0, 0.0, 1.0]
powell["minimizer"] = zeros(4)
powell["minimum"] = powell!(nothing, nothing, zeros(4))
problems["unconstrained"]["powell"] = powell


function rosenbrock!(H, ∇f, x::Vector)
    fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    if is_something(∇f)
    ∇f[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    ∇f[2] = 200.0 * (x[2] - x[1]^2)
    end
    if is_something(H)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
    end
    objective_return(fx, ∇f, H)
end

rosenbrock = Dict()
rosenbrock["twicediffed!"] = TwiceDiffed(rosenbrock!)
rosenbrock["oncediffed!"] = OnceDiffed(rosenbrock!)
rosenbrock["initial_x"] = [-1.2, 1.0]
rosenbrock["minimizer"] = ones(2)
rosenbrock["minimum"] = rosenbrock!(nothing, nothing, ones(4))
problems["unconstrained"]["rosenbrock"] = rosenbrock

function run_all(obj, x0, approx, globals)
    res = []
    _res = minimize!(obj, copy(x0), approx)
    push!(res, _res)
    for glob in globals
      _res = minimize!(obj, copy(x0), (approx, glob))
      printstyled(_res; color=maximum(abs, _res[3]) ≤ sqrt(eps(Float64)) ? :green : :red);println()
      push!(res, _res)
    end
    res
end

direct_tr = (NWI(), Dogleg())
direct_globals = (HZAW(), Backtracking(), Backtracking(;interp=FFQuadInterp()), NWI(), Dogleg())
inverse_ls = (HZAW(), Backtracking(), Backtracking(; interp=FFQuadInterp()))
inverse_globals = (HZAW(), Backtracking(), Backtracking(;interp=FFQuadInterp()))
run_all(od_obj, x0, BFGS(Inverse()), inverse_globals)
run_all(od_obj, x0, DBFGS(Inverse()), inverse_globals)
#run_all(od_obj, x0, SR1(Inverse()), inverse_ls)
run_all(od_obj, x0, DFP(Inverse()), inverse_globals)
run_all(od_obj, x0, BFGS(Direct()), direct_globals)
run_all(od_obj, x0, DBFGS(Direct()), direct_globals)
run_all(od_obj, x0, SR1(Direct()), direct_globals)
run_all(od_obj, x0, DFP(Direct()), direct_globals)

run_all(td_obj, x0, Newton(), direct_globals)
run_all(td_obj, x0, Newton(), direct_globals)




#==
  Flether-Powell
==#



run_all(od_obj, x0, BFGS(Inverse()), inverse_globals)
run_all(od_obj, x0, DBFGS(Inverse()), inverse_globals)
run_all(od_obj, x0, SR1(Inverse()), inverse_ls)
run_all(od_obj, x0, DFP(Inverse()), inverse_globals)
run_all(od_obj, x0, BFGS(Direct()), direct_globals)
run_all(od_obj, x0, DBFGS(Direct()), direct_globals)
run_all(od_obj, x0, SR1(Direct()), direct_globals)
run_all(od_obj, x0, DFP(Direct()), direct_globals)


function himmelblau!(H, g, x::Vector)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

    if !(g isa Nothing)
    g[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
    44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    g[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
    4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end
    if !(H isa Nothing)
    H[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
    H[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    H[2, 1] = 4.0 * x[1] + 4.0 * x[2]
    H[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
    end
    objective_return(fx, g, H)
end
x0 = [2.0, 2.0]
xopt = [3.0, 2.0]
od_obj = OnceDiffed(himmelblau!)
td_obj = TwiceDiffed(himmelblau!)

run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);


function hosaki!(H, ∇f, x)
    a = (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4)
    fx = a * x[2]^2 * exp(-x[2])

    if !(∇f isa Nothing)
    ∇f[1] = (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8)* x[2]^2 * exp(-x[2])
    ∇f[2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) - (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
    end
    if !(H isa Nothing)
    H[1, 1] = (3.0 * x[1]^2 - 14.0 * x[1] + 14.0) * x[2]^2 * exp(-x[2])
    H[1, 2] = 2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    H[2, 1] =  2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    H[2, 2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * exp(-x[2]) - 4.0 * ( 1.0 - 8.0 * x[1] + 7.0 *  x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) + (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
    end
    objective_return(fx, ∇f, H)
end
td_obj = TwiceDiffed(hosaki!)

x0 = [3.6, 1.9]
xopt = [4.0, 2.0]
run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);

examples["Hosaki"] = OptimizationProblem("Hosaki",
                                         hosaki,
                                         hosaki_gradient!,
                                         nothing,
                                         hosaki_hessian!,
                                         nothing, # Constraints
                                         hosaki([4.0, 2.0]),
                                         true,
                                         true)
