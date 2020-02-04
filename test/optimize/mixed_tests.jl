using NLSolvers, StaticArrays, Test
@testset "mixed optimization problems" begin
function theta(x)
   if x[1] > 0
       return atan(x[2] / x[1]) / (2.0 * pi)
   else
       return (pi + atan(x[2] / x[1])) / (2.0 * pi)
   end
end
f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

function f∇f!(∇f, x)
    if !(∇f==nothing)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        end
        ∇f[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
        ∇f[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
        ∇f[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
    end

    fx = f(x)
    objective_return(fx, ∇f)
end

function f∇f(∇f, x)
    if !(∇f == nothing)
        ∇f = similar(x)
    end
    fx, ∇f = f∇f!(gx, x)
    objective_return(fx, ∇f)
end
function f∇fs(∇f, x)
    if !(∇f == nothing)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
        end

        s1 = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
        s2 = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
        s3 = 200.0*(x[3]-10.0*theta(x)) + 2.0*x[3]
        ∇f = @SVector [s1, s2, s3]
    end
    objective_return(f(x), ∇f)
end
obj_inplace = OnceDiffed(f∇f!)
obj_outofplace = OnceDiffed(f∇f)
obj_static = OnceDiffed(f∇fs)

x0 = [-1.0, 0.0, 0.0]
xopt = [1.0, 0.0, 0.0]
x0s = @SVector [-1.0, 0.0, 0.0]

println("Starting  from: ", x0)
println("Targeting     : ", xopt)

shortname(::GradientDescent) = "GD  "
shortname(::Inverse) = "(inverse)"
shortname(::Direct) = " (direct)"

function printed_minimize(f∇f, x0, method)
    res = minimize(f∇f, x0, method, MinOptions())
    print("NN  $(shortname(method)) $(shortname(method.approx)): ")
    @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
end
function printed_minimize!(obj_inplace, x0, method)
    res = minimize!(obj_inplace, x0, method, MinOptions())
    print("NN! $(shortname(method)) $(shortname(method.approx)): ")
    @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
end

printed_minimize(obj_inplace, x0, GradientDescent(Inverse()))
printed_minimize!(obj_inplace, copy(x0), GradientDescent(Inverse()))
res = minimize(obj_static, x0s, GradientDescent(Inverse()))
@printf("NN  GD(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

printed_minimize(obj_inplace, x0, GradientDescent(Direct()))
printed_minimize!(obj_inplace, copy(x0), GradientDescent())
res = minimize(obj_static, x0s, GradientDescent(Direct()))
@printf("NN  GD(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
println()

res = minimize(obj_inplace, x0, BFGS(Inverse()), MinOptions())
@test res[4] == 31
@printf("NN  BFGS    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), (BFGS(Inverse()), Backtracking()), MinOptions())
@test res[4] == 31
@printf("NN! BFGS    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), (BFGS(Inverse()), Backtracking(interp=FFQuadInterp())), MinOptions())
@test res[4] == 33
@printf("NN! BFGS    (inverse, quad): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, x0s, BFGS(Inverse()))
@test res[4] == 31
@printf("NN  BFGS(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, x0s, (BFGS(Inverse()), Backtracking(interp=FFQuadInterp())))
@test res[4] == 33
@printf("NN  BFGS(S) (inverse, quad): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

res = minimize(obj_inplace, x0, BFGS(Direct()), MinOptions())
@printf("NN  BFGS    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), BFGS(Direct()), MinOptions())
@printf("NN! BFGS    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, x0s, BFGS(Direct()))
@printf("NN  BFGS(S) (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
println()
# res = optimize(obj_inplace, copy(x0), Optim.BFGS(linesearch=Backtracking()))
# @time optimize(obj_inplace, copy(x0), Optim.BFGS(linesearch=Backtracking()))
# @printf("OT! BFGS (inverse): %2.2e  %2.2e\n", norm(Optim.minimizer(res)-xopt,Inf), Optim.g_residual(res))

res = minimize(obj_inplace, x0, DFP(Inverse()))
@printf("NN  DFP    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), DFP(Inverse()))
@printf("NN! DFP    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, x0s, DFP(Inverse()))
@printf("NN  DFP(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_inplace, x0, DFP(Direct()))
@printf("NN  DFP    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), DFP(Direct()))
@printf("NN! DFP    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, x0s, DFP(Direct()))
@printf("NN  DFP(S) (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
println()

res = minimize(obj_inplace, x0, SR1(Inverse()))
@printf("NN  SR1     (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), SR1(Inverse()))
@printf("NN! SR1     (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, x0s, SR1(Inverse()))
@printf("NN  SR1(S)  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_inplace, x0, SR1(Direct()))
@printf("NN  SR1     (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), SR1(Direct()))
@printf("NN! SR1     (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, x0s, SR1(Direct()))
@printf("NN  SR1(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(x0), (SR1(Direct()), NWI()))
@printf("NN! SR1   (direct, NW): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

xrand = rand(3)
xrands = SVector{3}(xrand)
println("\nFrom a random point: ", xrand)
res = minimize(obj_inplace, xrand, GradientDescent(Inverse()))
@printf("NN  GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), GradientDescent(Inverse()))
@printf("NN! GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_inplace, xrand, GradientDescent(Direct()))
@printf("NN  GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), GradientDescent(Direct()))
@printf("NN! GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
println()
res = minimize(obj_inplace, xrand, BFGS(Inverse()))
@printf("NN  BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), (BFGS(Inverse()), Backtracking()))
@printf("NN! BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), (BFGS(Inverse()), HZAW()))
@printf("NN! BFGS (inverse) HZ: %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_inplace, xrand, BFGS(Direct()))
@printf("NN  BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), BFGS(Direct()))
@printf("NN! BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_inplace, xrand, DBFGS(Direct()))
@printf("NN  DBFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), DBFGS(Direct()))
@printf("NN! DBFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), (DBFGS(Direct()), NWI()))
@printf("NN! DBFGS  (direct, NW): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), (DBFGS(Direct()), Dogleg()))
@printf("NN! DBFGS  (direct, DL): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
println()

res = minimize(obj_inplace, xrand, DFP(Inverse()))
@printf("NN  DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), DFP(Inverse()))
@printf("NN! DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_inplace, xrand, DFP(Direct()))
@printf("NN  DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), DFP(Direct()))
@printf("NN! DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
println()

res = minimize(obj_inplace, xrand, SR1(Inverse()))
@printf("NN  SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), SR1(Inverse()))
@printf("NN! SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, xrand, SR1(Inverse()))
@printf("NN  SR1(S)   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_inplace, xrand, SR1(Direct()))
@printf("NN  SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), SR1(Direct()))
@printf("NN! SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize(obj_static, xrand, SR1(Direct()))
@printf("NN  SR1(S)   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

# res = minimize(obj_inplace, xrand, (SR1(Direct()), NWI()))
# @printf("NN  SR1   (direct, tr): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
res = minimize!(obj_inplace, copy(xrand), (SR1(Direct()), NWI()))
@printf("NN! SR1   (direct, NW): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
# res = minimize(obj_static, xrand, (SR1(Direct()), NWI()))
# @printf("NN  SR1(S)   (direct, tr): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

println()


function himmelblau!(∇f, x)
    if !(∇f == nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f)
end


function himmelblaus(∇f, x)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    if !(∇f == nothing)
        ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        ∇f = @SVector([∇f1, ∇f2])
    end
    objective_return(fx, ∇f)
end

function himmelblau(∇f, x)
    g = ∇f == nothing ? ∇f : similar(x)

    return himmelblau!(g, x)
end

him_inplace = OnceDiffed(himmelblau!)
him_static = OnceDiffed(himmelblaus)
him_outofplace = OnceDiffed(himmelblau)

println("\nHimmelblau function")
x0 = [3.0, 1.0]
x0s = SVector{2}(x0)
res = minimize(him_outofplace, x0, GradientDescent(Inverse()))
@printf("NN  GD    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), GradientDescent(Inverse()))
@printf("NN! GD    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_static, x0s, GradientDescent(Inverse()))
@printf("NN  GD(S) (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
println()
res = minimize(him_outofplace, x0, GradientDescent(Direct()))
@printf("NN  GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), GradientDescent(Direct()))
@printf("NN! GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_static, x0s, GradientDescent(Direct()))
@printf("NN  GD(S) (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
println()

res = minimize(him_outofplace, x0, BFGS(Inverse()))
@printf("NN  BFGS    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), BFGS(Inverse()))
@printf("NN! BFGS    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0s, BFGS(Inverse()))
@printf("NN  BFGS(S) (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
println()
res = minimize(him_outofplace, x0, BFGS(Direct()))
@printf("NN  BFGS    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), BFGS(Direct()))
@printf("NN! BFGS    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0s, BFGS(Direct()))
@printf("NN  BFGS(S) (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), (DBFGS(Direct()), NWI()))
@printf("NN! DBFGS  (direct, NW): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), (DBFGS(Direct()), Dogleg()))
@printf("NN! DBFGS  (direct, DL): %2.2e  %d\n", norm(res[3], Inf), res[4])

println()

res = minimize(him_outofplace, x0, DFP(Inverse()))
@printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), DFP(Inverse()))
@printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0s, DFP(Inverse()))
@printf("NN  DFP(S)  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0, DFP(Direct()))
println()
@printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), DFP(Direct()))
@printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0s, DFP(Direct()))
@printf("NN  DFP(S)   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
println()

res = minimize(him_outofplace, x0, SR1(Inverse()))
@printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), SR1(Inverse()))
@printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0s, SR1(Inverse()))
@printf("NN  SR1(S)  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0, SR1(Direct()))
@printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(x0), SR1(Direct()))
@printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, x0s, SR1(Direct()))
@printf("NN  SR1(S)   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
println()
xrand = rand(2)
xrands = SVector{2}(xrand)
println("\nFrom a random point: ", xrand)

res = minimize(him_outofplace, xrand, GradientDescent(Inverse()))
@printf("NN  GD   (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(xrand), GradientDescent(Inverse()))
@printf("NN! GD   (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, xrand, GradientDescent(Direct()))
@printf("NN  GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(xrand), GradientDescent(Direct()))
@printf("NN! GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

res = minimize(him_outofplace, xrand, BFGS(Inverse()))
@printf("NN   BFGS (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(xrand), BFGS(Inverse()))
@printf("NN!  BFGS (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_static, xrand, BFGS(Direct()))
@printf("NN(S) BFGS (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, xrand, BFGS(Direct()))
@printf("NN  BFGS  (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace, copy(xrand), BFGS(Direct()))
@printf("NN! BFGS  (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

res = minimize(him_outofplace, xrand, DFP(Inverse()))
@printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace,  copy(xrand), DFP(Inverse()))
@printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, xrand, DFP(Direct()))
@printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace,  copy(xrand), DFP(Direct()))
@printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

res = minimize(him_outofplace, xrand, SR1(Inverse()))
@printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace,  copy(xrand), SR1(Inverse()))
@printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize(him_outofplace, xrand, SR1(Direct()))
@printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
res = minimize!(him_inplace,  copy(xrand), SR1(Direct()))
@printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

end
