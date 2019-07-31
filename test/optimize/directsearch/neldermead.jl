using NLSolvers
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
   return ∇f==nothing ? fx : (fx, ∇f)
end

f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2
function theta(x)
  if x[1] > 0
      return atan(x[2] / x[1]) / (2.0 * pi)
  else
      return (pi + atan(x[2] / x[1])) / (2.0 * pi)
  end
end


obj! = OnceDiff(f∇f!)

minimize!(obj!, -rand(3)*9 .- 3, NLSolvers.NelderMead())
V = [[1.0,1.0,1.0], [0.0,1.0,1.0],[0.40,0.0,0.0],[-1.0,2.0,.03]]
F = f∇f!.(nothing, V)
splx = NLSolvers.ValuedSimplex(V, F)
minimize!(obj!, splx, NLSolvers.NelderMead())
@profiler minimize!(obj!, splx, NLSolvers.NelderMead(); itermax=1500)




function powell(∇f, x)
    fx = (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
        (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4

    if !isa(∇f, Nothing)
        ∇f[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
        ∇f[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
        ∇f[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
        ∇f[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
        return ∇f, fx
    end
    return fx
end


function powell_optim(F, ∇f, x)
    if !isa(F, Nothing)
        fx = (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
            (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4
    end
    if !isa(∇f, Nothing)
        ∇f[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
        ∇f[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
        ∇f[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
        ∇f[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
        if !isa(F, Nothing)
            return fx
        else
            return Nothing
        end
    end
    return fx
end
obj_powell = OnceDiff(powell)
# Define vertices in V
x0 = [1.0,1.0,1.0,1.0]
V = [copy(x0)]
for i = 1:4
    push!(V, x0+39*rand(4))
end
V
F = obj_powell.(V)
splx = NLSolvers.ValuedSimplex(V, F)
@time minimize!(obj_powell, splx, NLSolvers.NelderMead(); itermax=3000)
@time minimize!(obj_powell, copy(x0), NLSolvers.NelderMead(); itermax=3000)
@allocated minimize!(obj_powell, splx, NLSolvers.NelderMead(); itermax=3000)
@profiler minimize!(obj_powell, copy(x0), NLSolvers.NelderMead(); itermax=3000)

using Optim
res = optimize(Optim.only_fg!(powell_optim), copy(x0), NelderMead())
res = optimize(Optim.only_fg!(powell_optim), copy(x0), Optim.GradientDescent())
@time optimize(Optim.only_fg!(powell_optim), x0, NelderMead(), Optim.Options(g_tol=0, iterations =3000))


function extros!(storage, x::AbstractArray)
   n = length(x)
   jodd = 1:2:n-1
   jeven = 2:2:n
   xt = similar(x)
   @. xt[jodd] = 10.0 * (x[jeven] - x[jodd]^2)
   @. xt[jeven] = 1.0 - x[jodd]

   if !isa(storage, Nothing)
       @. storage[jodd] = -20.0 * x[jodd] * xt[jodd] - xt[jeven]
       @. storage[jeven] = 10.0 * xt[jodd]
       return 0.5*sum(abs2, xt), storage
   end
   return 0.5*sum(abs2, xt)
end
x0=rand(300)
V = [copy(x0)]
for i = 1:300
    push!(V, x0+39*rand(length(x0)))
end
V
extros_obj! = OnceDiff(extros!)
F = extros_obj!.(V)
splx = NLSolvers.ValuedSimplex(V, F)
@time minimize!(extros_obj!, splx, NLSolvers.NelderMead(); itermax=3000)
