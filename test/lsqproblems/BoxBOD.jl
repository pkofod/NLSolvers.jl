boxbod_f(x, b) = b[1]*(1-exp(-b[2]*x))

ydata = [109, 149, 149, 191, 213, 224]
xdata = [1, 2, 3, 5, 7, 10]

start1 = [1.0, 1.0]
start2 = [100.0, 0.75]

using Plots
using NLSolvers

nd = NonDiffed(t->sum(abs2, boxbod_f.(xdata, Ref(t)).-ydata))
bounds = (fill(0.0, 2), fill(500.0, 2))
problem = MinProblem(; obj=nd, bounds=bounds)
@show minimize!(problem, zeros(2), APSO(), MinOptions())
@show minimize!(nd, [200.0, 1], NelderMead(), MinOptions())

@show minimize!(nd, minimize!(nd, [200.0, 10.0], NelderMead(), MinOptions()).info.minimizer, NelderMead(), MinOptions())
@show minimize!(nd, minimize!(nd, [200.0, 5.0], NelderMead(), MinOptions()).info.minimizer, NelderMead(), MinOptions())

function F(b, F, J=nothing)
  @. F = b[1]*(1 - exp(-b[2]*xdata)) - ydata
  if !isa(J, Nothing)
  	@. J[:, 1] = 1 - exp(-b[2]*xdata)
  	@. @views J[:, 2] = xdata*b[1]*(J[:, 1]-1)
    return F, J
  end
  F
end

Fc = zeros(6)
minimize!(lsqwrap, [100.0, 1.0], NelderMead(), MinOptions())

lsqwrap = NLSolvers.LsqWrapper(F, zeros(6), zeros(6,2))
minimize!(MinProblem(;obj=lsqwrap,bounds=([0.0,0.0],[250.0,2.0])), [100.0, 1.0], APSO(), MinOptions())

function F(F, b)
  @. F = b[1]*(1 - exp(-b[2]*xdata)) - ydata

  F
end
function F(J, F, b)
  @. F = b[1]*(1 - exp(-b[2]*xdata)) - ydata
  if !isa(J, Nothing)
  	@. J[:, 1] = 1 - exp(-b[2]*xdata)
  	@. @views J[:, 2] = xdata*b[1]*(J[:, 1]-1)
    return F, J
  end
  F
end
OnceDiffed(F)(rand(2), rand(6), rand(6,2)

Fc = zeros(6)


lsqwrap = NLSolvers.LsqWrapper(OnceDiffed(F), zeros(6), zeros(6,2))
minimize!(lsqwrap, [100.0, 1.0], LineSearch(LBFGS()), MinOptions())


#using Plots
#theme(:ggplot2)
#gr(size=(500,500))
#X = range(000.0, 350.0; length=420)
#Y = range(0.0, 3.00; length=420)
#contour(X, Y, (x, y)->nd([x, y]);
#       fill=true,
#       c=:turbid,levels=200, ls=:dash,
#       xlims=(minimum(X), maximum(X)),
#       ylims=(minimum(Y), maximum(Y)),
#       colorbar=true)
