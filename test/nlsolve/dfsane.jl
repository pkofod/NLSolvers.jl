using NLSolvers
function dfsane_exponential(x, Fx)
  Fx[1] = exp(x[1]-1)-1
  for i = 2:length(Fx)
  	Fx[i] = i*(exp(x[i]-1)-x[i])
  end
  Fx
end
n = 1000
x0 = fill(n/(n-1), n)

nlsolve!(NEqProblem(NonDiffed(dfsane_exponential)), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())

n = 10000
x0 = fill(n/(n-1), n)
nlsolve!(NEqProblem(NonDiffed(dfsane_exponential)), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())


function dfsane_exponential2(x, Fx)
  Fx[1] = exp(x[1])-1
  for i = 2:length(Fx)
  	Fx[i] = i/10*(exp(x[i])+x[i-1]-1)
  end
  Fx
end
n = 2000
x0 = fill(1/(n^2), n)
nlsolve!(NEqProblem(NonDiffed(dfsane_exponential2)), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())
n = 500
x0 = fill(1/(n^2), n)
nlsolve!(NEqProblem(NonDiffed(dfsane_exponential2)), x0, NLSolvers.DFSANE(), NLSolvers.NEqOptions())
