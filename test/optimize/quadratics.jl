using LinearAlgebra
A = rand(2,2)
A = abs.(A)
A = Symmetric(A*A')
x = rand(2)
b = A*x


f(x) = -dot(b, x) + dot(x, A*x)/2
function obj(x, G, H=nothing)
  if !(G isa Nothing)
          G.=A*x-b
  end
  if !(H isa Nothing)
          H .= A
  end
  objective_return(f(x), G, H)
end


using NLSolvers
for approx in (BFGS(), DBFGS(), SR1(), DFP(), Newton(), BB()) # BB, CBB GradientDescent
	lsres =  minimize!(OnceDiffed(obj), zeros(2), LineSearch(approx, Backtracking()), MinOptions(maxiter=20000))
	println(summary(approx), "\n          ||   $(lsres.info.iter)   ||   $(lsres.info.∇fz)")
end
