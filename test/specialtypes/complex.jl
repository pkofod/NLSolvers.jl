using Revise
using NLSolvers
using Random, LinearAlgebra
Random.seed!(0) # Set the seed for reproducibility
# μ is the strength of the quartic. μ = 0 is just a quadratic problem
n = 4
A = randn(n,n) + im*randn(n,n)
A = A'A + I
b = randn(n) + im*randn(n)
μ = 0.5

fcomplex(x) = real(dot(x,A*x)/2 - dot(b,x)) + μ*sum(abs.(x).^4)
gcomplex(x) = A*x-b + 4μ*(abs.(x).^2).*x
gcomplex!(stor,x) = copyto!(stor,gcomplex(x))

function obj_complex(g, x)
  if !(g isa Nothing)
  	gcomplex!(g, x)
  end
  fx = fcomplex(x)
  objective_return(fx, g)
end

x0 = randn(n)+im*randn(n)

minimize!(OnceDiffed(obj_complex), copy(x0), LineSearch(LBFGS()), MinOptions())
minimize!(OnceDiffed(obj_complex), copy(x0), LineSearch(SR1()), MinOptions())
