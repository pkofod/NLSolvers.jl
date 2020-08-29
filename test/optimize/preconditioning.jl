using NLSolvers, SparseArrays, LinearAlgebra
using Test
debug_printing = true

plap(U; n=length(U)) = (n-1) * sum((0.1 .+ diff(U).^2).^2) - sum(U) / (n-1)
plap1(U; n=length(U), dU = diff(U), dW = 4 .* (0.1 .+ dU.^2) .* dU) =
                        (n - 1) .* ([0.0; dW] .- [dW; 0.0]) .- ones(n) / (n-1)
precond(x::Vector) = precond(length(x))
precond(n::Number) = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1)) * (n+1)
function f(x, ∇f)
    fx = plap([0;x;0])
    if !(∇f isa Nothing)
        copyto!(∇f, (plap1([0;x;0]))[2:end-1])
    end
    objective_return(fx, ∇f)
end
GRTOL = 1e-6

debug_printing && println("Test a basic preconditioning example")
for N in (10, 50, 250)  
    debug_printing && println("N = ", N)
    initial_x = zeros(N)
    Plap = precond(initial_x)
    ID = nothing
    for (P, wwo) in zip((nothing, (x, P=nothing)->inv(Array(precond(N)))), (" WITHOUT", " WITH"))
        for optimizer in (LineSearch(GradientDescent(Direct(), P)), LineSearch(ConjugateGradient(HZ(), P), HZAW()), LineSearch(LBFGS(Inverse(), NLSolvers.TwoLoop(), 5, P)))
            println("Iter should be lower for WITH")
            @show wwo
            @show summary(optimizer)
            results = minimize!(OnceDiffed(f), copy(initial_x), optimizer, MinOptions())
            @show results.info.minimum
            @show results.info.iter
        end
    end
end