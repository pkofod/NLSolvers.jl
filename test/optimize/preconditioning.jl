using NLSolvers, SparseArrays, LinearAlgebra
using Test

for N in (10, 50, 250)  
    println("N = ", N)
    initial_x = OPT_PROBS["laplacian"]["array"]["x0(n)"](N)
    Plap = precond(initial_x)
    ID = nothing
    iter = []
    mino = []
    for optimizer in (p->LineSearch(GradientDescent(Direct(), p)), p->LineSearch(ConjugateGradient(HZ(), p), HZAW()), p->LineSearch(LBFGS(Inverse(), NLSolvers.TwoLoop(), 5, p)), p->LineSearch(LBFGS(Inverse(), NLSolvers.TwoLoop(), 5, p),Backtracking()))
        for (P, wwo) in zip((nothing, (x, P=nothing)->inv(Array(precond(N)))), (" WITHOUT", " WITH"))
            results = solve!(OPT_PROBS["laplacian"]["array"]["mutating"], copy(initial_x), optimizer(P), MinOptions(g_abstol=1e-6))
            push!(mino, results.info.minimum)
            push!(iter, results.info.iter)
        end
        println("Iterations without precon: $(iter[end-1])")
        println("Iterations with    precon: $(iter[end])")
        println("Minimum without precon:    $(mino[end-1])")
        println("Minimum with    precon:    $(mino[end])")
        println()
    end
end