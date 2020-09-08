using NLSolvers
function G(x, Gx)
  V1 = [1.0, 0.0] .+ 0.99*[0.1 0.9; 0.5 0.5]*x
  V2 = [0.0, 2.0] .+ 0.99*[0.5 0.5; 1.0 0.0]*x
  K = max(maximum(V1), maximum(V2))
  Gx .= K .+ log.(exp.(V1 .- K) .+ exp.(V2 .- K))
end
fp1 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(10000000,1, nothing,nothing))
fp2 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(2, 2, 0.3, 1e2))
fp3 = NLSolvers.fixedpoint!(G, zeros(2), Anderson(2, 2, 0.01, 1e2))
@test all(fp1.x .≈ fp2.x)
@test all(fp1.x .≈ fp3.x)

fp1 = NLSolvers.solve!(NEqProblem((x, F)->G(x, F).-x), zeros(2), Anderson(10000000,1, nothing,nothing))
fp2 = NLSolvers.solve!(NEqProblem((x, F)->G(x, F).-x), zeros(2), Anderson(2, 2, 0.3, 1e2))
fp3 = NLSolvers.solve!(NEqProblem((x, F)->G(x, F).-x), zeros(2), Anderson(2, 2, 0.01, 1e2))
@test all(fp1.x .≈ fp2.x)
@test all(fp1.x .≈ fp3.x)
