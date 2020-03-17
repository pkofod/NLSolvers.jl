using NLSolvers, StaticArrays
@testset "Random search" begin
function himmelblau!(x, ∇f)
    if !(∇f == nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f)
end


function himmelblaus(x, ∇f)
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

function himmelblau(x, ∇f)
    g = ∇f == nothing ? ∇f : similar(x)

    return himmelblau!(x, g)
end

him_inplace = OnceDiffed(himmelblau!)
him_static = OnceDiffed(himmelblaus)
him_outofplace = OnceDiffed(himmelblau)

minimize!(him_inplace, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]))
minimize(him_inplace, PureRandomSearch(lb=[0.0,0.0], ub=[4.0,4.0]))

minimize!(him_inplace, [0.0,0.0], SimulatedAnnealing(), MinOptions())
minimize(him_inplace, [0.0,0.0], SimulatedAnnealing(), MinOptions())
end