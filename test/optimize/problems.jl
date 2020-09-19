OPT_PROBS = Dict()

#### Himmelblau

OPT_PROBS["himmelblau"] = Dict()
OPT_PROBS["himmelblau"]["array"] = Dict()

function himmelblau!(x)
    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    return fx
end
function himmelblau_batched_f!(X)
    F = map(himmelblau!, X)
    return F
end
function himmelblau_batched_f!(F, X)
    map!(himmelblau!, F, X)
    return F
end
function himmelblau_g!(∇f, x)
    ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    ∇f
end

function himmelblau_fg!(∇f, x)
    ∇f = himmelblau_g!(∇f, x)
    fx = himmelblau!(x)
    return fx, ∇f
end
function himmelblau_fgh!(∇f, ∇²f, x)
    ∇²f = himmelblau_h!(∇²f, x)
    fx, ∇f = himmelblau_fg!(∇f, x)
    return fx, ∇f, ∇²f
end
function himmelblau_h!(∇²f, x)
    ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
    ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
    ∇²f[2, 1] = ∇²f[1, 2]
    ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    return ∇²f
end
function himmelblau_hv!(hv, x, v)
    hv[1] = (12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0)*v[1] + (4.0 * x[1] + 4.0 * x[2])*v[2]
    hv[2] =  (4.0 * x[1] + 4.0 * x[2])*v[1] + (2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0)*v[2]
    return hv
end

OPT_PROBS["himmelblau"]["array"]["x0"] = [3.0, 1.0]
OPT_PROBS["himmelblau"]["array"]["mutating"] = ScalarObjective(himmelblau!, himmelblau_g!, himmelblau_fg!, himmelblau_fgh!, himmelblau_h!, himmelblau_hv!, himmelblau_batched_f!, nothing)


### Exponential

exponential!(x) = exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
function exponential_g!(g, x)
    g[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
    g[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    return g
end
function exponential_h!(H, x)
    H[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
    H[1, 2] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 2.0 * exp((3.0 - x[2])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
    return H
end
function exponential_hv!(Hv, x, v)
    Hv[1, 1] = (2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9))*v[1]
    Hv[2, 2] = (2.0 * exp((3.0 - x[2])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19))*v[2]
    return Hv
end
function exponential_fg!(g, x)
    fx = exponential!(x)
    g = exponential_g!(g, x)
    return fx, g
end
function exponential_fgh!(g, H, x)
    fx, g = exponential_fg!(g, x)
    H = exponential_h!(H, x)
    return fx, g, H
end

OPT_PROBS["exponential"] = Dict()
OPT_PROBS["exponential"]["array"] = Dict()
# Byttet om på x og H
OPT_PROBS["exponential"]["array"]["x0"] = [0.0, 0.0]
OPT_PROBS["exponential"]["array"]["mutating"] = ScalarObjective(exponential!, exponential_g!, exponential_fg!, exponential_fgh!, exponential_h!, exponential_hv!, nothing, nothing)

OPT_PROBS["laplacian"] = Dict()
OPT_PROBS["laplacian"]["array"] = Dict()
# Byttet om på x og H
OPT_PROBS["laplacian"]["array"]["x0(n)"] = n->zeros(n)

plap(U; n=length(U)) = (n-1) * sum((0.1 .+ diff(U).^2).^2) - sum(U) / (n-1)
plap1(U; n=length(U), dU = diff(U), dW = 4 .* (0.1 .+ dU.^2) .* dU) =
(n - 1) .* ([0.0; dW] .- [dW; 0.0]) .- ones(n) / (n-1)
precond(x::Vector) = precond(length(x))
precond(n::Number) = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1)) * (n+1)
_f(x) = plap([0;x;0])
function _fg(∇f, x)
    fx = _f(x)
    copyto!(∇f, (plap1([0;x;0]))[2:end-1])
    fx, ∇f
end

OPT_PROBS["laplacian"]["array"]["mutating"] = OptimizationProblem(ScalarObjective(_f, nothing, _fg, nothing, nothing, nothing, nothing, nothing))