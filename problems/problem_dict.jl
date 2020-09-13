OPT_PROBS = Dict()

# Himmelblau

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
function himmelblau_g!(x, ∇f)
    if !(∇f === nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end
    ∇f
end
function himmelblau_fg!(x, ∇f)
    if !(∇f === nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f)
end
function himmelblau_fgh!(x, ∇f, ∇²f)
    if !(∇²f === nothing)
        ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
        ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 1] = ∇²f[1, 2]
        ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    end
    if !(∇f === nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f, ∇²f)
end
function himmelblau_h!(x, ∇²f)
    if !(∇²f === nothing)
        ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
        ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 1] = ∇²f[1, 2]
        ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    end
    ∇²f
end
function himmelblau_hv!(x, v, hv)
    if !(∇²f === nothing)
        hv[1] = (12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0)*v[1] + (4.0 * x[1] + 4.0 * x[2])*v[2]
        hv[2] =  (4.0 * x[1] + 4.0 * x[2])*v[1] + (2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0)*v[2]
    end
    hv
end

OPT_PROBS["himmelblau"]["array"]["x0"] = [3.0, 1.0]
OPT_PROBS["himmelblau"]["array"]["mutating"] = ScalarObjective(himmelblau!, himmelblau_g!, himmelblau_fg!, himmelblau_fgh!, himmelblau_h!, himmelblau_hv!, himmelblau_batched_f!)

# 

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
function himmelblau_g!(x, ∇f)
    if !(∇f === nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end
    ∇f
end
function himmelblau_fg!(x, ∇f)
    if !(∇f === nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f)
end
function himmelblau_fgh!(x, ∇f, ∇²f)
    if !(∇²f === nothing)
        ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
        ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 1] = ∇²f[1, 2]
        ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    end
    if !(∇f === nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    objective_return(fx, ∇f, ∇²f)
end
function himmelblau_h!(x, ∇²f)
    if !(∇²f === nothing)
        ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0
        ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 1] = ∇²f[1, 2]
        ∇²f[2, 2] = 2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0
    end
    ∇²f
end
function himmelblau_hv!(x, v, hv)
    if !(∇²f === nothing)
        hv[1] = (12.0 * x[1]^2 + 4.0 * x[2] - 44.0 + 2.0)*v[1] + (4.0 * x[1] + 4.0 * x[2])*v[2]
        hv[2] =  (4.0 * x[1] + 4.0 * x[2])*v[1] + (2.0 + 4.0 * x[1] + 12.0 * x[2]^2 - 28.0)*v[2]
    end
    hv
end

OPT_PROBS["himmelblau"]["array"]["x0"] = [3.0, 1.0]
OPT_PROBS["himmelblau"]["array"]["mutating"] = ScalarObjective(himmelblau!, himmelblau_g!, himmelblau_fg!, himmelblau_fgh!, himmelblau_h!, himmelblau_hv!, himmelblau_batched_f!)

