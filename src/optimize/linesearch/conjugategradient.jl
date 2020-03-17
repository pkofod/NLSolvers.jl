# Might consider Daniel 1967 that uses second order
# add preconditioner
abstract type CGUpdate end
struct ConjugateGradient{Tu, TP}
  update::Tu
  P::TP
end
ConjugateGradient(;update=HZ()) = ConjugateGradient(update, nothing)
hasprecon(::ConjugateGradient{<:Any, <:Nothing}) = NoPrecon()
hasprecon(::ConjugateGradient{<:Any, <:Any}) = HasPrecon()

struct CGVars{T1, T2, T3}
    y::T1 # change in successive gradients
    d::T2 # search direction
    α::T3
    β::T3
    ls_success::Bool
end

function prepare_variables(objective, approach::LineSearch{<:ConjugateGradient, <:Any}, x0, ∇fz)
    z = x0
    x = copy(z)

    # first evaluation
    fz, ∇fz = objective(x, ∇fz)

    fx = copy(fz)
    ∇fx = copy(∇fz)
    return x, fx, ∇fx, z, fz, ∇fz
end

summary(::ConjugateGradient) = "Conjugate Gradient Descent"
#### Conjugate Descent [Fletcher] (CD)
struct CD <: CGUpdate end
_β(::CD, d, ∇fz, ∇fx, y, P) = -norm(∇fz)^2/dot(d, ∇fx)

#### Hager-Zhang (HZ)
struct HZ{Tη} <: CGUpdate
    η::Tη # a "forcing term"
end
HZ() = HZ(1/100)
function _β(cg::HZ, d, ∇fz, ∇fx, y, P)
    T = eltype(∇fz)

    ddy = dot(d, y)
    βN = dot(y.-T(2).*d.*norm(y)^2/ddy, ∇fz)/ddy

    ηk = -inv(norm(d)*min(T(cg.η), norm(∇fx)))

    βkp1 = max(βN, ηk)
end

#### Hestenes-Stiefel (HS)
struct HS <: CGUpdate end
_β(::HS, d, ∇fz, ∇fx, y, P) = dot(∇fz, y)/dot(d, y)

#### Fletcher-Reeves (FR)
struct FR <: CGUpdate end
_β(::FR, d, ∇fz, ∇fx, y, P) = norm(∇fz)^2/norm(∇fx)^2

#### Polak-Ribiére-Polyak (PRP)
struct PRP{Plus} end
PRP(;plus=true) = PRP{plus}()
_β(::PRP, d, ∇fz, ∇fx, y, P) = dot(∇fz, y)/norm(∇fx)^2
function _β(::PRP{true}, d, ∇fz, ∇fx, y, P)
  β = dot(∇fz, y)/norm(∇fx)^2
  max(typeof(β)(0), β)
end

#### Liu-Storey (LS)
struct LS <: CGUpdate end
_β(::LS, d, ∇fz, ∇fx, y, P) = dot(∇fz, y)/dot(d, ∇fx)

#### Dai-Yuan (DY)
struct DY <: CGUpdate end
_β(::DY, d, ∇fz, ∇fx, y, P) = norm(∇fz)^2/dot(d, y)

#### Wei-Yao-Liu (VPRP)
struct VPRP <: CGUpdate end
_β(::VPRP, d, ∇fz, ∇fx, y, P) = (norm(∇fz)^2-norm(∇fz)/norm(∇fx)*dot(∇fz, ∇fx))/dot(d, y)


# using an "initial vectors" function we can initialize s if necessary or nothing if not to save on vectors
minimize!(obj::ObjWrapper, x0, cg::ConjugateGradient, options::MinOptions) =
  _minimize(MinProblem(;obj=obj), x0, LineSearch(cg, HZAW()), options, InPlace())
minimize!(obj::ObjWrapper, x0, approach::LineSearch{<:ConjugateGradient, <:LineSearcher}, options::MinOptions) =
  _minimize(MinProblem(;obj=obj), x0, approach, options, InPlace())
minimize(obj::ObjWrapper, x0, cg::ConjugateGradient, options::MinOptions) =
  _minimize(MinProblem(;obj=obj), x0, LineSearch(cg, HZAW()), options, OutOfPlace())
minimize(obj::ObjWrapper, x0, approach::LineSearch{<:ConjugateGradient, <:LineSearcher}, options::MinOptions) =
  _minimize(MinProblem(;obj=obj), x0, approach, options, OutOfPlace())

function _minimize(prob::MinProblem, x0, approach::LineSearch{<:ConjugateGradient, <:LineSearcher}, options::MinOptions, mstyle::MutateStyle)
    t0 = time()

    obj = prob.objective
    #==============
         Setup
    ==============#
    Tx = eltype(x0)

    x, fx, ∇fx, z, fz, ∇fz = prepare_variables(obj, approach, x0, copy(x0))
    f0, ∇f0 = fx, norm(∇fx)

    P = initial_preconditioner(approach, x0)

    y, d, α, β = copy(∇fz), -copy(∇fx), Tx(0), Tx(0)
    cgvars = CGVars(y, d, α, β, true)

    k = 1
    x, fx, ∇fx, z, fz, ∇fz, P, cgvars = iterate(mstyle, cgvars, x, fx, ∇fx, z, fz, ∇fz, P, approach, prob, obj, options)
    is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options)
    while k ≤ options.maxiter && !any(is_converged)
        k += 1
        x, fx, ∇fx, z, fz, ∇fz, P, cgvars = iterate(mstyle, cgvars, x, fx, ∇fx, z, fz, ∇fz, P, approach, prob, obj, options, false)
        is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options) 
    end
    return ConvergenceInfo(approach, (beta=β, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fx=∇fx, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=k, time=time()-t0), options)
end
function iterate(mstyle::InPlace, cgvars::CGVars, x, fx, ∇fx, z, fz, ∇fz, P, approach::LineSearch{<:ConjugateGradient, <:Any}, prob, obj::ObjWrapper, options::MinOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    Tx = eltype(x)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    y, d, α, β = cgvars.y, cgvars.d, cgvars.α, cgvars.β

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Update preconditioner
    P = update_preconditioner(approach, x, P)
    # Update current gradient and calculate the search direction
    @. d = -∇fx + β*d

    α_0 = initial(scheme.update, a-> obj(x.+a.*d), α, x, fx, dot(d, ∇fx), ∇fx, is_first)
    φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tx(1))

    # Calculate final step vector and update the state
    @. z = x + α*d
    fz, ∇fz = obj(z, ∇fz)
    @. y = ∇fz - ∇fx
    β = _β(scheme.update, d, ∇fz, ∇fx, y, P)

    return x, fx, ∇fx, z, fz, ∇fz, P, CGVars(y, d, α, β, ls_success)
end

function iterate(mstyle::OutOfPlace, cgvars::CGVars, x, fx, ∇fx, z, fz, ∇fz, P, approach::LineSearch{<:ConjugateGradient, <:Any}, prob, obj::ObjWrapper, options::MinOptions, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    Tx = eltype(x)
    scheme, linesearch = modelscheme(approach), algorithm(approach)
    d, α, β = cgvars.d, cgvars.α, cgvars.β

    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Update preconditioner
    P = update_preconditioner(approach, x, P)
    d = @. -∇fx + β*d

    α_0 = initial(scheme.update, a-> obj(x.+a.*d), α, x, fx, dot(d, ∇fx), ∇fx, is_first)
    φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tx(1))

    z = @. x + α*d
    fz, ∇fz = obj(z, ∇fz)
    y = @. ∇fz - ∇fx
    β = _β(scheme.update, d, ∇fz, ∇fx, y, P)

    return x, fx, ∇fx, z, fz, ∇fz, P, CGVars(y, d, α, β, ls_success)
end



function initial(cg, φ, α, x, φ₀, dφ₀, ∇fx, is_first)
    T = eltype(x)
    ψ₀ = T(0.01)
    ψ₁ = T(0.1)
    ψ₂ = T(2.0)
    quadstep = true
    if is_first isa Nothing
        if !all(x .≈ T(0)) # should we define "how approx we want?"
            return ψ₀ * norm(x, Inf)/norm(∇fx, Inf)
        elseif !(φ₀ ≈ T(0))
            return ψ₀ * abs(φ₀)/norm(∇fx, 2)^2
        else
            return T(1)
        end
    elseif quadstep
        R = ψ₁*α
        φR = φ(R)
        if φR ≤ φ₀
            c = (φR - φ₀ - dφ₀*R)/R^2
            if c > 0
               return -dφ₀/(T(2)*c) # > 0 by df0 < 0 and c > 0
            end
        end
    end
    return ψ₂*α
end
