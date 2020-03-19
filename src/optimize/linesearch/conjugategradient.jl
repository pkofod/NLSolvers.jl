#===============================================================================
  Conjugate Gradient Descent

  We implement a generic conjugate gradient descent method that includes many
  different β-choices, preconditioning, precise line searches, and more.

  A conjugate gradient method here will have to implement a CGUpdate type (CD,
  HZ, HS, ...) that controls the β update.

  Todos:
  Might consider Daniel 1967 that uses second order
===============================================================================#

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

function prepare_variables(objective, approach::LineSearch{<:ConjugateGradient, <:Any, <:Any}, x0, ∇fz)
    z = x0
    x = copy(z)

    fz, ∇fz = objective(x, ∇fz)

    fx = copy(fz)
    ∇fx = copy(∇fz)

    Pg = approach.scheme.P isa Nothing ? ∇fz : copy(∇fz)
    return x, fx, ∇fx, z, fz, ∇fz, Pg
end

summary(::ConjugateGradient) = "Conjugate Gradient Descent"
#===============================================================================
  Conjugate Descent [Fletcher] (CD)
===============================================================================#
struct CD <: CGUpdate end
function _β(mstyle, cg::CD, d, ∇fz, ∇fx, y, P, Pg)
  num = -dot(∇fz, Pg)
  P∇fx = apply_preconditioner(mstyle, Pg, P, ∇fx)
  den = dot(d, P∇fx)
  num/den
end

#===============================================================================
  Hager-Zhang (HZ)
===============================================================================#
struct HZ{Tη} <: CGUpdate
    η::Tη # a "forcing term"
end
HZ() = HZ(1/100)
function _β(mstyle, cg::HZ, d, ∇fz, ∇fx, y, P, Pg)
    T = eltype(∇fz)

    ddy = dot(d, y)
    βN = dot(y.-T(2).*d.*norm(y)^2/ddy, ∇fz)/ddy

    ηk = -inv(norm(d)*min(T(cg.η), norm(∇fx)))

    βkp1 = max(βN, ηk)
end

#===============================================================================
  Hestenes-Stiefel (HS)
===============================================================================#
struct HS <: CGUpdate end
function _β(mstyle, cg::HS, d, ∇fz, ∇fx, y, P, P∇fz)
  num = dot(P∇fz, y)
  Py = apply_preconditioner(mstyle, P∇fz, P, y)
  num/dot(d, Py)
end
#===============================================================================
  Fletcher-Reeves (FR)
===============================================================================#
struct FR <: CGUpdate end
function _β(mstyle, cg::FR, d, ∇fz, ∇fx, y, P, P∇fz)
    num = dot(∇fz, P∇fz)
    P∇fx = apply_preconditioner(mstyle, P∇fz, P, ∇fx)
    num/dot(∇fx, P∇fx)
end
#===============================================================================
  Polak-Ribiére-Polyak (PRP)
===============================================================================#
struct PRP{Plus} end
PRP(;plus=true) = PRP{plus}()
function _β(mstyle, ::PRP, d, ∇fz, ∇fx, y, P, P∇fz)
    num = dot(y, P∇fz)
    P∇fx = apply_preconditioner(mstyle, P∇fz, P, ∇fx)
    num/dot(∇fx, P∇fx)
end
function _β(::PRP{true}, d, ∇fz, ∇fx, y, P, P∇fz)
  β = _β(RP{false}(), d, ∇fz, ∇fx, y, P, P∇fz)
  max(typeof(β)(0), β)
end

#===============================================================================
  Liu-Storey (LS)
===============================================================================#
struct LS <: CGUpdate end
function _β(mstyle, ::LS, d, ∇fz, ∇fx, y, P, P∇fz)
  num = dot(y, P∇fz)
  P∇fx = apply_preconditioner(mstyle, P∇fz, P, ∇fx)
  num/dot(d, P∇fx)
end
#===============================================================================
  Dai-Yuan (DY)
===============================================================================#
struct DY <: CGUpdate end
function _β(mstyle, cg::DY, d, ∇fz, ∇fx, y, P, P∇fz)
  num = dot(∇fz, P∇fz)
  Py = apply_preconditioner(mstyle, P∇fz, P, y)
  num/dot(d, Py)
end

#===============================================================================
  Wei-Yao-Liu (VPRP)
===============================================================================#
struct VPRP <: CGUpdate end
function _β(mstyle, cg::VPRP, d, ∇fz, ∇fx, y, P, P∇fz)
  a = dot(∇fz, P∇fz)
  b = dot(∇fx, P∇fz)
  P∇fx = apply_preconditioner(mstyle, P∇fz, P, ∇fx)
  c = sqrt(dot(∇fx, P∇fx))
  num1 = a - sqrt(a)/c*b
  Py = apply_preconditioner(mstyle, P∇fz, P, y)
  num/dot(d, Py)
end

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

    x, fx, ∇fx, z, fz, ∇fz, Pg = prepare_variables(obj, approach, x0, copy(x0))
    f0, ∇f0 = fx, norm(∇fx)

    y, d, α, β = copy(∇fz), -copy(∇fx), Tx(0), Tx(0)
    cgvars = CGVars(y, d, α, β, true)

    k = 1
    x, fx, ∇fx, z, fz, ∇fz, P, cgvars = iterate(mstyle, cgvars, x, fx, ∇fx, z, fz, ∇fz, Pg, approach, prob, obj, options)
    is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options)
    while k ≤ options.maxiter && !any(is_converged)
        k += 1
        x, fx, ∇fx, z, fz, ∇fz, P, cgvars = iterate(mstyle, cgvars, x, fx, ∇fx, z, fz, ∇fz, Pg, approach, prob, obj, options, P, false)
        is_converged = converged(approach, x, z, ∇fz, ∇f0, fx, fz, options) 
    end
    return ConvergenceInfo(approach, (beta=β, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fx=∇fx, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=k, time=time()-t0), options)
end
function iterate(mstyle::InPlace, cgvars::CGVars, x, fx, ∇fx, z, fz, ∇fz, Pg, approach::LineSearch{<:ConjugateGradient, <:Any, <:Any}, prob::MinProblem, obj::ObjWrapper, options::MinOptions, P=nothing, is_first=nothing)
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
    P∇fz = apply_preconditioner(mstyle, P, Pg,  ∇fz)
    # Update current gradient and calculate the search direction
    @. d = -P∇fz + β*d

    α_0 = initial(scheme.update, a-> obj(x.+a.*d), α, x, fx, dot(d, ∇fx), ∇fx, is_first)
    φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tx(1))

    # Calculate final step vector and update the state
    @. z = x + α*d
    fz, ∇fz = obj(z, ∇fz)
    @. y = ∇fz - ∇fx
    β = _β(mstyle, scheme.update, d, ∇fz, ∇fx, y, P, P∇fz)

    return x, fx, ∇fx, z, fz, ∇fz, P, CGVars(y, d, α, β, ls_success)
end

function iterate(mstyle::OutOfPlace, cgvars::CGVars, x, fx, ∇fx, z, fz, ∇fz, Pg, approach::LineSearch{<:ConjugateGradient, <:Any, <:Any}, prob::MinProblem, obj::ObjWrapper, options::MinOptions, P=nothing, is_first=nothing)
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
    P∇fz = apply_preconditioner(mstyle, P, Pg, ∇fz)
    # Update current gradient and calculate the search direction
    d = @. -P∇fz + β*d

    α_0 = initial(scheme.update, a-> obj(x .+ a.*d), α, x, fx, dot(d, ∇fx), ∇fx, is_first)
    φ = _lineobjective(mstyle, prob, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tx(1))

    z = @. x + α*d
    fz, ∇fz = obj(z, ∇fz)
    y = @. ∇fz - ∇fx
    β = _β(mstyle, scheme.update, d, ∇fz, ∇fx, y, P, P∇fz)

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
