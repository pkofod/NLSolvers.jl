# Might consider Daniel 1967 that uses second order
# Make them inputs to a CG type
abstract type CGUpdate end
struct ConjugateGradient{Tu}
  update::Tu
end
ConjugateGradient(;update=HZ()) = ConjugateGradient(update)

#### Conjugate Descent [Fletcher] (CD)
struct CD <: CGUpdate end
_β(::CD, d, ∇fz, ∇fx, y) = -norm(∇fz)^2/dot(d, ∇fx)

#### Hager-Zhang (HZ)
struct HZ{Tη} <: CGUpdate
	η::Tη # a "forcing term"
end
HZ() = HZ(1/100)
function _β(cg::HZ, d, ∇fz, ∇fx, y)
	T = eltype(∇fz)

	ddy = dot(d, y)
	βN = dot(y.-T(2).*d.*norm(y)^2/ddy, ∇fz)/ddy

	ηk = -inv(norm(d)*min(T(cg.η), norm(∇fx)))

	βkp1 = max(βN, ηk)
end

#### Hestenes-Stiefel (HS)
struct HS <: CGUpdate end
_β(::HS, d, ∇fz, ∇fx, y) = dot(∇fz, y)/dot(d, y)

#### Fletcher-Reeves (FR)
struct FR <: CGUpdate end
_β(::FR, d, ∇fz, ∇fx, y) = norm(∇fz)^2/norm(∇fx)^2

#### Polak-Ribiére-Polyak (PRP)
struct PRP{Plus} end
PRP(;plus=true) = PRP{plus}()
_β(::PRP, d, ∇fz, ∇fx, y) = dot(∇fz, y)/norm(∇fx)^2
function _β(::PRP{true}, d, ∇fz, ∇fx, y)
  β = dot(∇fz, y)/norm(∇fx)^2
  max(typeof(β)(0), β)
end

#### Liu-Storey (LS)
struct LS <: CGUpdate end
_β(::LS, d, ∇fz, ∇fx, y) = dot(∇fz, y)/dot(d, ∇fx)

#### Dai-Yuan (DY)
struct DY <: CGUpdate end
_β(::DY, d, ∇fz, ∇fx, y) = norm(∇fz)^2/dot(d, y)

#### Wei-Yao-Liu (VPRP)
struct VPRP <: CGUpdate end
_β(::VPRP, d, ∇fz, ∇fx, y) = (norm(∇fz)^2-norm(∇fz)/norm(∇fx)*dot(∇fz, ∇fx))/dot(d, y)


# using an "intial vectors" function we can initialize s if necessary or nothing if not to save on vectors
minimize!(obj::ObjWrapper, x0, cg::ConjugateGradient; maxiter=100) =
  _minimize(obj, x0, (cg, HZAW()), InPlace(); maxiter=maxiter)
minimize!(obj::ObjWrapper, x0, approach::Tuple{<:ConjugateGradient, <:LineSearch}; maxiter=100) =
  _minimize(obj, x0, approach, InPlace(); maxiter=maxiter)
minimize(obj::ObjWrapper, x0, cg::ConjugateGradient; maxiter=100) =
  _minimize(obj, x0, (cg, HZAW()), OutOfPlace(); maxiter=maxiter)
minimize(obj::ObjWrapper, x0, approach::Tuple{<:ConjugateGradient, <:LineSearch}; maxiter=100) =
  _minimize(obj, x0, approach, OutOfPlace(); maxiter=maxiter)

function _minimize(obj::ObjWrapper, x0, approach::Tuple{<:ConjugateGradient, <:LineSearch}, mstyle::MutateStyle; maxiter=100)

	Tx=eltype(x0)
	cg, linesearch = approach

    fx, ∇fx = obj(x0, copy(x0))

    fz = fx
	α = Tx(0)

	d, ∇fz = -copy(∇fx), copy(∇fx)
    x, y = copy(x0), copy(∇fz)
	z = copy(x)
	k = 0
    while norm(∇fx, Inf) ≥ 1e-8 && k ≤ maxiter
		# Set up line search
        α_0 = initial(cg.update, a-> obj(x.+a.*d), α, k, x, fx, dot(d, ∇fx), ∇fx)
		φ = _lineobjective(mstyle, obj, ∇fz, z, x, d, fx, dot(∇fx, d))

		# Perform line search
		α, f_α, ls_success = find_steplength(linesearch, φ, Tx(1.0))
		# move in direction
		if mstyle isa InPlace
			@. z = x + α*d
        	fz, ∇fz = obj(z, ∇fz)
        	@. y = ∇fz - ∇fx
			βkp1 = _β(cg.update, d, ∇fz, ∇fx, y)
			fx = fz
			x .= z
			∇fx .= ∇fz

			@. d = -∇fx + βkp1*d
			k = k+1
		elseif mstyle isa OutOfPlace
			z = @. x + α*d
			fz, ∇fz = obj(z, ∇fz)
			y = @. ∇fz - ∇fx
			βkp1 = _β(cg.update, d, ∇fz, ∇fx, y)
			fx = fz
			x = copy(z)
			∇fx = copy(∇fz)

			d = @. -∇fx + βkp1*d
			k = k+1
		end
    end
    x, fx, ∇fx, k
end

function initial(cg, φ, α, k, x, φ₀, dφ₀, ∇fx)
    T = eltype(x)
    ψ₀ = T(0.01)
    ψ₁ = T(0.1)
    ψ₂ = T(2.0)
    quadstep = true
    if k == 0
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


# function _β(method, ∇fx, ∇fz)
#
# if method == :fr
# return dot(∇fz, ∇fz)/dot(∇fx, ∇fx)
# elseif method == :pr
# β = dot(∇fz, s)/dot(∇fx, ∇fx) # dot(∇fx, ∇fx) can be saved between iteratoins only s is needed
# return    max(β, 0) # We need this according to Powell 1986 Gilbert Powell 1980 show it is globally  convergenct for exact and inexact
# # pr+ max(beta, 0)
# elseif method ==:hs
# dot(∇fz, s)/dot(s,p)
# end
