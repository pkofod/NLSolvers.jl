# Property (*) means that the step size α*d is small -> β = 0
struct CGRestart{T} end
struct FletcherReeves{T} end
# using an "intial vectors" function we can initialize s if necessary or nothing if not to save on vectors
function optimize(obj::OnceDiffed, x0, cg::FletcherReeves, linesearch = Backtracking(); maxiter=100)
    T=eltype(x0)
    fx, ∇fx = obj(x0, copy(x0))
    Tf = typeof(fx)
    fz = fx
    d = -copy(∇fx)
    ∇fz = copy(∇fx)
    k = 0
    x = copy(x0)
    while norm(∇fx, Inf) ≥ 1e-8 && k ≤ maxiter
        α, f_α, ls_success = NLSolvers.find_steplength(linesearch, obj, d, x, Tf(1.0), fx, ∇fx)
        z=x+α*d
        fz, ∇fz = obj(z, ∇fz)

        βkp1 = calc_β(cg, ∇fz, ∇fx)
        fx = fz
        x .= z
        ∇fx .= ∇fz

        d .= -∇fx + βkp1*d
        k = k+1
    end
    x, fx
end
cg_restart(::FletcherReeves{<:CGRestart{true}}) = true

function calc_β(cg, ∇fz, ∇fx)
    T = eltype(∇fz)
    βkp1 = dot(∇fz, ∇fz)/dot(∇fx, ∇fx)
    if cg_restart(cg)
        βkp1 = max(βkp1, T(0))
    end
    βkp1
end
# function calc_β(method, ∇fx, ∇fz)
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
