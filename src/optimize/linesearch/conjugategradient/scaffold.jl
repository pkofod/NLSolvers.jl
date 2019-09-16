# Property (*) means that the step size α*d is small -> β = 0

# using an "intial vectors" function we can initialize s if necessary or nothing if not to save on vectors
function fletcher_reeves(f, ∇f, x0)
f0 = f(x0)
∇f0 = ∇f(x0)
p0 = -∇f0
pk = copy(p0)
k = 0
xk = copy(x0)
while abs(∇f0) ≥ 1e-8 && k ≤ maxiter
    α = 0.5
    xkp1=xk+α*pk
    ∇fkp1 = ∇f(xkp1)
    βkp1 = dot(∇fkp1, ∇fkp1)/dot(∇fk)
    pkp1 = -∇fkp1 + βkp1*pk
    k = k+1
end
end

function calc_β(method, ∇fk, ∇fkp1)

if method == :fr
return dot(∇fkp1, ∇fkp1)/dot(∇fk, ∇fk)
elseif method == :pr
β = dot(∇fkp1, s)/dot(∇fk, ∇fk) # dot(∇fk, ∇fk) can be saved between iteratoins only s is needed
return    max(β, 0) # We need this according to Powell 1986 Gilbert Powell 1980 show it is globally  convergenct for exact and inexact
# pr+ max(beta, 0)
elseif method ==:hs
dot(∇fkp1, s)/dot(s,p)
end
