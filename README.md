# NLSolvers
| **Source**  | **Build Status** |
|:-:|:-:|
| [![Source](https://img.shields.io/badge/GitHub-source-green.svg)](https://github.com/pkofod/NLSolvers.jl) |  [![Build Status](https://travis-ci.org/pkofod/NLSolvers.jl.svg?branch=master)](https://travis-ci.org/pkofod/NLSolvers.jl) | [![](https://badges.gitter.im/pkofod/NLSolvers.jl.svg)](https://gitter.im/pkofod/NLSolvers.jl) |
| [![Codecov branch](https://img.shields.io/codecov/c/github/pkofod/NLSolvers.jl/master.svg)](https://codecov.io/gh/pkofod/NLSolvers.jl) |[![Build Status](https://ci.appveyor.com/api/projects/status/prp8ygfp4rr9tafe?svg=true)](https://ci.appveyor.com/project/blegat/optim-jl) 

NLSolvers provides optimization, curve fitting, and equation solving functionalities for Julia.
The goal is to provide a set of robust and flexible methods that runs fast and is easy to use.

## solve and solve!
NLSolvers.jl uses different problem types for different problems. For example, a `MinProblem` would
be `solve!`ed or `solve`ed depending of the circumstances, where an objective function can be
`minimize!`ed or `minimized`.

```
using NLSolvers
function scalarobj(x, ∇f, ∇²f)
    if ∇²f !== nothing
        ∇²f = 12x^2 - sin(x)
    end
    if ∇f !== nothing
        ∇f = 4x^3 + cos(x)
    end

    fx = x^4 + sin(x)
    objective_return(fx, ∇f, ∇²f)
end
scalar_obj = TwiceDiffed(scalarobj)

minimize(scalar_obj, 4.0, LineSearch(Newton()), MinOptions())
```
which returns
```
Results of minimization

* Algorithm:
  Newton's method with default linsolve with backtracking (no interp)

* Candidate solution:
  Final objective value:    -4.35e-01
  Final gradient norm:      5.52e-14

  Initial objective value:  2.55e+02
  Initial gradient norm:    2.55e+02

* Convergence measures
  |x - x'|              = 8.58e-08 <= 0.00e+00 (false)
  |x - x'|/|x|          = 1.45e-07 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 1.75e-14 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(function scalarobj(x, ∇f, ∇²f)
    if ∇²f !== nothing
        ∇²f = 12x^2 - sin(x)
    end
    if ∇f !== nothing
        ∇f = 4x^3 + cos(x)
    end

    fx = x^4 + sin(x)
    objective_return(fx, ∇f, ∇²f)
end
scalar_obj = TwiceDiffed(scalarobj)

minimize(scalar_obj, 4.0, LineSearch(Newton()), MinOptions())x)| = 4.03e-14 <= 0.00e+00 (false)
  |g(x)|                = 5.52e-14 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 2.16e-16 <= 0.00e+00 (false)

* Work counters
  Seconds run:   3.10e-06
  Iterations:    10
```
Now, if we had cast this as a MinProblem, we would have defined a very simple problem instance because we have no bounds
```
mp = MinProblem(scalar_obj)
```
Then, we would use `solve` to solve the instance
```
solve(mp, x0, , LineSearch(Newton()), MinOptions())
```
and then 
```
julia> solve(mp, 4.0, LineSearch(ConjugateGradient()), MinOptions())
```
which gives
```
Results of minimization

* Algorithm:
  Conjugate Gradient Descent (HZ) with backtracking (no interp)

* Candidate solution:
  Final objective value:    -4.35e-01
  Final gradient norm:      2.88e-09

  Initial objective value:  2.55e+02
  Initial gradient norm:    2.55e+02

* Convergence measures
  |x - x'|              = 4.11e-07 <= 0.00e+00 (false)
  |x - x'|/|x|          = 6.94e-07 <= 0.00e+00 (false)
  |f(x) - f(x')|        = 4.01e-13 <= 0.00e+00 (false)
  |f(x) - f(x')|/|f(x)| = 9.22e-13 <= 0.00e+00 (false)
  |g(x)|                = 2.88e-09 <= 1.00e-08 (true)
  |g(x)|/|g(x₀)|        = 1.13e-11 <= 0.00e+00 (false)

* Work counters
  Seconds run:   1.94e-01
  Iterations:    18

```

Two types of functions:
WorkVars # x, F, J, H, ??
AlgVars # s, y, z, ...
Documented in each type's docstring including LineSearch, BFGS, ....

AlgVars = (LSVars, QNVars, ...)
OptVars?
Initial modelvars and QNvars

tracing!
Abstract arrays!!! :|
manifolds
Use user norms
MArray support
Banded Jacobian
AD
nan return, nan gradient, nan hessian

line search should have a short curcuit for very small steps

MaxProblem
KrylovNEqProblem
NLsqProblem

[[[iterate(Problem) -> (state), iterate(state) -> state]]]
[[[on a state you can call -> lsiterate or triterate to sub-iterate on the line search problem]]]
# NLSolvers
Still under construction, so stuff will break (and improve!). Feel free to reach out with ideas and requests.

## Installation
Installing NLSolvers is easy, simply write
```
using Pkg
Pkg.add("NLSolvers")
```

or use the REPL mode to install the package
```
]add NLSolvers
```

## Documentation

## Common interface

See OptimAll.jl

This provides an interface for other sovers as well
You can provide

Option instances for th package
Args for the package
Kwargs for the packages
Algs for the packages

## Scalar optimization (w/ different number types)
```
using NLSolvers, DoubleFloats

function myfun(x)
    myfun(nothing, nothing, x)
end

function myfun(∇f, x)
    myfun(nothing, ∇f, x)
end
function myfun(∇²f, ∇f, x::T) where T
   if !(∇²f == nothing)
       ∇²f = 12x^2 - sin(x)
   end
   if !(∇f == nothing)
       ∇f = 4x^3 + cos(x)
   end

   fx = x^4 +sin(x)
   if ∇f == nothing && ∇²f == nothing
       return T(fx)
   elseif ∇²f == nothing
       return T(fx), T(∇f)
   else
       return T(fx), T(∇f), T(∇²f)
   end
end
my_obj_1 = OnceDiffed(myfun)
res = minimize(my_obj_1, Float64(4), BFGS(Inverse()))
res = minimize(my_obj_1, Double32(4), BFGS(Inverse()))
res = minimize(my_obj_1, Double64(4), BFGS(Inverse()))
my_obj_2 = TwiceDiffed(myfun)
res = minimize(my_obj_2, Float64(4), Newton())
res = minimize(my_obj_2, Double32(4), Newton())
res = minimize(my_obj_2, Double64(4), Newton())
```
## Multivariate optimization (w/ different number and array types)
```
using NLSolvers, StaticArrays
function theta(x)
   if x[1] > 0
       return atan(x[2] / x[1]) / (2.0 * pi)
   else
       return (pi + atan(x[2] / x[1])) / (2.0 * pi)
   end
end
f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

function f∇f!(∇f, x)
    if !(∇f==nothing)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        end
        ∇f[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
        ∇f[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
        ∇f[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
    end

    fx = f(x)
    return ∇f==nothing ? fx : (fx, ∇f)
end

function f∇f(∇f, x)
    if !(∇f == nothing)
        gx = similar(x)
        return f∇f!(gx, x)
    else
        return f∇f!(∇f, x)
    end
end
function f∇fs(∇f, x)
    if !(∇f == nothing)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
        end

        s1 = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
        s2 = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
        s3 = 200.0*(x[3]-10.0*theta(x)) + 2.0*x[3]
        ∇f = @SVector [s1, s2, s3]
        return f(x), ∇f
    else
        return f(x)
    end
end

x0 = [-1.0, 0.0, 0.0]
res = minimize(f∇f, x0, DFP(Inverse()))
res = minimize!(f∇f!, copy(x0), DFP(Inverse()))

x0s = @SVector [-1.0, 0.0, 0.0]
res = minimize(f∇fs, x0s, DFP(Inverse()))

```

# Second order optimization
```
    using NLSolvers
    function himmelblau!(x)
        fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
        return fx
    end
    function himmelblau!(∇f, x)
        if !(∇f == nothing)
            ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
                44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
            ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        end

        fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
        return ∇f == nothing ? fx : (fx, ∇f)
    end

    function himmelblau!(∇²f, ∇f, x)
        if !(∇²f == nothing)
            ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
            ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
            ∇²f[2, 1] = 4.0 * x[1] + 4.0 * x[2]
            ∇²f[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
        end

        if ∇f == nothing && ∇²f == nothing
            fx = himmelblau!(∇f, x)
            return fx
        elseif ∇²f == nothing
            return himmelblau!(∇f, x)
        else
            fx, ∇f = himmelblau!(∇f, x)
            return fx, ∇f, ∇²f
        end
    end

    res = minimize!(NonDiffed(himmelblau!), copy([2.0,2.0]), NelderMead())
    res = minimize!(OnceDiffed(himmelblau!), copy([2.0,2.0]), BFGS())
    res = minimize!(TwiceDiffed(himmelblau!), copy([2.0,2.0]), Newton())
```

# Mix'n'match
```
using NLSolvers
function himmelblau!(∇f, x)
    if !(∇f == nothing)
        ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    end

    fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    return ∇f == nothing ? fx : (fx, ∇f)
end

function himmelblau!(∇²f, ∇f, x)
    if !(∇²f == nothing)
        ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
        ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 1] = 4.0 * x[1] + 4.0 * x[2]
        ∇²f[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
    end


    if ∇f == nothing && ∇²f == nothing
        fx = himmelblau!(∇f, x)
        return fx
    elseif ∇²f == nothing
        return himmelblau!(∇f, x)
    else
        fx, ∇f = himmelblau!(∇f, x)
        return fx, ∇f, ∇²f
    end
end

res = minimize!(himmelblau!, copy([2.0,2.0]), Newton(Direct()))
res = minimize!(himmelblau!, copy([2.0,2.0]), (Newton(Direct()), Backtracking()))
res = minimize!(himmelblau!, copy([2.0,2.0]), (Newton(Direct()), NWI()))
```

## Custom solve
Newton methods generally accept a linsolve argument.

## Preconditioning
Many methods accept preconditioners. A preconditioner is provided as a function that has two methods: `p(x)` and `p(x, P)` where the first prepares and returns the preconditioner and the second is the signature for updating the preconditioner. If the preconditioner is constant, both method
will simply return this preconditioner. A preconditioner is used in two contexts: in `ldiv!(pgr, factorize(P), gr)` that accepts a cache array for the preconditioned gradient `pgr`, the preconditioner `P`, and the gradient to be preconditioned `gr`, and in `mul!(x, P, y)`. For the out-of-place methods (`minimize` as opposed to `minimize!`) it is sufficient to have `\(P, gr)` and `*(P, y)` defined.

## Wrapping a LeastSquares problem for MinProblems
To be able to do inplace least squares problems it is necessary to provide proper cache arrays to be used internally. To do this we write

```julia
@. model(x, p) = p[1]*exp(-x*p[2])
xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

using ForwardDiff
function F(p)
  model(xdata, p)
end
function J(p)
  ForwardDiff.jacobian(F, p)
end
function obj(_J, _F, x)
    f = F(x)
    j = _J isa Nothing ? _J : J(x)
    objective_return(f, j)
end
od = OnceDiffed(obj)
lw = LsqWrapper1(od, true, true)
```
# Beware, chaotic gradient methods!
https://link.springer.com/article/10.1007/s10915-011-9521-3

# next steps 
Mixed complementatiry
SAMIN, BOXES, Projected solver
Univariate!!
IP Newotn
Krylov Hessian
LsqFit wrapper

