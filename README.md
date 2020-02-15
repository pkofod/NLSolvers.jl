Problem types
SHOULD EACH TR HAVE AN UPDATER FIELD?

line search should have a short curcuit for very small steps

MinProblem
MaxProblem
NEqProblem
KrylovNEqProblem
NLsqProblem

ParallelObjWrapper (different types and then algorithms can call  `batch_value` and
    take advantage. We can also set up N runs and wait for N calls to the objective
    to launch on a gpu cluster (say N independent BFGS that just runs till whenever
        they need a f(x) and then they wait for the others))

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
