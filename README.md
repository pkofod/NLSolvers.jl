# NLSolvers
Still under construction, so stuff will break (and improve!)

# Examples

## Scalar optimization (w/ different number types)
```
using NLSolvers, Test, DoubleFloats

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


res = minimize(myfun, Float64(4), BFGS(Inverse()))
res = minimize(myfun, Double32(4), BFGS(Inverse()))
res = minimize(myfun, Double64(4), BFGS(Inverse()))
res = minimize(myfun, Float64(4), Newton(Direct()))
res = minimize(myfun, Double32(4), Newton(Direct()))
res = minimize(myfun, Double64(4), Newton(Direct()))
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

## Custom solve
