using NLSolvers, StaticArrays
function theta(x)
   if x[1] > 0
       return atan(x[2] / x[1]) / (2.0 * pi)
   else
       return (pi + atan(x[2] / x[1])) / (2.0 * pi)
   end
end
f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

function f∇f!(x, ∇f)
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
    objective_return(fx, ∇f)
end
obj_inplace = OnceDiffed(f∇f!)
x0m = @MVector [-1.0, 0.0, 0.0]
x0 = [-1.0, 0.0, 0.0]
@time res = minimize!(obj_inplace, copy(x0m), ConjugateGradient(update=VPRP()), MinOptions());
@time res = minimize!(obj_inplace, copy(x0), ConjugateGradient(update=VPRP()), MinOptions());
@time_broken res = minimize!(obj_inplace, copy(x0m), LineSearch(BFGS()), MinOptions());
@time res = minimize!(obj_inplace, copy(x0), LineSearch(BFGS()), MinOptions());
