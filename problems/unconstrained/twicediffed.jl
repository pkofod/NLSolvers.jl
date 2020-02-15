direct_tr = (NWI(), Dogleg())
direct_globals = (HZAW(), Backtracking(), Backtracking(;interp=FFQuadInterp()), NWI(), Dogleg())
inverse_ls = (HZAW(), Backtracking(), Backtracking(; interp=FFQuadInterp()))
inverse_globals = (HZAW(), Backtracking(), Backtracking(;interp=FFQuadInterp()))
run_all(td_obj, x0, BFGS(Inverse()), inverse_globals)
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals)
#run_all(td_obj, x0, SR1(Inverse()), inverse_ls)
run_all(td_obj, x0, DFP(Inverse()), inverse_globals)
run_all(td_obj, x0, BFGS(Direct()), direct_globals)
run_all(td_obj, x0, DBFGS(Direct()), direct_globals)
run_all(td_obj, x0, SR1(Direct()), direct_globals)
run_all(td_obj, x0, DFP(Direct()), direct_globals)

run_all(td_obj, x0, Newton(), direct_globals)
run_all(td_obj, x0, Newton(), direct_globals)

run_all(od_obj, x0, BFGS(Inverse()), inverse_globals)
run_all(od_obj, x0, DBFGS(Inverse()), inverse_globals)
run_all(od_obj, x0, SR1(Inverse()), inverse_ls)
run_all(od_obj, x0, DFP(Inverse()), inverse_globals)
run_all(od_obj, x0, BFGS(Direct()), direct_globals)
run_all(od_obj, x0, DBFGS(Direct()), direct_globals)
run_all(od_obj, x0, SR1(Direct()), direct_globals)
run_all(od_obj, x0, DFP(Direct()), direct_globals)

run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);

run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);


run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);

run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);

run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);


run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);



examples["Polynomial"] = OptimizationProblem("Polynomial",
                                             polynomial,
                                             polynomial_gradient!,
                                             nothing,
                                             polynomial_hessian!,
                                             nothing, # Constraints
                                             [0.0, 0.0, 0.0],
                                             [10.0, 7.0, 108.0],
                                             polynomial([10.0, 7.0, 108.0]),
                                             true,
                                             true)

##########################################################################
###
### Powell (d=4)
###
### Problem 35 in [1]
### Difficult since the hessian is singular at the optimum
##########################################################################


run_all(td_obj, x0, BFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);

##########################################################################
###
### Rosenbrock (2D)
###
### Problem 38 in [1]
###
### Saddle point makes optimization difficult
##########################################################################

td_obj = TwiceDiffed(rosenbrock!)

run_all(td_obj, x0, BFGS(Inverse()), inverse_globals[2:end]);
run_all(td_obj, x0, DBFGS(Inverse()), inverse_globals[2:end]);
run_all(td_obj, x0, SR1(Inverse()), inverse_ls[2:end]);
run_all(td_obj, x0, DFP(Inverse()), inverse_globals[2:end]);
run_all(td_obj, x0, BFGS(Direct()), direct_globals);
run_all(td_obj, x0, DBFGS(Direct()), direct_globals);
run_all(td_obj, x0, SR1(Direct()), direct_globals);
run_all(td_obj, x0, DFP(Direct()), direct_globals);
run_all(td_obj, x0, Newton(), direct_globals);
