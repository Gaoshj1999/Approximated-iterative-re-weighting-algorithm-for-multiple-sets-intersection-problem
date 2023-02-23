# Approximated-iterative-re-weighting-algorithm-for-multiple-sets-intersection-problem
The problem of minimizing a convex function on the intersection of multiple sets
appears in many machine learning and large-scale optimization problems. The main
difficulty of this problem is to find the projection of points outside the intersection on
the intersection In the past decades, many first-order algorithms have been proposed
to solve the problem that constraint sets have efficient projection counting algorithms,
but it is still challenging to solve the problem that constraint sets without such efficient
algorithms such as the constraint sets are the intersection of multiple sets. The iterative
re-weighting method (IRWA) solves the intersection projection onto multipl sets by the
weighted approximation of the objective function, but the subproblems of this problem
need to be solved exactly, which requires large computational cost in large scale setting.
In this paper, we propose an approximated iterative re-weighting algorithm (AIRWA),
which inexactly solves the subproblems by a primal-dual algorithm. We prove the subproblem algorithm has a convergence rate of 𝑂(1/N)
In the numerical experiment, we show that the proposed algorithm outperforms the IRWA algorithm by test results on
large scale quadratic programming problems.
