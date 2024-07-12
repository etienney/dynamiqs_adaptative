import jax

def wrap_solver(solver_class):

    class WrappedSolver(solver_class):

        def update_est(self, t0, x, est):
            est = 0
            return est

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            (y, y_error, dense_info, solver_state, solver_result) = super().step(terms, t0, t1, y0, args, solver_state, made_jump)
            x = y
            est = 0
            new_est = self.update_est(t0, x, est)
            jax.debug.print("eee{e}", e = est)
            return (x, y_error, dense_info, solver_state, solver_result)
    jax.debug.print("eeeddd")
    return WrappedSolver