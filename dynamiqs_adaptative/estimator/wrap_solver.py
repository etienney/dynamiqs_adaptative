
def wrap_solver(solver_class):

    class WrappedSolver(solver_class):

        def update_est(self, t0, x, est):
            est = 0
            return est

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            (y, y_error, dense_info, solver_state, solver_result) = super().step(terms, t0, t1, y0, args, solver_state, made_jump)
            x, est = y
            new_est = self.update_aux(t0, x, est)
            return ((x, new_est), y_error, dense_info, solver_state, solver_result)
      
    return WrappedSolver