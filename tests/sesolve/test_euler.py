import pytest

from dynamiqs_adaptative.gradient import Autograd, CheckpointAutograd
from dynamiqs_adaptative.solver import Euler

from ..solver_tester import SolverTester
from .closed_system import cavity, tdqubit


class TestSEEuler(SolverTester):
    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(system, solver, esave_atol=1e-3)

    @pytest.mark.parametrize('system', [cavity, tdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        solver = Euler(dt=1e-4)
        self._test_gradient(system, solver, gradient, rtol=1e-2, atol=1e-2)
