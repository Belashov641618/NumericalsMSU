from __future__ import annotations

import matplotlib.pyplot
from belashovplot import TiledPlot

import numpy
from typing import Callable, Iterable

class EulerI(Iterable):
    _steps:int
    _dt:float
    _function:Callable[[numpy.ndarray, float], numpy.ndarray]
    _values:numpy.ndarray
    _filled:numpy.ndarray
    def __init__(self, t0:float, N:int, function:Callable[[numpy.ndarray, float], numpy.ndarray], u0:numpy.ndarray):
        self._dt = t0 / N
        self._steps = N
        self._function = function
        self._values = numpy.zeros((N, len(u0)))
        self._values[0] = u0
        self._generated = False
        self._filled = numpy.zeros(N, dtype=bool)
        self._filled[0] = True
    _counter:int
    def __iter__(self) -> EulerI:
        self._counter = 0
        return self
    def _sub_next(self):
        self._counter += 1
        if self._counter >= self._steps:
            self._generated = True
            raise StopIteration
        if not self._filled[self._counter]:
            self._step()
            self._filled[self._counter] = True
    def _step(self):
        self._values[self._counter] = self._values[self._counter-1] + self._dt * self._function(self._values[self._counter-1], self._counter*self._dt)
    def __next__(self) -> numpy.ndarray:
        self._sub_next()
        return self._values[self._counter]
    _generated:bool
    def array(self):
        if not self._generated:
            for v in self:
                pass
        return self._values.swapaxes(0,1)

class EulerII(EulerI):
    def __init__(self, t0:float, N:int, function:Callable[[numpy.ndarray, float], numpy.ndarray], u0:numpy.ndarray, u1:numpy.ndarray):
        super().__init__(t0, N, function, u0)
        self._values[1] = u1
    def __iter__(self):
        self._counter = 1
        return self
    def _step(self):
        self._values[self._counter] = self._values[self._counter-2] + 2*self._dt * self._function(self._values[self._counter-1], self._counter*self._dt)

class Differential:
    _function:Callable[[numpy.ndarray, float], float]
    _order:int
    _initials:numpy.ndarray
    def __init__(self, function:Callable[[numpy.ndarray, float], float], initials:numpy.ndarray):
        self._function = function
        self._order = len(initials)
        self._initials = initials
    def _euler_function(self, u:numpy.ndarray, t:float) -> numpy.ndarray:
        temp = numpy.roll(u, -1)
        temp[-1] = self._function(u, t)
        return temp

    def solveI(self, t0:float, N:int):
        solver_ = EulerI(t0, N, self._euler_function, self._initials)
        class IteratorSolutionI(Iterable):
            solver : EulerI
            def __init__(self, solver:EulerI):
                self.solver = solver
            def __iter__(self):
                iter(self.solver)
                return self
            def __next__(self):
                return next(self.solver)
            def array(self):
                return self.solver.array()
            def result(self):
                return self.solver.array()[0]
        return IteratorSolutionI(solver_)
    def solveII(self, t0:float, N:int, u1:numpy.ndarray=None):
        if u1 is None:
            u1 = next(iter(self.solveI(t0, N)))
        if isinstance(u1, int):
            for i, values in enumerate(self.solveI(t0, u1*N)):
                if i + 1 == u1:
                    u1 = values
                    break
        solver_ = EulerII(t0, N, self._euler_function, self._initials, u1)
        class IteratorSolutionII(Iterable):
            solver: EulerII
            def __init__(self, solver: EulerII):
                self.solver = solver
            def __iter__(self):
                iter(self.solver)
                return self
            def __next__(self):
                value = next(self.solver)
                return value
            def array(self):
                return self.solver.array()[:]
            def result(self):
                return self.solver.array()[0]
        return IteratorSolutionII(solver_)

def test_euler():
    t0 = 20.0
    N = int(t0)
    t_array = numpy.linspace(0, t0, N)
    u_array = 3.13 * numpy.cos(t_array)

    H = 1000
    t_high_resolution = numpy.linspace(0, t0, H)
    def solution(time_array_:numpy.ndarray=H):
        if isinstance(time_array_, int):
            time_array_ = numpy.linspace(0, t0, time_array_)
        return 3.13 * numpy.cos(time_array_)
    u_high_resolution = solution()

    initials = numpy.array([3.13, 0])

    def function(u:numpy.ndarray, t: float):
        return -u[0]

    system = Differential(function, initials)
    solution_methods = {
        "с перешагиванием (вторая точка из решения)"        : lambda M: system.solveII(t0, M, numpy.array([3.13*numpy.cos(t0/M), -3.13*numpy.sin(t0/M)])),
        "с перешагиванием (вторая точка из Эйлера)"         : lambda M: system.solveII(t0, M, u1=1),
        "с перешагиванием (вторая точка из двойного Эйлера)": lambda M: system.solveII(t0, M, u1=2),
    }
    points_amount = {
        "на границе устойчивости"               : N,
        "в два раза меньше границы устойчивости": N * 2,
        "в пять раз меньше границы устойчивости": N * 5,
    }

    plot = TiledPlot(10*21/9, 10)
    plot.FontLibrary.MultiplyFontSize(0.5)
    plot.description.bottom("t -->")
    plot.description.left("u -->")

    row_:int
    col_: int
    for row_, (solution_description, solution_call) in enumerate(solution_methods.items()):
        plot.description.row.left(f'Тип решения: {solution_description}', 2*row_, 2*row_+1)
        plot.description.row.right("Решение", 2*row_)
        plot.description.row.right("Отклонение", 2*row_+1)
    for col_, (points_description, points_amount) in enumerate(points_amount.items()):
        time_array = numpy.linspace(0, t0, points_amount)
        u_real = solution(time_array)
        col0 = 2 * col_
        col1 = 2 * col_ + 1

        plot.description.column.bottom(f"Количество точек: {points_amount}", col0, col1)
        for row_, (solution_description, solution_call) in enumerate(solution_methods.items()):

            row0 = 2 * row_
            row1 = 2 * row_ + 1
            u_calculated = solution_call(points_amount).result()

            axes = plot.axes.add((col0, row0), (col1, row0))
            axes.grid(True)
            axes.plot(t_high_resolution, u_high_resolution, linestyle='-', color='maroon')
            axes.scatter(time_array, u_calculated, marker='.', c='green')
            axes.plot(time_array, u_calculated, linestyle='--', color='yellow')
            # plot.graph.description(f"Количество точек: {points_amount} ({points_description})\nТип схемы Эйлера: {solution_description}")

            deviations = numpy.abs(u_real - u_calculated)

            axes = plot.axes.add((col0, row1), (col1, row1))
            axes.grid(True)
            axes.plot(time_array, deviations, linestyle='--', color='blue')

    plot.finalize()
    plot._Figure.savefig('temp.svg')
    plot.show()
