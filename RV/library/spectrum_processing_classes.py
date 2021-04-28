import numpy as np
import scipy.linalg as lg


class DesignMatrix:
    def __init__(self, standard_vals, span):
        self.vals = standard_vals
        self.span = span
        self.mat = self.map()
        self.n = self.vals.size
        self.m = self.span

    def map(self):
        # Matrix is shape (m, n-m)
        n = self.vals.size
        m = self.span

        if np.mod(m, 2) != 1.0:
            raise ValueError('Design Matrix span must be odd.')
        if np.mod(n, 2) != 0.0:
            raise ValueError('Number of values must be even.')

        mat = np.zeros(shape=(m, n-m))
        for i in range(0, m):       # TODO: Ask Karsten if IDL code includes last part of range, if it stops at m or m-1
            mat[i, :] = self.vals[m-i:n-i]
        return mat


class SingularValueDecomposition:
    # noinspection PyTupleAssignmentBalance
    def __init__(self, standard_vals, span):
        self.design_matrix = DesignMatrix(standard_vals, span)
        self.U, self.w, self.V = lg.svd(self.design_matrix, compute_uv=True, full_matrices=False)


class BroadeningFunction:
    def __init__(self, program_vals, standard_vals, span):
        self.design_matrix = DesignMatrix(standard_vals, span)
        self.spectra = self.truncate(program_vals, self.design_matrix)

    @staticmethod
    def truncate(spectra, design_matrix):
        n, m = design_matrix.n, design_matrix.m
        return spectra[m/2:n-m/2, :]

    def solve(self):
        solutions = np.empty(shape=(self.design_matrix.m, self.design_matrix.m, self.spectra[0, :].size))
        for i in range(0, self.design_matrix.m):
            solutions[:, i] = lg.solve(self.design_matrix, )
        return
