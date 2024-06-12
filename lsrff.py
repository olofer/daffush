"""
Least-squares model fitting with Random Fourier Features.
Batches are added one-by-one, and the model is updated via the .fit() method.
The .eval(.) method gives predictions. 
Supports multiple outputs.
"""

import numpy as np


class LSRFF:
    def __init__(self, units: int, inputs: int, outputs: int, sigma: np.array):
        assert len(sigma.shape) == 1, "sigma should be a 1D array"
        assert len(sigma) == inputs, "sigma has incorrect length"
        self.inputs = inputs
        self.outputs = outputs
        self.units = units
        self.sigma = np.copy(sigma)
        assert self.units % 2 == 0 and self.units > 0, "even number of units required"
        self.omega = np.row_stack(
            [
                np.random.randn(self.units // 2) * self.sigma[k]
                for k in range(self.inputs)
            ]
        )
        self.gamma = 1.0e-8
        self.samples = int(0)
        self.coefs = np.zeros((self.units, self.outputs))
        self.HTH = np.zeros((self.units, self.units))
        self.HTy = np.zeros((self.units, self.outputs))

    def featurize(self, X: np.array) -> np.array:
        phase = X @ self.omega
        return np.concatenate([np.cos(phase), np.sin(phase)], axis=1) * (
            2 / np.sqrt(self.units)
        )

    def add_batch(self, X: np.array, y: np.array) -> None:
        assert len(X.shape) == 2 and len(y.shape) == 2
        B = X.shape[0]
        assert B == y.shape[0]
        assert X.shape[1] == self.inputs
        assert y.shape[1] == self.outputs

        H = self.featurize(X)
        total = self.samples + B
        aa = self.samples / total
        bb = B / total

        self.HTH = aa * self.HTH + bb * (H.T @ H)
        self.HTy = aa * self.HTy + bb * (H.T @ y)
        self.samples = total

    def fit(self, gamma_per_sample=False) -> None:
        if gamma_per_sample:
            self.coefs = np.linalg.solve(
                self.HTH + (self.gamma / (1 + self.samples)) * np.eye(self.units),
                self.HTy,
            )
        else:
            self.coefs = np.linalg.solve(
                self.HTH + self.gamma * np.eye(self.units), self.HTy
            )

    def eval(self, X: np.array) -> np.array:
        return self.featurize(X) @ self.coefs

    def __len__(self):
        return self.samples

    def __str__(self):
        return "LSRFF object (id=%i) with %i units; i/o=%i/%i; seen %i samples" % (
            (id(self), self.units, self.inputs, self.outputs, self.samples)
        )

    def set_regularization(self, gamma_: float) -> None:
        self.gamma = gamma_


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=250)
    parser.add_argument("--inputs", type=int, default=2)
    parser.add_argument("--outputs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    Model = LSRFF(args.units, args.inputs, args.outputs, np.ones(args.inputs))

    HX = Model.featurize(np.random.randn(args.batch_size, args.inputs))
    print(HX.shape)

    print(len(Model))

    Model.add_batch(
        np.random.randn(args.batch_size, args.inputs),
        np.random.randn(args.batch_size, args.outputs),
    )

    print(len(Model))
    Model.fit()

    Y = Model.eval(np.random.randn(args.batch_size, args.inputs))
    print(Y.shape)

    print("done.")
    