import jax
import jax.numpy as jnp
from jax import lax

from typing import Optional


def return_lr(lr, step):
    if callable(lr):
        return lr(step)
    else:
        return lr


class Optimizer:
    def __init__(self):
        self.step = 0

    def update(self, grads: Optional[jnp.ndarray] = None):
        self.step += 1

    def __call__(self, params, grads):
        update_val = self.update(grads)
        return params + update_val


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        return -lr * grads


class MomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        self.velocity = self.momentum * self.velocity - lr * grads
        return self.velocity


class NesterovMomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        self.velocity = self.momentum * self.velocity - lr * grads
        return self.momentum * self.velocity - (lr * grads)


class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.g_acc = 0.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        self.g_acc += grads ** 2
        return -lr / (self.epsilon + jnp.sqrt(self.g_acc)) * grads


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8, rho: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho
        self.g_acc = 0.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        self.g_acc = self.rho * self.g_acc + (1 - self.rho) * grads ** 2
        return -lr / (self.epsilon + jnp.sqrt(self.g_acc)) * grads


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0.
        self.v = 0.
        self.beta1_div = 1.
        self.beta2_div = 1.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()
        self.beta1_div *= self.beta1
        self.beta2_div *= self.beta2

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat = self.m / (1 - self.beta1_div)
        v_hat = self.v / (1 - self.beta2_div)
        return -lr / (self.epsilon + jnp.sqrt(v_hat)) * m_hat


class AdaBelief(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-16, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0.
        self.s = 0.
        self.beta1_div = 1.
        self.beta2_div = 1.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()
        self.beta1_div *= self.beta1
        self.beta2_div *= self.beta2

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.s = self.beta2 * self.s + (1 - self.beta2) * (grads - self.m) ** 2 + self.epsilon
        m_hat = self.m / (1 - self.beta1_div)
        s_hat = self.s / (1 - self.beta2_div)
        return -lr * m_hat / (jnp.sqrt(s_hat) + self.epsilon)
