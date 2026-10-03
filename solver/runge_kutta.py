from   typing import Callable
import torch


__all__: list[str] = [
    'RK1_Euler',
    'one_step_RK1_Euler',
    'RK2_Heun',
    'one_step_RK2_Heun',
    'RK3_Heun',
    'one_step_RK3_Heun',
    'RK4_classic',
    'one_step_RK4_classic',
]


##################################################
def RK1_Euler(
        t_curr:     float,
        y_curr:     torch.Tensor,
        delta_t:    float,
        derivative: Callable[[float, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
    """## Forward Euler single-step integrator

## Description
Performs a single step of the first-order explicit Euler method to integrate an ordinary differential equation.

## Arguments
`t_curr` (`float`): Current time coordinate.
`y_curr` (`torch.Tensor`): Current state tensor.
`delta_t` (`float`): Time step size.
`derivative` (`Callable[[float, torch.Tensor], torch.Tensor]`): Callable computing the time derivative given time and state.

## Returns
`torch.Tensor`: Updated state tensor at time `t_curr + delta_t`.
"""
    k1: torch.Tensor = derivative(t_curr, y_curr)
    return y_curr + delta_t * k1


one_step_RK1_Euler: Callable[[float, torch.Tensor, float, Callable[[float, torch.Tensor], torch.Tensor]], torch.Tensor] = RK1_Euler


def RK2_Heun(
        t_curr:     float,
        y_curr:     torch.Tensor,
        delta_t:    float,
        derivative: Callable[[float, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
    """## Heun's second-order Runge-Kutta integrator

## Description
Performs a single step of Heun's explicit second-order Runge-Kutta method.

## Arguments
`t_curr` (`float`): Current time coordinate.
`y_curr` (`torch.Tensor`): Current state tensor.
`delta_t` (`float`): Time step size.
`derivative` (`Callable[[float, torch.Tensor], torch.Tensor]`): Callable computing the time derivative given time and state.

## Returns
`torch.Tensor`: Updated state tensor at time `t_curr + delta_t`.
"""
    k1: torch.Tensor = derivative(t_curr, y_curr)
    k2: torch.Tensor = derivative(t_curr + delta_t, y_curr + delta_t * k1)
    return y_curr + delta_t * (k1 + k2) / 2.0


one_step_RK2_Heun: Callable[[float, torch.Tensor, float, Callable[[float, torch.Tensor], torch.Tensor]], torch.Tensor] = RK2_Heun


def RK3_Heun(
        t_curr:     float,
        y_curr:     torch.Tensor,
        delta_t:    float,
        derivative: Callable[[float, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
    """## Heun's third-order Runge-Kutta integrator

## Description
Performs a single step of Heun's explicit third-order Runge-Kutta method.

## Arguments
`t_curr` (`float`): Current time coordinate.
`y_curr` (`torch.Tensor`): Current state tensor.
`delta_t` (`float`): Time step size.
`derivative` (`Callable[[float, torch.Tensor], torch.Tensor]`): Callable computing the time derivative given time and state.

## Returns
`torch.Tensor`: Updated state tensor at time `t_curr + delta_t`.
"""
    k1: torch.Tensor = derivative(t_curr, y_curr)
    k2: torch.Tensor = derivative(t_curr + (1.0 / 3.0) * delta_t, y_curr + (1.0 / 3.0) * delta_t * k1)
    k3: torch.Tensor = derivative(t_curr + (2.0 / 3.0) * delta_t, y_curr + (2.0 / 3.0) * delta_t * k2)
    return y_curr + delta_t * (k1 + 3.0 * k3) / 4.0


one_step_RK3_Heun: Callable[[float, torch.Tensor, float, Callable[[float, torch.Tensor], torch.Tensor]], torch.Tensor] = RK3_Heun


def RK4_classic(
        t_curr:     float,
        y_curr:     torch.Tensor,
        delta_t:    float,
        derivative: Callable[[float, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
    """## Classical fourth-order Runge-Kutta integrator

## Description
Performs a single step of the classical fourth-order Runge-Kutta (RK4) method to numerically advance an initial value problem.

## Arguments
`t_curr` (`float`): Current time coordinate.
`y_curr` (`torch.Tensor`): Current state tensor.
`delta_t` (`float`): Time step size.
`derivative` (`Callable[[float, torch.Tensor], torch.Tensor]`): Callable computing the time derivative given time and state.

## Returns
`torch.Tensor`: Updated state tensor at time `t_curr + delta_t`.
"""
    k1: torch.Tensor = derivative(t_curr, y_curr)
    k2: torch.Tensor = derivative(t_curr + 0.5 * delta_t, y_curr + 0.5 * delta_t * k1)
    k3: torch.Tensor = derivative(t_curr + 0.5 * delta_t, y_curr + 0.5 * delta_t * k2)
    k4: torch.Tensor = derivative(t_curr + delta_t, y_curr + delta_t * k3)
    return y_curr + delta_t * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


one_step_RK4_classic: Callable[[float, torch.Tensor, float, Callable[[float, torch.Tensor], torch.Tensor]], torch.Tensor] = RK4_classic
