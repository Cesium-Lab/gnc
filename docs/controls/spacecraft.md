- **Lyapunov stability**: find a positive-definite "energy-like" function V(x) that decreases monotonically along trajectories (V̇ < 0) — proves stability without solving the dynamics explicitly. LaSalle's invariance principle extends this when V̇ is only negative *semi*-definite.
- **Attitude control**: quaternion-based PD control is the standard building block — u = −Kp·q_error(vector part) − Kd·ω, where q_error is the relative rotation from current to desired attitude. Watch for the quaternion sign ambiguity (q and −q represent the same rotation, need to pick the "short way" rotation).
- **Momentum management/desaturation**: reaction wheels saturate over time from persistent disturbance torques, requiring periodic desaturation via thrusters or magnetorquers — a good practical follow-up if attitude control comes up.
- Discretization: zero-order hold, matrix exponential (Van Loan's method) — converts continuous A, B to discrete Ad, Bd. Comes up any time you're implementing a continuous-time design digitally.



# Reaction wheels
- Allocation matrix
  - $\Tau_{wheels}$ --> $\Tau_{body}$
  - invert matrix (or pseudo inverse) to allocate
- clamp wheel if can't reach torque limit
- Null space: $\tau_{w} = A^+ \tau_b\ +\ (I - A^* A)z$
- Keep wheel near limit
- W matrix for weighting (large is more "costly")