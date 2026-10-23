

**Sim**
- Potential bug in sim with frames that doesn't crash but has some other bad effect later down the line
  - IMU having an offset in the sim, but not in the algorithm. spinning would make the spacecraft see a translation due to being in the rotating frame of the spacecraft
  - see a force pointed inwards at a given moment if spinning, and a force along direction if speeding up rotation too
1. accel misalignment, wrong DCM
2. specific gravity changes under gravity with a wrong orientation
3. EKF omits or has the wrong lever arm
4. Gyro bias/error
5. SIGN FLIP
6. Wrong measurement time

**ekf**
- Large residual in sensor mesurement
  - Normalized innovation squared NIS = y_bar.T * inv(S) * y_bar where S = H.T@P@H + R
  - Bad sensor, overconfident (small) P, wrong measurement matrix, smaller than actual R
  - R inflation
  - COULD DO
    - skip some outliers
    - covariance matching for many
    - Remove
    - NIS high on ALL sensors meant maybe model bug

**reaction wheels**
- allocation matrix: wheel torque to body torque
- desat with external torque
- null space can be used to reallocate among different wheels
  - if a wheel saturates then clamp column
