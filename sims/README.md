# Sims

Copying my stuff from the trajectorylib and dronesim projects


Simulator
- Cool part is having different simulators for different environments (rocket/drone, orbits)




# Planning
Outline of sim modules

## Utility
- `RK4`
- `RK45`

## Calculations
Generic, and not attached to any object

- `integration`
  - `RK4`
  - `RK45`
- `states`
  - `coes2rv`
  - `rv2coes`
- `perturbations`
  - `gravity`
  - `J2`
  - `J3`
  - `Drag`
  - `SRP`
- `frames`
  - ECI
    - Earth centered, intertial
  - ECEF
    - Earth centered, rotates with Earth
  - Perifocal
    - Earth centered, orthogonal to orbit
  - SCI
    - SC centered, inertial
  - SCF
    - SC centered, rotates with SC
  - SPF
    - Surface point centered, rotates with Earth
    - Points up and North
  - NED
    - Surface point centered, rotates with Earth
  - Sensor (multiple)

<!-- # Simulation -->


## Requirements
Things to keep in mind when designing

### General Goals
- Model and design orbits over time
  - With Perturbations (aero, solar, J2, etc.)
  <!-- - Minimal interface (optional) -->
- Design/determine orbit based on parameters
  - Basically orbital determination chapter from textbook
- Write down most/all equations from textbook (education)
  - For practical purposes

### Specific Goals
- Obtain orbitkeeping requirements
  - Total torque/thrust for
    - SRP
    - Drag
    - Pointing (sun, down, etc.)
    - Thrusting
  - Reaction wheel torque, thruster thrust, etc.
- Inform ground station tracking/design
  - Elevation/azimuth
    - elevation = f(azimuth) 
  - Ground track over real map



