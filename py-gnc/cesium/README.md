perception folder?

## Ideas

### Launch
- `guidance/ascent.py`, `physics/propulsion.py`, `physics/aerodynamics.py`
- closed-loop guidance, variable mass/staging, atmosphere/drag/max-Q, rotating Earth, steering vs. gravity losses.
- Projects
  - Point mass sim (atmosphere, thrust, stages)
  - Open loop guidance
  - Closed loop guidance (PEG)
  - 6DOF, TVC, attitude

### Nav for mars-courier

Today every burn is planned from the **true** state (`RpoTimeline` plans from the propagated
truth COES; arrival/departure targeting uses exact states) and executed perfectly. Closing the
loop means guidance only ever sees an **estimate**, and burns are executed with errors.

```
truth propagation ──► sensors (noisy meas.) ──► filter (EKF/UKF) ──► x̂, P
      ▲                                                               │
      └──── burn applied with execution error ◄── guidance plans from x̂
```
- Separate truth from knowledge
  - Guidance takes on estimate, sim knows truth
- Measurement models
  - Optical, relative-range
- Filters
- Burn execution errors
- Trajectory correction maneuvers
- Monte Carlos
  - Dispersed state, noise, control error
  - B-plane ellipse, capture orbit dispersion, RPO miss, ∆V statistical budget