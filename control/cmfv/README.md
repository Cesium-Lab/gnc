# CMFV


## Design Process / Order
### `intro.ipynb`
- Observer / Pressure Transducer (Module)
  - Easiest one because it was just noise, but helped me figure out how I wanted to modularize things with a setup and "run" functions
- Valve integrator (Module)
  - Since the controller realistically will output some torque-related electrical command, it made sense for the output to be as a "theta_dot"
  - Therefore, I made this block second so I could figure out how to do a "stateful" block (i will need to watch for windup though)
- Basic valve (Module)
  - Starting with simple (p1, v1) --> p2 commands
  - Will be more complicated later, but definitely just want something that works now
- Test that pressure lowers as valve closes
  - With all 3 modules above
  - Sanity check
- Proportional Controller! (Module)
  - Super simple
- More tests
  - Controller respond to step setpoints?
- Function-izing
  - inputs, outputs, initial conditions
  - running the sim
  - plotting the sim results

### controllers.ipynb



## Useful book stuff
All in the `images/` folder

Kv = 0.865*Cv
Cv = 1.16*Kv