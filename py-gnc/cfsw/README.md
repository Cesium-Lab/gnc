# Flight software

Onboard logic (navigation filters, guidance modes, sequencing). Stateless math lives
in `cesium`; truth, sensors and actuators live in `csim`.

```
cfsw -> cesium
csim -> cesium, cfsw   (runs the flight software in the loop)
```

`FlightComputer.step(t, z)` is the only interface.

## TODO:
- [ ] Guidance: return commands from `step`
- [ ] Nonlinear propagation in `_propagate`
- [ ] Range/az/el measurements
