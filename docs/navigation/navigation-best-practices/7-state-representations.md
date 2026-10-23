# State Representations
- Not good practice to include unobservable states
- In early stages, may not be clear what to track
- Best Practice: Start with more states then subtract

## Units and Precision
- SI units preferred usually
- Double precision preferred
- Time in quadruple precision if available
- COULD do km for cislunar space

## Coordinate and Time Systems
- Most use inertial coordinate frame
  - e.g. ICRF/J2000
- Convenient for center of gravitational body
- COULD do body-fixed (ITRF for GNSS and ground tracking systems)
  - Early GPS used this
- True-of-date inertial frame which is just body-fixed at a moment in time, then only polar rotation
  - e.g. rough ECI on ecliptic vs equatorial
- UTC bad (discontinuous)

## Orbit Parameterization
- Orbit elements are appealing
- Long-term error covariance naturally occurs in this
- Cartesian offers computational efficiency for hi-fi models
  - Free of singularities

## Relative State Representation
