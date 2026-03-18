# gnc-playground
Part 1 of the journey


1. Estimation
2. Guidance (path planning assuming perfect estimation)
3. Control (follow guidance assuming perfect estimation)
4. Sims
5. System (All of GNC)
   1. Might integrate some parsers
   2. Stationkeeping mini (large) project
   3. lunar lander ooo
6. Extras
   1. NMEA Parser
   2. TLE Parser
   3. Cart pendulum system but showing natural frequency too

## Data flow-ish
Measurements (from sim) --> Navigation --> Guidance --> Control --> Actuator (to sim)
