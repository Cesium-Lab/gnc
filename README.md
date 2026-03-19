# gnc-playground
Part 1 of the journey


1. Estimation
   1. Filtering
   2. IMU Datasets
      1. Oxford Inertial Odometry Dataset

2. [Guidance](https://en.wikipedia.org/wiki/Guidance_system) (assuming perfect estimation)
   1. Proportional navigation
   2. Path planning
   3. Q-system (not sure how relevant this is but it is on the Wiki page)

3. Control (follow guidance assuming perfect estimation)

5. Sims (All of GNC)
   1. Might integrate some parsers
   2. Stationkeeping mini (large) project
   3. lunar lander ooo
6. Extras
   1. NMEA Parser
   2. TLE Parser
   3. Cart pendulum system but showing natural frequency too

## Data flow-ish
Measurements (from sim) --> Navigation --> Guidance --> Control --> Actuator (to sim)
