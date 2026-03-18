# gnc-playground
Where I put my non-flight stuff and experiments

1. Estimation
   1. Filtering
2. Guidance (path planning assuming perfect estimation)
3. Control (follow guidance assuming perfect estimation)
4. System (All of GNC)
   1. Might integrate some parsers
5. Embedded
   1. Random stuff here
6. Extras
   1. NMEA Parser
   2. TLE Parser
   3. 
   <!-- 3. Cart pendulum system but showing natural frequency too -->

## Data flow-ish
Measurements (from sim) --> Navigation --> Guidance --> Control --> Actuator (to sim)
