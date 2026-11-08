# Mars mission


## Outline

### Guidance Planning

1. Mars launch window - optimal trajectory there
   1. Use porkchop plot for launch window and interplanetary characteristics
   2. Get Mars and Earth ephemeris
   3. Use *Lambert* and get an optimal trajectory dV probably
2. Departure - *parking orbit* and *escape burn*
   1. Pick times such that we can launch into an interplanetary orbit (ECEF <--> ECI)
      1. On homework where we choose launch date
   2. Maybe model the rocket launch
   3. (Find a paper that approximates rocket flight profile and deltaV required or something)
   4. Two launch times per day to get into an orbital plane that we want
3. Interplanetary
   1. Navigation or comms budget or something
   2. Simple two-body
      1. Synodic check
   3. Inject departure error? (angle, like in lecture)
4. Arrival 
   1. B-plane targeting to get into orbit of satellite
   2. Planetary limb?
   3. Rendezvous with Hill's equations
5. Landing
   1. Reentry with atmospheric targeting
   2. Project from Lunar Lander

### Order of Implementation