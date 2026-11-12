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
Each stage writes `trajectories/<stage>-<mission>.mission` (NumPy NPZ), where `<mission>`
uses underscores, e.g. `departure-birthday_christmas_earlier.mission`. Naming and I/O live
in `mission_io.py` (`mission_path`, `save_mission`, `load_mission`).

1. `interplanetary.ipynb`
   1. Saves `interplanetary-<mission>`: Lambert epochs and HCI Earth/Mars/spacecraft states [km, km/s]
2. `departure.ipynb`
   1. Loads `interplanetary-<mission>` and models the Earth escape trajectory
   2. Saves `departure-<mission>`: burn-reference epoch [JD], ECI states [m, m/s], escape and midcourse Δv
3. `ground_segment.ipynb`
   1. Loads `departure-<mission>`
   2. Computes ECEF states, ground tracks, sky plots, and per-station/network visibility
4. `arrival.ipynb`
   1. Loads `interplanetary-<mission>`, B-plane targets the direct-injection capture
   2. Saves `arrival-<mission>`: capture epoch, approach hyperbola, post-capture MCI state, target orbit, capture Δv
5. `rpo.ipynb`
   1. Loads `arrival-<mission>`: starts at the capture epoch from the post-capture state
   2. Runs nominal and dispersed cases; saves `rpo-<mission>`: timelines, burns, Δv by phase
6. `summary.ipynb`
   1. Loads all four stages: mission timeline and full Δv budget
7. `descent.ipynb`
   1. Descending to landing site

## Approximations
- TDB ≈ TT