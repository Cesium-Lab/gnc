# gnc

A mini-capstone exploring navigation, spacecraft mission design, orbital mechanics, and other GNC-related topics. I have simulation tools and Jupyter notebooks that connect the math to mission analysis and visualization.

## Featured projects

- **[Voyager trajectory analysis](py-gnc/playgrounds/orbital/voyager.ipynb)** — Models Voyager 1 and 2 interplanetary transfers using Lambert solutions, numerical propagation, and planetary flyby geometry. Includes comparisons with JPL Horizons ephemerides and visualizations of the trajectories and gravity assists.
- **[Mars Courier mission design](projects/mars-courier)** — A Earth-to-Mars mission plan linking launch-window selection, Earth escape, ground-station coverage, with Mars arrival as the next phase. Built around reusable trajectory, coordinate-transformation, and plotting tools.
  - [Interplanetary transfer](projects/mars-courier/interplanetary.ipynb): porkchop plots, launch-window constraints, and transfer selection.
  - [Earth departure](projects/mars-courier/departure.ipynb): hyperbolic escape geometry, propagation, and trajectory corrections.
  - [Ground segment](projects/mars-courier/ground_segment.ipynb): ground tracks, sky plots, and visibility windows across a station network.
  - Planned extensions: Mars arrival and capture, orbital rendezvous, and descent.

## Code and validation

[Cesium](py-gnc/cesium) contains reusable orbital mechanics, guidance, navigation, and coordinate-transformation code. [Tests](py-gnc/tests) include textbook reference cases, frame-transformation round trips, and geometric consistency checks. Analyses use NumPy, SciPy, Matplotlib, and Plotly.

## Submodules
Initialize if a project is a submodule
```sh
git submodule update --init --recursive
```

<!-- 1. Estimation
   1. Filtering
      1. U-D filter
      2. Square root filter
      3. MEKF vs. EKF
   2. IMU Datasets
      1. Oxford Inertial Odometry Dataset

2. [Guidance](https://en.wikipedia.org/wiki/Guidance_system) (assuming perfect estimation)
   1. Proportional navigation
   2. Path planning
   3. Q-system (not sure how relevant this is but it is on the Wiki page)

3. Control (follow guidance assuming perfect estimation)

4. Sims (All of GNC)
   1. Might integrate some parsers
   2. Stationkeeping mini (large) project
   3. lunar lander ooo
5. Extras
   1. NMEA Parser
   2. TLE Parser
   3. Cart pendulum system but showing natural frequency too

## Data flow-ish
Measurements (from sim) -> Navigation -> Guidance -> Control -> Actuator (to sim) ->


 -->
<!-- tree -I '__pycache__|*.pyc|.git|.venv' -->
