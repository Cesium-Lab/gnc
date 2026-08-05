# Guidance Laws

## Chasing

## RPOD

### Glidescope Guidance
[paper](https://sites.utexas.edu/near/files/2017/04/glideslope_v1.pdf)
- Rendezvous in circular orbit via straight line
  - For approaching ISS

**Optimal guidance**

### Lambert Targeting 3


### CW Impulsive Targeting
[Paper](https://ntrs.nasa.gov/api/citations/19690004917/downloads/19690004917.pdf)
## Landing


### ZEM/ZEV 2


### Convex/lossless convexification (G-FOLD-style) 5



### Apollo E-guidance 4



<!-- 

### Orbital maneuvers & propulsion
- **Hohmann transfer**: two-impulse, minimum-energy transfer between coplanar circular orbits — both Δv's come straight out of vis-viva
- **Bi-elliptic transfer**: beats Hohmann in total Δv for large orbit-radius ratios, at the cost of transfer time
- **Plane change**: Δv = 2v·sin(Δi/2) — expensive; cheapest when done at the lowest-velocity point of the orbit, ideally combined with another maneuver
- **Tsiolkovsky rocket equation**: Δv = Isp·g₀·ln(m₀/m_f) — connects propellant mass fraction to achievable Δv; know how to invert it to solve for required propellant given a Δv budget -->

<!-- 

## RPO
*Rendezvous/proximity ops*
- **Glideslope guidance** — commands closing velocity proportional to range along a fixed approach corridor line; this is literally the algorithm behind Shuttle and ISS final approach. highest priority you don't have yet. This is the actual algorithm behind real docking approaches (Shuttle, ISS, and almost certainly conceptually similar to whatever Starbase runs for final approach), 
- CW-based impulsive targeting — closed-form impulsive rendezvous targeting using the Clohessy-Wiltshire state transition matrix — solve for the velocity impulse that drives relative position to a target after a fixed transfer time. Valid for circular reference orbit + small relative separation.
- Lambert targeting — solve Lambert's problem (find the orbit/impulses connecting two position vectors in a given transfer time), used for the larger-scale approach phase before you're close enough for CW to apply well
- Phasing orbit
- Relative orbital elements
- Q-law
- Station-keeping

## Landing
- ZEM/ZEV guidance - (zero-effort-miss / zero-effort-velocity): compute where you'd end up with no further control input (ZEM) and what velocity error you'd have (ZEV), command acceleration proportional to correcting both. Near-optimal for a wide class of terminal guidance/landing problems, and cheap enough to run onboard in closed form.
- Apollo E-guidance (quadratic guidance) — the actual lunar descent guidance law flown on Apollo, closed-form based on a polynomial acceleration profile
- Convex/lossless convexification (G-FOLD-style) - (e.g., fuel-optimal powered descent): the true fuel-optimal problem is often nonconvex (due to thrust magnitude/pointing constraints), but techniques like *lossless convexification* prove the convex relaxation shares the same optimum under certain conditions — turns it into an SOCP solvable fast enough for real-time onboard use (this is roughly how G-FOLD-style landing guidance works). [SpaceX paper](https://govindchari.com/assets/pdf/gfold_paper.pdf)
- SCP (as a guidance-law tool for landing, when lossless convexification doesn't cleanly apply)


## Reentry?
(lower relevance for your target, listed for completeness)
- Apollo entry guidance — drag-feedback based
- Numerical predictor-corrector entry guidance — modern approach, propagates forward and corrects online

## Ascent
- Powered Explicit Guidance (PEG) — Shuttle-heritage ascent guidance
- Gravity turn guidance

## Path-following / waypoint
(robotics-adjacent, useful if rover/autonomy work comes up)
- Line-of-sight guidance
- Carrot-chasing algorithm

## Chasing
- Pure pursuit — point velocity straight at target, simple but inefficient, large miss against maneuvering targets
- Proportional Navigation (PN) — a = N'·Vc·λ̇, drives LOS rate to zero
- Pure PN vs True PN — variants differ in whether commanded acceleration is applied normal to the LOS or normal to the chaser's velocity vector. classical intercept/rendezvous law — commanded acceleration a = N'·Vc·λ̇ (N' = navigation gain, Vc = closing velocity, λ̇ = line-of-sight rate). Drives LOS rate to zero, which guarantees intercept for constant-velocity targets.

- Augmented PN (APN) — PN plus a term accounting for target acceleration, needed if target isn't constant-velocity


---

**What to actually go over, given your target (rendezvous, docking, terminal-phase sensor handoff, landing):**

1. **Glideslope guidance** — directly matches your stated project focus.
2. **ZEM/ZEV** — already in your notes, but worth deepening since it spans both docking and landing, matches your Blue Origin lunar landing context too.
3. **Lambert targeting** — fills a real gap. CW-based targeting only works for small/local relative motion, Lambert is what handles the larger-scale rendezvous approach phase before CW applies, worth knowing both ends of that handoff.
4. **Apollo E-guidance** — good one for your Blue Origin lunar landing context specifically, historically significant, likely to come up if landing guidance is discussed at all.

Skip for now unless it comes up: PN/APN (more missile/intercept-flavored, less central to docking), entry guidance, ascent guidance, path-following. Not useless, just lower yield given where you're aiming.
 -->
