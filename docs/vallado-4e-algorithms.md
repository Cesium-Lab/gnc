# Vallado 4e Algorithms

Algorithm index for David A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Edition.
Pages are where each algorithm box appears. "cesium" lists the implementation in `py-gnc/cesium`, if any.

Cite in docstrings as: `Vallado 4e, Algorithm N (NAME), p. X.`

| # | Name | Page | cesium |
|---|---|---|---|
| 1 | Find c2 c3 | 71 | |
| 2 | KepEqtnE | 65 | `orbital.coes.kepler_eq` |
| 3 | KepEqtnP | 69 | `orbital.coes.kepler_eq_p` |
| 4 | KepEqtnH | 71 | `orbital.coes.kepler_eq_h` |
| 5 | ν to Anomaly | 85 | |
| 6 | Anomaly to ν | 85 | |
| 7 | KeplerCOE | 81 | |
| 8 | Kepler | 93 | |
| 9 | RV2COE | 113 | `orbital.coes.rv_to_coes` |
| 10 | COE2RV | 118 | `orbital.coes.coes_to_rv` |
| 11 | FindTOF | 126 | |
| 12 | ECEF TO LatLon | 172 | |
| 13 | ECEF TO LatLon | 173 | `orbital.transformations.ECEF_to_geodetic_LLA` |
| 14 | Julian Date | 183 | `math.time.greg_to_jd` |
| 15 | LSTime | 188 | |
| 16 | ConvTime | 195 | |
| 17 | DMSToRad | 197 | `math.time.dms_to_rad` |
| 18 | RadToDMS | 197 | `math.time.rad_to_dms` |
| 19 | HMSToRad | 198 | |
| 20 | RadToHMS | 198 | |
| 21 | TimeToHMS | 199 | |
| 22 | JDToGregorianDate | 202 | `math.time.jd_to_greg` |
| 23 | IAU-2000, CIO-Based | 219 | `orbital.transformations.itrf_to_gcrs[_matrix]`, `gcrs_to_itrf` |
| 24 | IAU-76/FK5 REDUCTION | 228 | |
| 25 | Geocentric Radec | 259 | `orbital.transformations.cartesian_to_radec` |
| 26 | Topocentric | 260 | |
| 27 | RAZEL | 265 | `orbital.transformations.geodetic_ECEF_to_SEZ` |
| 28 | AzElToRaDec | 268 | `orbital.transformations.azel_to_radec` |
| 29 | Sun | 279 | |
| 30 | SunriseSet | 283 | |
| 31 | Moon | 288 | |
| 32 | MoonRiseSet | 290 | |
| 33 | PlanetRV | 296 | |
| 34 | Shadow | 301 | |
| 35 | SIGHT | 308 | |
| 36 | Hohmann Transfer | 325 | `guidance.coplanar.guid_hohmann` |
| 37 | Bi-elliptic Transfer | 326 | `guidance.coplanar.guid_bielliptic` |
| 38 | One-Tangent Burn | 333 | `guidance.coplanar.guid_one_tangent` |
| 39 | Inclination Only | 344 | `guidance.noncoplanar.guid_i_change` |
| 40 | Change in the Ascending Node—Circular | 347 | `guidance.noncoplanar.guid_raan_change_circular` |
| 41 | Combined Changes to i and Ω—Circular | 334 | |
| 42 | Minimum Combined Plane Change | 353 | |
| 43 | Fixed-Δv Maneuvers | 356 | |
| 44 | Circular Coplanar Phasing (Same Orbits) | 362 | |
| 45 | Circular Coplanar Phasing (Different Orbits) | 363 | |
| 46 | Noncoplanar Phasing | 368 | |
| 47 | Low Thrust Transfer | 387 | `guidance.low_thrust` (commented out) |
| 48 | Hill's Equations | 396 | `orbital.hcw.rv_hcw_3d`, `rv_hcw_2d` |
| 49 | HillEQCM to ECI | 414 | `orbital.hcw.hcw_rsw_to_pci` (simplified) |
| 50 | ECI to HillEQCM | 415 | `orbital.hcw.hcw_pci_to_rsw` (simplified) |
| 51 | Site-Track | 430 | |
| 52 | Angles-Only Gauss | 442 | |
| 53 | Angles-Double-r | 444 | |
| 54 | Gibbs | 460 | |
| 55 | Herrick-Gibbs | 466 | |
| 56 | Lambert's Problem—Minimum Energy | 475 | |
| 57 | Lambert—Gauss's Solution | 478 | |
| 58 | Lambert—Universal Variables | 492 | |
| 59 | Lambert—Battin Method | 494 | |
| 60 | Hit Earth | 503 | |
| 61 | Target | 503 | |
| 62 | Encke | 525 | |
| 63 | AP2Kp Conversion (Kp ↔ ap) | 558 | |
| 64 | Numerical Integration | 591 | |
| 65 | PKepler | 691 | `orbital.coes.pkepler` |
| 66 | Nominal State | 766 | |
| 67 | Differential Correction | 768 | |
| 68 | Kalman Filter—Linear System | 785 | |
| 69 | Linearized Kalman Filter | 791 | |
| 70 | Extended Kalman Filter | 793 | |
| 71 | Repeat Ground Track | 873 | |
| 72 | Maintain Repeat Groundtrack | 879 | |
| 73 | Minimum Altitude Variation | 885 | |
| 74 | Predict | 911 | |
| 75 | Rise/Set | 916 | |
| 76 | Close Approach | 935 | |
| 77 | Patched Conic | 953 | |
| 78 | B-Plane I | 962 | |
| 79 | B-Plane II | 963 | |
