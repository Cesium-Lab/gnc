

## To look at 
- P. S. Maybeck, "Stochastic Models, Estimation, and Control," vol. 1,
       Academic Press, 1979.

## References

Cited by short name in docstrings across `cesium/navigation`, `cfsw` and `csim`,
at chapter level

**Primary**
- Carpenter, J. R., & D'Souza, C. N. (eds.). *Navigation Filter Best Practices.* NASA Technical Publication. [NTRS](https://ntrs.nasa.gov/api/citations/20250002787/downloads/20250002787.pdf). Notes by chapter in [navigation-best-practices/](navigation-best-practices/README.md). Cited in docstrings as "NASA Navigation Filter Best Practices, Ch. N".

**Kalman filtering: foundations and implementation**
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering* 82(1), 35-45.
- Bucy, R. S., & Joseph, P. D. (1968). *Filtering for Stochastic Processes with Applications to Guidance.* Interscience. (Origin of the Joseph-form covariance update.)
- Grewal, M. S., & Andrews, A. P. (2015). *Kalman Filtering: Theory and Practice Using MATLAB*, 4th ed. Wiley. (Numerically robust implementation: Joseph form, symmetrization.)
- Bierman, G. J. (1977). *Factorization Methods for Discrete Sequential Estimation.* Academic Press. (UD / square-root filters.)
- Brown, R. G., & Hwang, P. Y. C. (2012). *Introduction to Random Signals and Applied Kalman Filtering*, 4th ed. Wiley. (Discrete process noise from continuous models.)
- Van Loan, C. F. (1978). "Computing Integrals Involving the Matrix Exponential." *IEEE Transactions on Automatic Control* 23(3), 395-404. (Exact discrete Q.)
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Johns Hopkins. (Solve rather than invert.)
- Crassidis, J. L., & Junkins, J. L. (2012). *Optimal Estimation of Dynamic Systems*, 2nd ed. CRC Press. (EKF, ch. 3.)

**Consistency testing and truth-vs-filter models**
- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation.* Wiley. (NEES/NIS consistency tests, ch. 5.)
- Maybeck, P. S. (1979). *Stochastic Models, Estimation, and Control*, Vol. 1. Academic Press. (Truth model vs filter model; performance analysis.)

**Orbit determination and relative motion**
- Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit Determination.* Elsevier. (STM and variational equations, ch. 4.)
- Clohessy, W. H., & Wiltshire, R. S. (1960). "Terminal Guidance System for Satellite Rendezvous." *Journal of the Aerospace Sciences* 27(9), 653-658.

**Attitude estimation (for a future MEKF)**
- Lefferts, E. J., Markley, F. L., & Shuster, M. D. (1982). "Kalman Filtering for Spacecraft Attitude Estimation." *Journal of Guidance, Control, and Dynamics* 5(5), 417-429.
- Markley, F. L., & Crassidis, J. L. (2014). *Fundamentals of Spacecraft Attitude Determination and Control.* Springer. (MEKF, ch. 6.)
