# Bias Modeling

## Zero-input Bias State Models
- Simplest measurement error is just noise
  - Constant or integrals of constants
  - Outputs of zero-input system

### Random constant
- Deterministic over all time, or random ones, but over time interval we want (like power cycling)
- Can be estimated via a filter if observable

### Random ramp
- Randomness is in initial conditions
- If unmeasured, bias variance (due to initial uncertainty) grows unbounded
- Repeated measurements, filter learns fixed unknown parameters, variance converges to 0
- No new process noise. 

### Higher order Derivatives of random constants
- Not often used

## Single-input Bias State Models

### Random Walk
- Inputs are random impulses
- Continuous but not differentiable
- Mean is 0 but has some variance
- Keeps adding process noise but state is only $b$
- Rapid uncertainty with no measurements ($t^2$)

### Random Run
- Still continuous but not differentiable
- Process noise has 2 components for drift $b$ and integral $\dot{b}$ (or $d$)
- Covariance of bias has constant noise that cannot converge now
- Rapid uncertainty with no measurements ($t^3$)

### Higher Order Derivatives of random walks
- Not often used
- "Random zoom?"

### First-order Gauss-Markov (FOGM)
- Bias changes slowly
- Recursive form has exponential in it for bias and covariance
- Approaches finite ss value over time
  - But reaches random walk as it gets close
- BEST PRACTICE for bias modeling

### Derivatives of Gauss-Markov
- Has damping coefficient as part of 2nd order
  - Smoother, memory of bias AND rate (momentum)
- BEST PRACTICE for bias drift modeling

### Vasicek Model
- Data outage will still have bias decay, undesirable
- FOGM with non-zero mean that it can converge to


### Recap
- ACCELEROMETER bias -> treated as state itself
  - Its covariance is a random walk driven by white noise
  - Covariance propagates into velocity and position
- ACCELEROMETER white noise -> random run for $p$, random walk for $v$
- GYROSCOPE bias -> treated as state itself
  - Its covariance is a random walk driven by white noise
  - Covariance propagates into velocity and position
- GYROSCOPE white noise -> random walk in attitude (small angle)
  - Propagates into $p$ and $v$ as well due to rotation

## Multi-input Bias State Models
TODO (5 pages)