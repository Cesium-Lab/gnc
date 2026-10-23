# Classical Controls - (with Transfer Function) 
- Broken into
  - Transfer function control (mmmm)
  - State space control (YAY)
- Basics
  - System/plant
    - Has an input and makes an output
    - ![alt text](images/classical/image.png)
  - Open and closed loop
    - Uses controller to shape input to plant to get desired output
    - ![alt text](images/classical/image-10.png)
## Transfer Function
  - Linear
  - Transformed via LAPLACE TRANSFORM
    - [Table of transformations](#appendix)
  - How a system responds (gain, phase) to different frequencies
  - Multiplication is convolution in time domain
  - ![alt text](images/classical/image-2.png)
- **Poles**
  - Roots of denominator of TF
  - Drive response to infinity
  - Negative real part means decay
  - Pole locations
    - ![alt text](images/classical/image-3.png)
  - Real poles
    - Exponential growth or decay
  - Imaginary poles
    - Sinusoidal component 
  - Complex poles
    - Some combination of the above two
    - Sinusoid enveloped by exponential decay or growth
    - ![alt text](images/classical/image-1.png)
- **Zeros**
  - Roots of numerator of TF
  - Output becomes 0
  - Affect how much a pole contributes to the



### Stability
- Stable pole
  - Real{p} < 0
- Critically/marginally stable pole
  - Real{p} = 0
- Unstable
  - Real{p} > 0

### Steady State
-  ![alt text](images/classical/image-6.png)
-  Exists If all poles are in left hand side
-  **Final Value Theorem**
   -  ![alt text](images/classical/image-6.png)
   -  Steady state must exist for this to be true


<!-- ### BIBO
- Bounded input, bounded output
  - All real parts of poles are BIBO stable -->



## Controller Overview

- Response of input, disturbance, and noise all add linearly
![alt text](images/classical/image-11.png)

Good controller:
1. Basic checks
   - Stability (location of poles)
   - Robustness to modeling errors/disturbances (closed loop does this)
2. Steady state requirements
   - Tracking error
3. Transient Requirements
   - Rise time, overshoot, settling time, peak time
   - Analytic for 2nd order system
   - ![alt text](images/classical/image-8.png)
4. Objective/cost function (not as much in TF, mainly in state space) 
   - Obstacle avoidance, control effort 

### 2nd order system
- Full formula
  - ![alt text](images/classical/image-9.png)
- Transfer function 
  - Damping coefficient $\eta$ in $H(s) = \frac{\omega_n^2}{s^2 + 2 \eta \omega_n s + \omega_n ^2}$ where $\omega_n > 0$
  - ![alt text](images/classical/image-7.png)
  - Want damping so it is stable
- Analytic transient specs

### Open vs Closed loop
- Open Loop
  - Bad at disturbance rejection
  - AND MODEL ERRORS 
- Closed loop deals with this
  - Good tracking, disturbance rejection, model error sensitivity
  - Poor with sensor noise
    - Filtering (observer)
    - Better sensors (less noise)
    - Multiple sensors (fusion)
  
## Control Design Techniques
The methods and tools used.

Key note: **hitting a gain of -1 ($\pm180\degree$) is bad and makes things flip to unstable**
(magnitude = $1$ AND phase = $\pm180\degree$ at the same time)

### Root Locus
![alt text](images/classical/image-12.png)
- Show poles for a specific gain of K
- Can change stability and response type (damped, undamped, etc.)
- **Helpful for transient specs**
- Applied when you have a specific "form" of a controller but need to adjust the gain as the last step to adjust the pole and zero locations
- Strategy for lead-lag
  - Design lead controller for transient specs
    - Angle condition ($180 \degree$) for poles and zeros
    - Magnitude condition for gain $K$
  - Design lag controller 
    - K basically unchanged since pole is approx at zero
    - Design ratio of pole and zero

### Bode Plot
![alt text](images/classical/image-13.png)
- **Helpful for steady state error**
- Magnitude and phase plots
- Can be experimental (system ID)
- Use just frequency response (no damping)
- Asymptotes
  - Just set $\omega$ to 0 or infinity and see where it leads
- Types of responses
  - ![alt text](images/classical/image-14.png)
- **System ID**
  - Get TF of system based on slopes, intercepts, and asymptotes
    - Pole adds negative slope, negative phase
    - Zero adds positive slope, positive phase

- **Gain margin** - When exactly out of phase ($\pm 180\degree$), how much extra gain can it tolerate before I risk being at $-1 + 0j$ and it goes unstable
  - That is, if the response flips past 180, the output does not blow up since it is less than 1
- **Phase margin** - When exactly at a magnitude of 1, how much extra phase can I tolerate before I risk hitting $\pm 180\degree$ and becoming unstable (by being at $-1 + 0j$)
  - That is, if 
- **Bandwidth** - Magnitude when system responds at half power
  - Higher bandwidth freq means faster rise time
  - Defines the point at which it "no longer follows the reference" and is more affected by noise (at the "ideal" corner frequency)
- Max phase added when pole and zero are FAR apart for lead and lag
- Strategy
  - High margins
    - To deal with the literal definition of instability with poles
  - High gain at low freq
    - References mainly live here so we want the system to respond quickly to them
    - tracking/disturbance rejection
  - High bandwidth freq
    - Allows system to react to higher and higher reference frequencies 
    -  But also more sensitivity to noise
  - Close to -20dB slope at (gain) crossover freq
    - Means one pole dominates, so -90 phase lag which is good for phase margin

### Nyquist Plots
- Maps right hand (unstable) plane encirclement (of where the unstable poles are) to some other locus
- Encirclement cancels out an unstable pole
  - Counter-clockwise good
- Helps with stability
- Also shows margins with $-1 + 0j$


## Controllers

### Lead Controller
- Pole further left than zero, adds phase (and phase margin)
- To meet transient specs
- Large poles and zeros to move root locus to be more stable
- Increases bandwidth


### Lag Controller
- Pole further right than zero, subtracts phase
  - Pole almost at origin (integrator) to remove steady state error
- Avoids windup (integrator never forgets)
- Small poles and zeros to not affect transient specs

### Lead-lag controller
- For transient specs AND steady state tracking


### P controller


### PD controller


### PI controller

### PID controller

### PID

- $P$
  - Proportional to the error term
  - Risk of overshooting for gain too large and under-reacting if the gain is too small
- $D$
  - Rate of change of error term. Prevents overshooting by damping the system
  - Depending on the damping ratio it makes, there is some form of a second order system (over, under, critically damped, undamped) on the denominator of transfer function
- $I$
  - Integral of error term
  - P and PD can still have small offsets when the control effort exactly equals the error 
  - Rejects steady state error


# Classical Controls - (with State Space)
- Also what advanced feedback controller design (AA 212) goes into
- not just SISO



![alt text](images/classical/image-15.png)
![alt text](images/classical/image-19.png)


### Solution
![alt text](images/classical/image-18.png)
### With TF

- **Controller Canonical Form**
  - Can have different state space representations for same system
  - ![alt text](images/classical/image-17.png)
  - ![alt text](images/classical/image-16.png)

## Stability
- Internally
  - Bounded for all initial states
- Asymptotically
  - Internally stable AND has steady state decay to 0
  - If real parts <0
- Exponentially
  - Every solution bounded by decaying exponential (so also asymptotically)
- BIBO stable
  - All inputs give bounded output
  - Unobservable means it really doesn't matter
  - Uncontrollable means we can't do anything about it even if ideally it starts at 0 (since we are testing the effects of INPUT)

## Controllability
TODO: better images

- Must be full rank matrix
  - ![alt text](images/classical/image-102.png)
- Can drive poles to anywhere (with desired characteristic polynomial)
- Advanced way of deriving
  - ![alt text](images/classical/image-105.png)
  - ![alt text](images/classical/image-202.png)

### Controller Pole Placement
- Place $eig(A - BK_x)$ at $eig(A_{cl})$
- Procedure for SISO
  - ![alt text](images/classical/image-8.png)
  - ![alt text](images/classical/image-7.png)


## Observability
Now have estimate of state
- Must be full rank
  - ![alt text](images/classical/image-103.png)
- Observation of state converges on real state (with desired characteristic polynomial)
- Advanced way of deriving
  - ![alt text](images/classical/image-104.png)

## Observer Pole Placemenet
- ![alt text](images/classical/image-10.png)
- Want poles to be 2-5 to the left of fastest CL control pole so that it controls something it is actually accurate for
- Procedure for SISO
  - ![alt text](images/classical/image-11.png)


### Luenberger Observer
- Observer has its own dynamics based on measurement error from incorrect estimate
  - ![alt text](images/classical/image-3.png)
  - ![alt text](images/classical/image-25.png)
  - ![alt text](images/classical/image-27.png)
- As long as stable, it converges (ocl = observer closed loop)
  - ![alt text](images/classical/image-26.png)
  - ![alt text](images/classical/image-9.png)

## **Separation principle**
- Can separate controllability and observability design
  - ONLY for linear systems
  - Means I can make a diagonal uncoupled matrix


## Similarity
- Same system can have different SS representations
![alt text](images/classical/image-1.png)
![alt text](images/classical/image-6.png)


## Duality
- Controller and Observer are similar in a way
- 




- **PID**: u = Kp·e + Ki∫e dt + Kd·de/dt. Know what each term physically does (P = current error, I = eliminates steady-state error but risks windup, D = damps oscillation but amplifies noise). Integral windup and anti-windup (clamping, back-calculation) is a common follow-up question.

- **Pole placement**: choose feedback gain K such that eigenvalues of (A − BK) land at desired locations (Ackermann's formula for SISO). Direct but doesn't optimize any cost, just places poles wherever you ask.


## Linearity
- How to linearize it
![alt text](image.png)


# APPENDIX

![alt text](images/classical/image-4.png)



TODO: finish 212