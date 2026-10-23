# Decision Making Under Uncertainty

TODO: do this, and orbital rendezvous

For putting all of my prototypes to put in the Python package

- Dynamic Programming
- MDP
- RL
- Planning

Theory and derivations --> docs
notebooks --> playground


| Algorithm       | Typical discipline   | Why                                                            |
| --------------- | -------------------- | -------------------------------------------------------------- |
| Value iteration | Decision-making      | Computes an optimal policy for an MDP                          |
| Q-learning      | Decision-making / RL | Learns a policy from experience                                |
| A\*             | Guidance / Planning  | Finds a path through a search space                            |
| RRT             | Guidance / Planning  | Searches feasible paths in configuration space                 |
| Lambert solver  | Guidance             | Computes orbital transfer trajectories                         |
| PID             | Control              | Regulates system behavior                                      |
| LQR             | Control              | Computes optimal feedback                                      |
| MPC             | Control + Guidance   | Optimizes predicted future motion and executes feedback        |
| iLQR / DDP      | Control + Guidance   | Optimizes trajectories and can produce local feedback policies |
| POMDP planning  | Decision-making      | Selects actions under state uncertainty                        |