# Star-Traveller problem
## Execution
With java 1.8:
java -jar tester.jar -exec "python StarTraveller_x.py" -seed 0 -vis

## The Problem

Given a 2D grid containing N stars, S ships and U ufos the problem was to have each star visited by at least one ship in 4N moves. The cost of moving a ship is given by its Euclidean distance. The problem can be described as a multiple travelling salesman problem (mTSP) with no restriction on the salesman having to return to a designated depot (which seems to be the most popular version in the literature).

There are two addendums to this problem which complicate matters. Firstly the cost of travelling is reduced by a factor of 1000 if a ship tracks a ufo and secondly there is a restriction on the computational capacity in that the simulation needs to run in 20 seconds. Note that N is at most 2000.

The core of the solution presented here involves calculating an intrinsic value for the ufos and then carrying out a cost-benefit analysis to decide whether it is worth moving a ship to track each of the ufos. The value of the ufos is given by their ability to visit new stars in the system. This in turn necessitates the valuation of the stars as well as the probability of visiting new stars.

Lets tackle the star evaluation first. If we were to do this as rigorously as possible then we would evaluate the mTSP at each timestep. The value of a star would then be given by the change in the cost of the mTSP by said stars removal. This is far too costly given the time constraints so we need some heuristic to make these evaluations instead. For each star we choose k of its unvisited nearest neighbors and calculate their average distance. This gives some sense of the remoteness of the star. The more remote the star the higher its valuation.

In order to calculate the intrinsic value of a ufo we also need the expected number of unvisited stars it expects to land on in the remaining number of moves. We record the previous 1000 moves and check how many of those moves involved a visit to a new star. This gives us the current probability of visiting a new star. This probability decreases in time however and we take account of this.

Finally we rank the moves according to (value - cost) and carry out the best moves first.

