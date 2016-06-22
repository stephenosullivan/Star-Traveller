import random
import heapq
from collections import OrderedDict
from collections import deque
from itertools import takewhile
from copy import deepcopy
import cProfile, pstats
from math import sqrt, exp
#from statistics import mean

#can call getRemainingTime()
# Class:	Server
# Method:	getRemainingTime
# Parameters:	integer
# Returns:	integer
# Sample Call: val = Server.getRemainingTime(ignore)


class StarTraveller:


    def init(self, stars):
        self.NStars = len(stars) // 2
        self.used = set([])
        self.cnt = 4*self.NStars - 1
        self.trackedcnt = 0
        self.untrackedufos = dict()
        self.trackingship = dict()
        self.ufos = dict()
        self.ufoDicoveryCnt = dict()
        self.firstmove = True
        self.ret = []
        self.ufoLocations = dict()
        self.UFODiscoveryCounterLength = min(1000, self.NStars)
        self.unvisited = {i:(stars[2*i],stars[2*i+1]) for i in range(self.NStars)}
        self.unvisitedLocationsStars = {(stars[2*i],stars[2*i+1]):i for i in range(self.NStars)}
        self.stars = {i:(stars[2*i],stars[2*i+1]) for i in range(self.NStars)}
        self.stardistances = None
        self.stardistancesunordered = None
        self.finalpaths = None
        self.sTSPIterations = 4000
        self.mTSPIterations = 100  # Maybe fn(len(ships))
        self.memoryConstraintNumberOfStarsInDistanceDict = 2001
        self.energyused = 0
        self.starcostlength = min(10, self.NStars)
        self.damping = 0.1
        self.move_3_damping = 0.15
        self.movecnt = [0]*4
        self.profiler = cProfile.Profile()
        self.results_holder = []
        self.starvalues = [0]*self.NStars
        self.nearestNeighborIndexList = [range(self.starcostlength) for _ in range(self.NStars)]
        self.neighborOccurences = [set() for _ in range(self.NStars)]

        return 0

    def makeMoves(self, ufos, ships):
        self.profiler.enable()
        self.results_holder.append(self.makeMovesProfiled(ufos, ships))
        self.profiler.disable()

        return self.results_holder[-1]


    def makeMovesProfiled(self, ufos, ships):
        self.ships = ships
        if self.firstmove:
            print >> sys.stderr, self.ships
        self.shipsSet = {ship for ship in ships}



        if self.firstmove:
            # Build distance matrix
            self.starDistancesBuild()

            # Construct UFO objects
            self.ufos = {i:UFO(i, (ufos[3*i], ufos[3*i+1], ufos[3*i+2]), self.UFODiscoveryCounterLength) for i in range(len(ufos)/3)}
            for i, dictufo in self.ufos.items():
                locationtuple = ufos[3 * i], ufos[3 * i + 1], ufos[3 * i + 2]

                dictufo.newstarcounter(self.newstar(locationtuple[0]))
                dictufo.addStarValueToHistory(self.getStarCostRemoteness(locationtuple[0]))

                dictufo.newstarcounter(self.newstar(locationtuple[1]))
                dictufo.addStarValueToHistory(self.getStarCostRemoteness(locationtuple[1]) - 0.001*self.getDistance(locationtuple[0], locationtuple[1]))

                dictufo.newstarcounter(self.newstar(locationtuple[2]))
                dictufo.addStarValueToHistory(self.getStarCostRemoteness(locationtuple[2]) - 0.001*self.getDistance(locationtuple[1], locationtuple[2]))

                for j in range(2):
                    dictufo.averageRange += self.getDistance(locationtuple[j], locationtuple[j+1])/(j+1)

            # Allocations
            self.ret = list(ships)
            self.UFOPriority = []



        else:
            for i, dictufo in self.ufos.items():
                dictufo.updateLocations((ufos[3*i], ufos[3*i + 1], ufos[3*i + 2]))
                dictufo.newstarcounter(self.newstar(ufos[3*i + 2]))
                tmpufodistance = self.getDistance(dictufo.nextlocations[0], dictufo.nextlocations[1])
                if dictufo.nextlocations[1] in self.unvisited:
                    dictufo.addStarValueToHistory(self.getStarCostRemoteness(ufos[3*i + 2]) - 0.001*tmpufodistance)

                else:
                    dictufo.addStarValueToHistory( - 0.001*tmpufodistance)
                dictufo.averageRange += tmpufodistance/(self.moveCnt()+3)

        self.UFOPriority = sorted([(ufo.value, self.UFONextCnt(ufo), ufo.index) for ufo in self.ufos.values()], reverse=True)
        self.starUFOPriority = {self.ufos[item[2]].location:item[2] for item in self.UFOPriority}

        self.ufoLocations = dict()
        for j, ufo in enumerate(ufos[::3]):
            #self.ufos[j] = ufo, ufos[3 * j + 1], ufos[3 * j + 2]
            if ufo in self.ufoLocations:
                self.ufoLocations[ufo].append(j)
            else:
                self.ufoLocations[ufo] = [j]

        self.trackedcnt = self.tracking()


        self.temperature = min(1, self.damping * self.expectedNumberOfTurnsToReachEndTest2(self.ufos.values())/float(self.cnt))
        print >> sys.stderr, self.cnt, len(self.unvisited), self.energyused, self.temperature


        # In TSP mode
        if self.finalpaths:
            for i, ship in enumerate(ships):
                if len(self.finalpaths[i])>1:
                    assert ship == self.finalpaths[i].pop()
                    self.move(i, self.finalpaths[i][-1])
                    self.movecnt[3] += 1

        # Test to go into TSP mode
        elif self.finalStretch() or (not self.ufos and not self.firstmove):
            print >> sys.stderr, "Final Stretch"
            self.finalpaths = self.multipleTravellingSalesmen2()
            for i, ship in enumerate(self.ships):
                if len(self.finalpaths[i])>1:
                    assert ship == self.finalpaths[i].pop()
                    self.move(i, self.finalpaths[i][-1])
                    self.movecnt[3] += 1

        else:
            # Build a value list of ufos for each ship
            # Consider intrinsic value of ship based on stars and the distance to the star
            if self.ufos:
                jumpstar = dict()
                bestshipmoves = [[0 for _ in range(len(self.ufos))] for _ in range(len(self.ships))]
                valueUnvisitedStars = sum(self.getStarCostRemoteness(star_) for star_ in self.unvisited) / len(self.unvisited)

                if self.trackedcnt:
                    expectedUnvisited = [ufo_.UnvisitedStarsExpectedToBeFoundInRemainingTime(self.trackedcnt, self.NStars, self.cnt - len(self.unvisited)) for ufo_ in self.ufos.values()]
                else:
                    expectedUnvisited = [ufo_.UnvisitedStarsExpectedToBeFoundInRemainingTime(1, self.NStars, self.cnt - len(self.unvisited)) for ufo_ in self.ufos.values()]

                """
                Greed is good but we need not only to pick the best move at a given instant but we must check the
                likelihood of getting a better deal in the future. For instance a jump to a ufo may be the best option
                now but maybe there is a lot of time left so it's intrinsic value would not change so that it would be
                better to wait until the ufo is closer (cheaper). Once time starts to run out it may become necessary to
                jump no matter what the cost
                """

                for i in range(len(self.ships)):
                    jumpdistance, jumpstar[i] = self.getNearestNeighbors(self.ships[i])[0]
                    # should put in a closest ufo container so that we can check for viable jumps

                    for ii, ufo_ in self.ufos.items():
                        moves = [(-1, -1)] * 3
                        if ufo_.location == self.ships[i]:

                            # Follow UFO
                            moves[0] = (expectedUnvisited[ii] * valueUnvisitedStars, ufo_.index)

                            # Jump off ufo to star and jump back again
                            if ufo_.nextlocations[0] not in self.unvisited and jumpstar[i] in self.unvisited:
                                moves[1] =  (self.getStarCostRemoteness(jumpstar[i]) - (jumpdistance + self.getDistance(jumpstar[i], ufo_.nextlocations[1])), ufo_.index)

                        else:
                            # Jump to UFO
                            nextdistance = self.getDistance(self.ships[i], ufo_.nextlocations[0])
                            #if self.getDistance(self.ships[i], ufo_.location) > nextdistance and self.getDistance(self.ships[i], ufo_.nextlocations[1]) > nextdistance:
                            if True:
                                # How likely is the UFO to get closer?
                                # -- What are the fraction of points within distance D?
                                timescale = min(self.NStars / self.rankingOfNeighbor(self.ships[i], ufo_.nextlocations[0]), self.cnt)

                                # Or how long will it take to get closer?
                                # -- Use that timescale to work out how much the intrinsic value of the ufo will change in that time
                                value_diff = (ufo_.UnvisitedStarsExpectedToBeFoundInRemainingTime(self.trackedcnt + 1, self.NStars, self.cnt - len(self.unvisited)) - ufo_.UnvisitedStarsExpectedToBeFoundInRemainingTime(self.trackedcnt + 1, self.NStars, self.cnt - timescale - len(self.unvisited))) * sqrt(len(self.unvisited)/(len(self.unvisited) - (min(sum(expectedUnvisited), len(self.unvisited)-1))))

                                # -- Is it worth the wait?
                                if value_diff > self.getDistance(self.ships[i], ufo_.nextlocations[0]):
                                    moves[2] = (self.temperature*expectedUnvisited[ii] * valueUnvisitedStars - self.getDistance(self.ships[i], ufo_.nextlocations[0]), ufo_.index)


                        bestshipmoves[i][ii] = [(j, move_[0], move_[1]) for j, move_ in enumerate(moves)]
                    bestshipmoves[i] = [movetuple for movetuplelist in bestshipmoves[i] for movetuple in movetuplelist]
                    bestshipmoves[i].append((3, self.temperature *self.move_3_damping * self.getStarCostRemoteness(jumpstar[i]) - jumpdistance, -1))

                    # Remove all cases where the surplus is less than zero
                    bestshipmoves[i] = [move_ for move_ in bestshipmoves[i] if move_[1] > 0]

                    if bestshipmoves[i]:
                        bestshipmoves[i].sort(key=lambda x: x[1], reverse=True)
                    else:
                        bestshipmoves[i] = [(-1, - float('inf'), -1)]


                self.sortedBestMoves = sorted([(ii, moves) for ii, moves in enumerate(bestshipmoves)], key=lambda x: x[1][0][1], reverse=True)

                markedufos = set()
                for ship_i, moves in self.sortedBestMoves:
                    #print >> sys.stderr, "Moving ship", ship_i
                    not_moved = True
                    move_i = 0
                    while not_moved and move_i < len(moves):
                        movetype, surplus, ufo_i = moves[move_i]

                        if movetype == 0:
                            if ufo_i not in markedufos:
                                self.move(ship_i, self.ufos[ufo_i].nextlocations[0], tracked=True)
                                markedufos.add(ufo_i)
                                not_moved = False
                                self.movecnt[0] += 1

                        elif movetype == 1:
                            if jumpstar[ship_i] in self.unvisited:
                                self.move(ship_i, jumpstar[ship_i], tracked=False)
                                self.ufos[ufo_i].addStarValueToHistory(surplus)
                                not_moved = False
                                self.movecnt[1] += 1

                        elif movetype == 2:
                            if ufo_i not in markedufos:
                                self.move(ship_i, self.ufos[ufo_i].nextlocations[0], tracked=False)
                                markedufos.add(ufo_i)
                                not_moved = False
                                self.movecnt[2] += 1

                        elif movetype == 3:
                            if jumpstar[ship_i] in self.unvisited:
                                self.move(ship_i, jumpstar[ship_i], tracked=False)
                                not_moved = False
                                self.movecnt[3] += 1

                        move_i += 1

            # Tag stars as marked for stationary ships on the first move
            if self.firstmove:
                for i, ship_ in enumerate(self.ships):
                    if self.ret[i] == ship_:
                        self.removeFromUnvisited(ship_)

        self.cnt -= 1
        self.firstmove = False
        return self.ret

    def getStarCostRemoteness(self, star):
        return self.starvalues[star]

    def getDistance(self, star1, star2):
        if not star1 or not star2:
            return 0
        if star1 == star2:
            return 0
        sep = star2 - star1
        if star1 < star2:
            return self.stardistancesunordered[star1][sep - 1]
        return self.stardistancesunordered[star2][-sep - 1]

    def removeFromUnvisited(self, ship):
        if ship in self.unvisited:
            del self.unvisited[ship]

    def removeFromStarDistances(self, target):
        for starlist in takewhile(lambda x: x != target, self.stardistances.keys()):
            del self.stardistances[starlist][target]

    def moveCnt(self):
        return 4*self.NStars - self.cnt

    def starDistancesBuild(self):
        self.stardistances = [[None for _ in range(len(self.stars))] for _ in range(len(self.stars))]
        self.stardistancesunordered = [None for _ in range(len(self.stars)-1)]
        for i in range(len(self.stars)):
            for j in range(i):
                dist_i_j = self.distance(self.stars[i], self.stars[j])
                self.stardistances[i][j] = (dist_i_j, j)
                self.stardistances[j][i] = (dist_i_j, i)
            self.stardistances[i][i] = (float('inf'), i)  # 1024*1.414

        # Copy distances into unordered matrix
        for i in range(len(self.stars)-1):
            self.stardistancesunordered[i] = [item[0] for item in self.stardistances[i][i+1:]]

        for row in self.stardistances:
            row.sort()


        for star, neighbors in enumerate(self.nearestNeighborIndexList):
            for rank, neighborindex in enumerate(neighbors):
                self.neighborOccurences[self.stardistances[star][neighborindex][1]].add((star, rank))

        for i in range(self.NStars):
            self.starvalues[i] = sum([self.stardistances[i][index][0] for index in self.nearestNeighborIndexList[i]]) / self.starcostlength


    def rankingOfNeighbor(self, star1, star2):
        """
        Cannot afford to do a linear sweep m(ships) x n(ufos) times for each iteration
        """
        approximateRank = min(self.NStars, 10**int(self.getDistance(star1, star2) /self.stardistances[star1][10][0]))
        if approximateRank <= 0:
            return 1
        return approximateRank

    def newstarmove(self, star):
        if star in self.unvisited:
            del self.unvisited[star]

    def test(self, x):
        return x[1] in self.unvisited


    def getNearestNeighbors(self, star, neighborcnt=1, unvisited=True):
        """

        Parameters
        ----------
        star
        neighborcnt
        unvisited

        Returns
        -------
        k nearest neighbors as list [(distance, star), ...]
        """

        nearestNeighbors = [self.stardistances[star][index] for index in self.nearestNeighborIndexList[star][:neighborcnt]]
        return nearestNeighbors

    def getNearestNeighborsExcluding(self, star, neighborcnt=1, excluding=set(), unvisited=True):
        """

        Parameters
        ----------
        star
        neighborcnt
        unvisited

        Returns
        -------
        k nearest neighbors as list [(distance, star), ...]
        """
        if not unvisited:
            return self.stardistances[:neighborcnt]
        nearestNeighbors = [None] * neighborcnt
        cnt = 0
        found = 0
        while found < neighborcnt:
            if self.stardistances[star][cnt][1] in self.unvisited and self.stardistances[star][cnt][1] not in excluding:
                nearestNeighbors[found] = self.stardistances[star][cnt]
                found += 1
            cnt += 1

        return nearestNeighbors

    def greedyPath(self, star): # TODO  forgetting to put the initial star in path
        distance = 0
        path = [None]*(len(self.visited)+1)
        current = star
        path[0]  = current

        cnt = 1
        while cnt - 1 < len(self.unvisited):
            nextstar = self.getNearestNeighbors(current)[0] # Get first elem from list
            path[cnt] = nextstar[1]
            current = nextstar[1]
            distance += nextstar[0]
            cnt += 1

        return path, distance

    def greedyMultiplePaths(self, ships):
        initialstars = ships[:]
        distance = 0
        paths = [[None for _ in range(len(self.unvisited)+1)] for _ in range(len(self.ships))]
        marked = set()

        for i in range(len(self.ships)):
            paths[i][0] = initialstars[i]

        pathcnters = [1 for _ in range(len(self.ships))]
        currentstars = initialstars

        while sum(pathcnters)-len(self.ships) < len(self.unvisited):
            index, nextstar = min([(i, self.getNearestNeighborsExcluding(currentstars[i], excluding=marked)[0]) for i in range(len(self.ships))], key=lambda x: x[1][0])
            paths[index][pathcnters[index]] = nextstar[1]
            pathcnters[index] += 1
            currentstars[index] = nextstar[1]
            marked.add(nextstar[1])
            distance += nextstar[0]

        return [paths[i][:pathcnters[i]] for i in range(len(self.ships))], distance

    def getPathLength(self, starlist):
        """
        Given a list of stars evaluate the length of the path
        Returns
        -------

        """
        sum = 0
        for i in range(len(starlist)-1):
            sum += self.getDistance(starlist[i],starlist[i+1])
        return sum

    def twoOpt(self, starlist, Niterations=10):
        if len(starlist)>2:
            for i in range(Niterations):
                twonodes = random.sample(range(1,len(starlist)+1), 2)
                edges = sorted(twonodes)
                if edges[1] == len(starlist):
                    if self.getDistance(starlist[edges[0] - 1], starlist[edges[0]]) > self.getDistance(starlist[edges[0] - 1], starlist[edges[1] - 1]):
                        starlist = starlist[:edges[0]] + starlist[edges[0]:edges[1]][::-1]
                else:
                    if self.getDistance(starlist[edges[0]-1], starlist[edges[0]]) + self.getDistance(starlist[edges[1]-1], starlist[edges[1]]) > \
                        self.getDistance(starlist[edges[0]-1], starlist[edges[1]-1]) + self.getDistance(starlist[edges[0]], starlist[edges[1]]):
                        starlist = starlist[:edges[0]] + starlist[edges[0]:edges[1]][::-1] + starlist[edges[1]:]

        return starlist

    def UFONextCnt(self, ufo):
        cnt = 0
        for star in ufo.nextlocations:
            if star in self.unvisited:
                cnt += 1
        return cnt

    def multipleTravellingSalesmen(self, iterations=None):
        """
        Outputs a list of stars to visit solving TSP on unvisited stars for multiple ships
        """
        if not iterations:
            iterations = self.mTSPIterations

        # Randomly select points drawn from unvisited and distribute equally among the ships
        paths = self.partition(self.unvisited.keys(), len(self.ships))
        [path.append(ship) for path, ship in zip(paths, self.ships)]

        # Run single travelling salesman on each of the ships
        paths = [self.singleTravellingSalesman(path, iterations=0) for path in paths]


        # IF TWO OR MORE SHIPS!!!
        # Point exchange between ships and rerun tsp on each ship
        if len(self.ships) > 1:
            for _ in range(iterations):
                twopathindices = random.sample(xrange(len(paths)), 2)
                newtwopaths = self.swapTwoPointsBetweenPaths([paths[index] for index in twopathindices])
                newtwopaths = [self.singleTravellingSalesman(path, iterations=max(len(path), 100)) for path in
                               newtwopaths]
                if self.totalPathDistance(newtwopaths) < self.totalPathDistance(
                        [paths[index] for index in twopathindices]):
                    for i, index in enumerate(twopathindices):
                        paths[index] = newtwopaths[i]

        return paths

    def multipleTravellingSalesmen2(self, iterations=None):
        """
        Outputs a list of stars to visit solving TSP on unvisited stars for multiple ships
        """
        if not iterations:
            iterations = self.mTSPIterations

        paths, self.totaldistanceMTSP = self.greedyMultiplePaths(self.ships)

        # Run single travelling salesman on each of the ships
        paths = [self.singleTravellingSalesman2(path, iterations=5000) for path in paths]

         # IF TWO OR MORE SHIPS!!!
        # Point exchange between ships and rerun tsp on each ship
        if len(self.ships) > 1:
            for _ in range(iterations):
                twopathindices = random.sample(xrange(len(paths)), 2)
                if min([len(path) for path in paths]) > 1:
                    newtwopaths = self.swapTwoPointsBetweenPaths([paths[index] for index in twopathindices])
                    newtwopaths = [self.singleTravellingSalesman2(path, iterations=100) for path in newtwopaths]
                    if self.totalPathDistance(newtwopaths) < self.totalPathDistance([paths[index] for index in twopathindices]):
                        for i, index in enumerate(twopathindices):
                            paths[index] = newtwopaths[i]

        return [path[::-1] for path in paths]

        # Might be a simple tsp whereby the new points search  for the twp closest points and
        # inserts between them on the path

    def partition(self, lst, n):
        """
        Evenly partition a list into n segments
        """
        division = len(lst) / float(n)
        return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n)]

    def singleTravellingSalesman(self, path, iterations=None):
        if not iterations:
            iterations = self.sTSPIterations
        for _ in range(iterations):
            newpath = self.swapTwoPoints(path, skiplast=True)

            if self.pathDistance(newpath) < self.pathDistance(path):
                path = newpath
        return path

    def singleTravellingSalesman2(self, path, iterations=None):
        if not iterations:
            iterations = self.sTSPIterations

        path = self.twoOpt(path, Niterations=iterations)

        return path

    def swapTwoPoints(self, path, skiplast=True):
        newpath = path[:]
        if skiplast:
            length = len(path)-1
        else:
            length = len(path)

        i, j = random.sample(range(length),2)
        newpath[i], newpath[j] = newpath[j], newpath[i]
        return newpath

    def swapTwoPointsBetweenPaths(self, paths, skipfirst=True):

        if skipfirst:
            skip = 1
        else:
            skip = 0

        index1, index2 = map(random.choice, (range(skip,len(paths[0])), range(skip, len(paths[1]))))

        # 50-50 whether to swap points or gain a point
        if random.getrandbits(1):
            paths[0] = self.greedyEmplacement(paths[1][index2], paths[0][:index1]+paths[0][index1+1:])
            paths[1] = self.greedyEmplacement(paths[0][index1], paths[1][:index2]+paths[1][index2+1:])

        else:
            paths[0] = self.greedyEmplacement(paths[1][index2], paths[0])
            paths[1] = paths[1][:index2] + paths[1][index2+1:]

        return paths

    def greedyEmplacement(self, newstar, starlist):
        if len(starlist) > 1:
            index, _ = min([(i+1, self.getDistance(newstar, starlist[i]) + self.getDistance(newstar, starlist[i+1])) for i in range(len(starlist)-1)], key=lambda x: x[1])
            return starlist[:index] + [newstar] + starlist[index:]
        return starlist + [newstar]

    def pathDistance(self, path):
        return sum(self.distance(self.stars[loc1], self.stars[loc2]) for loc1, loc2 in zip(path[:-1], path[1:]))

    def totalPathDistance(self, paths):
        return sum(self.pathDistance(path) for path in paths)

    def selectTwoPoints(self, path):
        return random.sample(path, 2)

    def conditionToWaitForUFOs(self):
        return self.expectedNumberOfTurnsToReachEndTest2(self.ufos.values()) < self.cnt

    def roundup(self, value):
        if int(value) == value:
            return value
        return value + 1

    def finalStretch(self):
        return len(self.unvisited) >= self.cnt

    def newstar(self, star):
        return star in self.unvisited


    def squareCounterPlusOne(self):
        pass

    def makeBestMove(self, index, ship):
        self.chooseRandomUnvisited(index, ship)

    def chooseRandomUnvisited(self, index, ship):
        if ship not in self.unvisited:
            self.ret[index] = self.visitnewstar()
        # else stay put and visit current star
        else:
            del self.unvisited[ship]


    def move(self, shipindex, star, tracked=False):
        if tracked is False:
            factor = 1
        else:
            factor = 0.001
        currentstar = self.ships[shipindex]
        self.ret[shipindex] = star
        if star in self.unvisited:
            del self.unvisited[star]
            self.starvalues[star] = 0
            for sourcestar, rank in self.neighborOccurences[star]:
                if self.unvisited:
                    self.starvalues[sourcestar] = (self.starvalues[sourcestar]*(len(self.nearestNeighborIndexList[sourcestar])+1) - self.stardistances[sourcestar][self.nearestNeighborIndexList[sourcestar][rank]][0]) / (len(self.nearestNeighborIndexList[sourcestar]))
                del self.nearestNeighborIndexList[sourcestar][rank]

                for offset, starindex_ in enumerate(self.nearestNeighborIndexList[sourcestar][rank:]):
                    self.neighborOccurences[self.stardistances[sourcestar][starindex_][1]].remove((sourcestar, rank+offset+1))
                    self.neighborOccurences[self.stardistances[sourcestar][starindex_][1]].add((sourcestar, rank+offset))

                if len(self.unvisited) >= self.starcostlength:
                    cnt = self.nearestNeighborIndexList[sourcestar][-1] + 1
                    while self.stardistances[sourcestar][cnt][1] not in self.unvisited:
                        cnt += 1

                    self.nearestNeighborIndexList[sourcestar].append(cnt)
                    self.starvalues[sourcestar] = (self.starvalues[sourcestar] * len(self.nearestNeighborIndexList[sourcestar]) + self.stardistances[sourcestar][self.nearestNeighborIndexList[sourcestar][-1]][0]) / (len(
                        self.nearestNeighborIndexList[sourcestar]) + 1)
                    self.neighborOccurences[self.stardistances[sourcestar][cnt][1]].add((sourcestar, self.starcostlength-1))

            self.neighborOccurences[star] = None

            #speedpoint 15 nebula source
        self.energyused += self.getDistance(currentstar, star) * factor

    def tracking(self):
        """
        Return number of ships tracking ufos and fill dicts untrackufos and trackingship
        """
        cnt = 0
        self.untrackedufos = deepcopy(self.ufoLocations)

        for ship in self.ships:
            if ship in self.untrackedufos:
                cnt += 1
                self.trackingship[ship] = self.untrackedufos[ship].pop()
                if not self.untrackedufos[ship]:
                    del self.untrackedufos[ship]

        return cnt

    def visitnewstar(self):
        newstar = self.unvisited.keys()[0]
        del self.unvisited[newstar]
        return newstar

    def expectedNumberOfTurnsToReachEnd(self, ufos):
        if self.trackedcnt:
            # unvisited stars visited by UFOs in next two rounds
            return 2 * float(len(self.unvisited)) / len(self.UFOUpcomingVisited(ufos))
        return float("inf")

    def expectedNumberOfTurnsToReachEndTest(self, ufos):
        numberOfNewVisits = len(self.UFOUpcomingVisitedTest(ufos))
        if numberOfNewVisits:
            return 2 * float(len(self.unvisited)) / numberOfNewVisits * float(len(ufos))/len(self.ships)
        return float("inf")

    def expectedNumberOfTurnsToReachEndTest2(self, ufos):
        average = self.averageprobability(ufos)
        if average:
            return (float(len(self.unvisited))/self.averageprobability(ufos)) /len(self.ships)
        return float("inf")

    def expectedNumberOfTurnsToReachEndSingle(self, ufo):
        if ufo.discoveryprobability():
            return float(len(self.unvisited))/ufo.discoveryprobability()
        return float("inf")

    def averageprobability(self, ufos):
        if ufos:
            return float(sum([ufo.discoveryprobability() for ufo in ufos]))/len(ufos)
        return 0

    def projectedNumberOfStarsFoundBeforeEnd(self, ufo):
        return self.cnt * ufo.discoveryprobability()

    def UFOUpcomingVisited(self, ufos):
        # Newly visited stars by UFOs
        ret = []

        for index in self.trackedufos:
            if ufos[3*index + 1] in self.unvisited:
                ret.append(ufos[3*index + 1])
            if ufos[3*index + 2] in self.unvisited:
                ret.append(ufos[3*index + 2])
        return ret

    def UFOUpcomingVisitedTest(self, ufos):
        # Need to normalize by the number of tracked ufos

        # Newly visited stars by UFOs
        ret = []

        for ufoTuple in self.ufos.values():
            if ufoTuple[1] in self.unvisited:
                ret.append(ufoTuple[1])
            if ufoTuple[2] in self.unvisited:
                ret.append(ufoTuple[2])

        return ret

    def findClosestUFO(self, ship):
        pass


    def findClosestStar(self, ship):
        mindistance = 2000
        for star in self.stars:
            if self.distance(star, ship):
                pass

    def distance(self, obj1, obj2):
        return ((obj2[0]-obj1[0])**2 + (obj2[1] - obj1[1])**2)**0.5

    def makeMovestest(self, ufos, ships):
        ret = []
        for i in range(self.NStars):
            if i not in self.used:
                self.used.add(i)
                ret.append(i)
                if len(ret) == len(ships):
                    break
        # Make sure the return is filled with valid moves for the final move.
        while len(ret) < len(ships):
            ret.append((ships[len(ret)] + 1) % self.NStars)
        return ret

    def nearestStar(self, loc):
        """
        Given location find nearest star
        """
        if loc in self.stardistances:
            return self.stardistances[loc].keys()[0]
        return False


    def nearestShip(self, item):
        pass

    def nearestUFO(self, item):
        pass

class UFO:
    def __init__(self, i, locations, discoveryLength=200):
        self.location = locations[0]
        self.nextlocations = (locations[1], locations[2])  # Dummy locations
        self.index = i
        self.discoveryLength = float(discoveryLength)
        self.discoveringpower = self.discoveryLength
        self.averageRange = 0
        self.history = deque(discoveryLength*[0.001], discoveryLength)
        self.unvisitednumber = self.discoveryLength
        self.value = 0

    def newstarcounter(self, newstar):
        if newstar:
            if self.discoveringpower < self.discoveryLength:
                self.discoveringpower += 1
        else:
            if self.discoveringpower > 0:
                self.discoveringpower -= 1

    def discoveryprobability(self):
        return self.unvisitednumber / self.discoveryLength

    def updateLocations(self, locationtuple):
        self.location = locationtuple[0]
        self.nextlocations = (locationtuple[1], locationtuple[2])

    def __lt__(self, other):
        return self.discoveryprobability() < other.discoveryprobability()

    def addStarValueToHistory(self, starcost):
        tmp = self.history[0]
        if starcost > 0 >= tmp:
            self.unvisitednumber += 1
        elif starcost <= 0 < tmp:
            self.unvisitednumber -= 1

        self.value += (starcost - tmp) / self.discoveryLength
        self.history.append(starcost)


    def historyvalue(self):
        return [item/self.discoveryLength for item in map(sum, zip(*self.history)) ]

    def amortized(self):
        return sum(item for item in self.history) / self.discoveryLength

    def remainingStarsDiscoveryLikelihood(self, unvisited):
        return len([elem for elem in self.history if elem[2] in unvisited])/ self.discoveryLength

    def UnvisitedStarsExpectedToBeFoundInRemainingTime(self, NTrackingShips, NStars, NStepsRemaining):
        # Expected Number To be found in next step
        p = self.discoveryprobability()

        """
        E[1] = p1
        E[2] = (NS - 1)* E[1]/NS = E[1] - E[1]/NS
        E[3] = (NS*E[1] - (E[1] + E[2]))/NS = (NS*E[1] - (2*E[1] - E[1]/NS) )/NS = E[1] - 2*E[1]/NS + E[1]/NS**2

        E[N] = E[1]*(1 - 1/NS)**(N-1)

        Sum(E[i] i from 1 to N) = E[1](N - N(N-1)/(2 NS) + N*(N-1)*(N-2)/(6 NS**2))

        Sum(E[i] i from 1 to N) = E[1] * (1 - x**N)/(1-x)       x = (1 - 1/NS)
                                = E[1] * (1 - (1-e)**N))/(1 - (1-e))
                                = E[1] * e * [1 - (1 - Ne + N(N-1)e**2/2)]
                                = E[1] * e * (Ne)


        E.G.:
        Nstars = 1000
        NTrackingShips = 1
        x = 0.999

        1 - x**N
        """
        p *= NStars * (1 - (1 - 1 / float(NStars)) ** (NStepsRemaining * NTrackingShips))
        return p

"""
Submission results:
---
Examples:
2) Score: 12086.01060445081 Run Time: 10986 ms Peak Memory Used: 466.137MB
---
- mTSP using greedy
"""

# -------8<------- end of solution submitted to the website -------8<-------

import sys

def getVector(v):
    for i in range(len(v)):
        v[i] = int(raw_input())

NStars = int(raw_input())
stars = NStars * [0]
getVector(stars)

algo = StarTraveller()
ignore = algo.init(stars)
print ignore
sys.stdout.flush()

while True:
    NUfo = int(raw_input())
    if NUfo < 0:
        break
    ufos = NUfo * [0]
    getVector(ufos)
    NShips = int(raw_input())
    ships = NShips * [0]
    getVector(ships)
    ret = algo.makeMoves(ufos, ships)
    print len(ret)
    for num in ret:
        print num
    sys.stdout.flush()
sortby = 'cumulative'

print >> sys.stderr, "This is the movecnt", algo.movecnt
with open('profiledata1', 'w') as f:
    ps = pstats.Stats(algo.profiler, stream=sys.stderr).sort_stats(sortby)
    ps.print_stats(10)
