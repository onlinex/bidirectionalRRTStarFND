import random
import math
import copy
import sys
import pygame
import timeit
import numpy as np
show_animation = True

XDIM = 720
YDIM = 500
windowSize = [XDIM, YDIM]

pygame.init()
fpsClock = pygame.time.Clock()

screen = pygame.display.set_mode(windowSize)
screen.fill((255, 255, 255))
pygame.display.set_caption('Performing RRT')


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList,
                 randArea, expandDis=5.0, goalSampleRate=15, maxIter=500):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.Xrand = randArea[0]
        self.Yrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=True):
        u"""
        Pathplanning
        animation: flag for animation on or off
        """
        self.nodeList = {0:self.start}
        i = 0
        t = 0
        m = 4
        #for i in range(self.maxIter):
        while True:
            i+=1
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd) # get nearest node index to random point

            newNode = self.steer(rnd, nind) # generate new node from that nearest node in direction of random point

            if self.__CollisionCheck(newNode, self.obstacleList): # if it does not collide
                nearinds = self.find_near_nodes(newNode, 5) # find nearest nodes to newNode
                newNode = self.choose_parent(newNode, nearinds) # from that nearest nodes find the best parent to newNode
                self.nodeList[newNode.parent].leaf = False
                self.nodeList[i+100] = newNode # add newNode to nodeList
                self.rewire(i+100, newNode, nearinds) # make newNode a parent of another node if necessary

                root = self.nodeList[0]
                if (root.x == 0 and root.y == 0) or (root.x == XDIM and root.y == 0) or (root.x == XDIM and root.y == YDIM) or (root.x == 0 and root.y == YDIM):
                    m += 1
                k = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
                #root.x += k[m%4][0]
                #root.y += k[m%4][1]
                
                #nearroot = self.find_near_nodes(root, 40)
                #self.rewire(0, root, nearroot)


                if i > self.maxIter:
                    leaves = [ key for key, node in self.nodeList.items() if node.leaf == True ]
                    ind = leaves[random.randint(0, len(leaves)-1)]

                    self.nodeList[self.nodeList[ind].parent].leaf = True
                    for value in self.nodeList.values():
                        if value.parent == self.nodeList[ind].parent and value != self.nodeList[ind]:
                            self.nodeList[self.nodeList[ind].parent].leaf = False
                            break

                    self.nodeList.pop(ind)

            if animation and i%10 == 0:
                self.DrawGraph(rnd)

            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYUP and e.key == pygame.K_ESCAPE):
                    sys.exit("Exiting")

            print(i)
        # generate coruse
        lastIndex = self.get_best_last_index()
        if lastIndex is None:
            return None
        path = self.gen_final_course(lastIndex)
        return path

    def choose_parent(self, newNode, nearinds):
        if len(nearinds) == 0:
            return newNode

        dlist = []
        for i in nearinds:
            dx = newNode.x - self.nodeList[i].x
            dy = newNode.y - self.nodeList[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            theta = math.atan2(dy, dx)
            if self.check_collision_extend(self.nodeList[i], theta, d):
                dlist.append(self.nodeList[i].cost + d)
            else:
                dlist.append(float("inf"))

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode.cost = mincost
        newNode.parent = minind

        return newNode

    def steer(self, rnd, nind):

        # expand tree
        nearestNode = self.nodeList[nind]
        theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
        newNode = copy.deepcopy(nearestNode)
        newNode.x += self.expandDis * math.cos(theta)
        newNode.y += self.expandDis * math.sin(theta)

        newNode.cost += self.expandDis
        newNode.parent = nind
        newNode.leaf = True
        return newNode

    def get_random_point(self):
        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(0, self.Xrand), random.uniform(0, self.Yrand)]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y]
        return rnd

    def get_best_last_index(self):

        disglist = [(key, self.calc_dist_to_goal(node.x, node.y)) for key, node in self.nodeList.items()]
        goalinds = [key for key, distance in disglist if distance <= self.expandDis]

        if len(goalinds) == 0:
            return None

        mincost = min([self.nodeList[key].cost for key in goalinds])
        for i in goalinds:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append([node.x, node.y])
            goalind = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def find_near_nodes(self, newNode, value):
        r = self.expandDis * value
        dlist = [(key, (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2) for key, node in self.nodeList.items()]
        nearinds = [key for key, distance in dlist if distance <= r ** 2]
        return nearinds

    def rewire(self, newNodeInd, newNode, nearinds):
        nnode = len(self.nodeList)
        for i in nearinds:
            nearNode = self.nodeList[i]

            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y
            d = math.sqrt(dx ** 2 + dy ** 2)

            scost = newNode.cost + d

            if nearNode.cost > scost:
                theta = math.atan2(dy, dx)
                if self.check_collision_extend(nearNode, theta, d):
                    self.nodeList[nearNode.parent].leaf = True
                    for value in self.nodeList.values():
                        if value.parent == nearNode.parent and value != nearNode:
                            self.nodeList[nearNode.parent].leaf = False
                            break

                    nearNode.parent = newNodeInd
                    nearNode.cost = scost
                    newNode.leaf = False
                    #print('rewired: ' + str(nearNode.x) + ', ' + str(nearNode.y))

    def check_collision_extend(self, nearNode, theta, d):

        tmpNode = copy.deepcopy(nearNode)

        for i in range(int(d / self.expandDis)):
            tmpNode.x += self.expandDis * math.cos(theta)
            tmpNode.y += self.expandDis * math.sin(theta)
            if not self.__CollisionCheck(tmpNode, self.obstacleList):
                return False

        return True

    def DrawGraph(self, rnd=None):
        u"""
        Draw Graph
        """
        screen.fill((255, 255, 255))
        for node in self.nodeList.values():
            if node.parent is not None:
                pygame.draw.line(screen,(0,255,0),[self.nodeList[node.parent].x,self.nodeList[node.parent].y],[node.x,node.y])

        for node in self.nodeList.values():
            if node.leaf == True:
                #pygame.draw.circle(screen, (255,0,255), [int(node.x),int(node.y)], 2)
                pass

        for (ox, oy, size) in self.obstacleList:
            pygame.draw.circle(screen, (0,0,0), [ox,oy], size)

        pygame.draw.circle(screen, (255,0,0), [self.start.x, self.start.y], 10)
        pygame.draw.circle(screen, (0,0,255), [self.end.x, self.end.y], 10)

        lastIndex = self.get_best_last_index()
        if lastIndex is not None:
            path = self.gen_final_course(lastIndex)

            ind = len(path)
            while ind > 1:
                pygame.draw.line(screen,(255,0,0),path[ind-2],path[ind-1])
                ind-=1

        pygame.display.update()
        #fpsClock.tick(10)


    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [ (key, (node.x - rnd[0]) ** 2 + (node.y - rnd[1])** 2) for key, node in nodeList.items()]
        minind = min(dlist, key=lambda d: d[1])
        return minind[0]

    def __CollisionCheck(self, node, obstacleList):
        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = dx * dx + dy * dy
            if d <= size ** 2:
                return False  # collision

        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
        self.leaf = True


def main():
    print("start RRT path planning")

    # ====Search Path with RRT====
    obstacleList = [
        (50, 50, 10),
        (400, 200, 15),
        (400, 250, 25),
        (300, 180, 20),
        #(7, 5, 2),
        #(9, 5, 2)
    ]  # [x,y,size]
    # Set Initial parameters
    rrt = RRT(start=[1, 0], goal=[500, 300],
              randArea=[XDIM, YDIM], obstacleList=obstacleList)
    path = rrt.Planning(animation=show_animation)


    # Draw final path
    if show_animation:

        rrt.DrawGraph()

        ind = len(path)
        while ind > 1:
            pygame.draw.line(screen,(255,0,0),path[ind-2],path[ind-1])
            ind-=1

        pygame.display.update()

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYUP and e.key == pygame.K_ESCAPE):
                    sys.exit("Exiting")

if __name__ == '__main__':
    main()