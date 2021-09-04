# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from queue import PriorityQueue
from queue import deque
from collections import deque

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """


    queue = []
    explored = set()
    queue.append(maze.start)
    while queue:
        if queue[0] == maze.start:
            cur_path = [queue.pop(0)]
        else:
            cur_path = queue.pop(0)
        cur_row, cur_col = cur_path[-1]
        if (cur_row, cur_col) in explored:
            continue
        explored.add((cur_row, cur_col))
        if maze[cur_row, cur_col] == maze.legend.waypoint:
            return cur_path
        for item in maze.neighbors(cur_row, cur_col):
            if item not in explored:
                queue.append(cur_path + [item])

    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    explored = set()
    pq = PriorityQueue()
    start = maze.start
    end = maze.waypoints[0]
    h = abs(start[0] - end[0]) + abs(start[1] - end[1])
    g = 0
    pq.put((h + g, [start]))
    while pq:
        cur_path = pq.get()[1]
        cur_row, cur_col = cur_path[-1]
        for item in maze.neighbors(cur_row, cur_col):
            if maze[item[0], item[1]] == maze.legend.waypoint:
                explored.add(item)
                cur_path += [item]
                return cur_path
            if item not in explored:
                explored.add(item)
                h = abs(item[0] - end[0]) + abs(item[1] - end[1])
                g = len(cur_path)
                pq.put((h+g, cur_path+[item]))


    return []

def find_path(maze, start, end):
    explored = set()
    pq = PriorityQueue()
    h = abs(start[0] - end[0]) + abs(start[1] - end[1])
    g = 0
    pq.put((h + g, [start]))
    while pq:
        cur_path = pq.get()[1]
        cur_row, cur_col = cur_path[-1]
        for item in maze.neighbors(cur_row, cur_col):
            if (item[0], item[1]) == end:
                explored.add(item)
                cur_path += [item]
                return cur_path
            if item not in explored:
                explored.add(item)
                h = abs(item[0] - end[0]) + abs(item[1] - end[1])
                g = len(cur_path)
                pq.put((h+g, cur_path+[item]))

    return []

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """

    if maze.size.x < 10:
        waypoints = maze.waypoints
        queue = list(waypoints)
        start = maze.start
        farthest_path = 0
        for item in queue:
            path_length = len(find_path(maze,start,item))
            if path_length > farthest_path:
                farthest_path = path_length
                farthest_waypoint = item
        queue.remove(farthest_waypoint)
        smallest_distance = 9999999
        for item in queue:
            path_length = len(find_path(maze,start,item))
            if path_length < smallest_distance:
                smallest_distance = path_length
                closest_to_farthest = item
        queue.remove(closest_to_farthest)
        smallest_distance = 9999999
        for item in queue:
            path_length = len(find_path(maze,start,item))
            if path_length < smallest_distance:
                smallest_distance = path_length
                second_point = item
        queue.remove(second_point)
        first_point = queue[0]

        path1 = find_path(maze, maze.start, first_point)
        path1.pop()
        path2 = find_path(maze, first_point, second_point)
        path2.pop()
        path3 = find_path(maze, second_point, closest_to_farthest)
        path3.pop()
        path4 = find_path(maze, closest_to_farthest, farthest_waypoint)

        path = path1 + path2 + path3 + path4
        return path

    
    explored = set()
    pq = PriorityQueue()
    start = maze.start
    queue = list(maze.waypoints)
    farthest_distance = 0
    for waypoint in queue:
        h = abs(start[0] - waypoint[0]) + abs(start[1] - waypoint[1])
        if h > farthest_distance:
            farthest_distance = h
            farthest_waypoint = waypoint
    queue.remove(farthest_waypoint)
    smallest_distance = 9999999
    for waypoint in queue:
        h = abs(farthest_waypoint[0] - waypoint[0]) + abs(farthest_waypoint[1] - waypoint[1])
        if h < smallest_distance:
            smallest_distance = h
            closest_to_farthest = waypoint
    queue.remove(closest_to_farthest)
    smallest_distance = 9999999
    for waypoint in queue:
        h = abs(closest_to_farthest[0] - waypoint[0]) + abs(closest_to_farthest[1] - waypoint[1])
        if h < smallest_distance:
            smallest_distance = h
            second_point = waypoint
    queue.remove(second_point)
    first_point = queue[0]

    path1 = find_path(maze, maze.start, first_point)
    path1.pop()
    path2 = find_path(maze, first_point, second_point)
    path2.pop()
    path3 = find_path(maze, second_point, closest_to_farthest)
    path3.pop()
    path4 = find_path(maze, closest_to_farthest, farthest_waypoint)

    path = path1 + path2 + path3 + path4
    return path




    return []

def get_closest_dot(waypoints, current):
    smallest_distance = 999999
    for waypoint in waypoints:
        h = abs(current[0] - waypoint[0]) + abs(current[1] - waypoint[1])
        if h < smallest_distance:
            smallest_distance = h
            closest_dot = waypoint
    return closest_dot


def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    waypoints = maze.waypoints
    queue = list(waypoints)
    start = maze.start
    farthest_distance = 0
    for waypoint in queue:
        h = abs(start[0] - waypoint[0]) + abs(start[1] - waypoint[1])
        if h > farthest_distance:
            farthest_distance = h
            farthest_waypoint = waypoint
    queue.remove(farthest_waypoint)
    stack = []
    stack.append(farthest_waypoint)
    current = farthest_waypoint
    path = []
    while queue:
        temp = get_closest_dot(queue, current)
        stack.append(temp)
        current = temp
        queue.remove(temp)
    stack.append(maze.start)
    while stack:
        temp_start = stack.pop()
        if len(stack) == 0:
            break
        temp_end = stack[-1]
        path += find_path(maze, temp_start, temp_end)
        path.pop()
    path += [farthest_waypoint]
    return path



    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
