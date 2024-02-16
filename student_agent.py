# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time 
from queue import Queue
import math
import random
INF = 10**6


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.maxUtilMoves = []
        self.interrupted = False
        self.maxWalls =0
    
    
    def filterMoves(self,my_pos,moves,chess_board,max_step):
        filteredList=[]
        for move in moves:
            (r,c),w = move
            connections =0
            if w ==0:
                if chess_board[r-1,c,1]:
                    connections = connections+1
                if chess_board[r-1,c,3]:
                    connections = connections+1
                if c+1<chess_board.shape[0] and chess_board[r-1,c+1,2]:
                    connections = connections+1
                if c-1>=0 and chess_board[r-1,c-1,2]:
                    connections = connections+1
                if chess_board[r,c,1]:
                    connections = connections+1
                if chess_board[r,c,3]:
                    connections = connections+1
            elif w==1:
                if chess_board[r,c,0]:
                    connections = connections+1
                if chess_board[r,c,2]:
                    connections = connections+1
                if c+1<chess_board.shape[0] and chess_board[r,c+1,2]:
                    connections = connections +1
                if c+1<chess_board.shape[0] and chess_board[r,c+1,0]:
                    connections = connections +1
                if r+1<chess_board.shape[0] and chess_board[r+1,c,1]:
                    connections = connections +1
                if r-1>=0 and chess_board[r-1,c,1]:
                    connections = connections +1
            elif w==2:
                if chess_board[r,c,1]:
                    connections = connections+1
                if chess_board[r,c,3]:
                    connections = connections+1
                if c+1<chess_board.shape[0] and chess_board[r,c+1,2]:
                    connections = connections+1
                if c-1>=0 and chess_board[r,c-1,2]:
                    connections = connections+1
                if r+1<chess_board.shape[0] and chess_board[r+1,c,3]:
                    connections = connections+1
                if r+1<chess_board.shape[0] and chess_board[r+1,c,1]:
                    connections = connections+1
            elif w==3:
                if chess_board[r,c,0]:
                    connections = connections +1
                if chess_board[r,c,2]:
                    connections = connections +1
                if r-1>=0 and chess_board[r-1,c,3]:
                    connections = connections +1
                if r+1<chess_board.shape[0] and chess_board[r-1,c,3]:
                    connections = connections +1
                if c-1>=0 and chess_board[r,c-1,2]:
                    connections = connections +1
                if c-1>=0 and chess_board[r,c-1,0]:
                    connections = connections +1
            if connections>0:
                filteredList.append(move)
            elif self.countManhattan(my_pos, move[0])==max_step:
                filteredList.append(move)
        return filteredList
                
    def alpha_beta(self,my_pos,adv_pos,chess_board,max_step,initial_limit, depth, alpha, beta, maximizing_player,start_time):
        if (time.time()-start_time>=1.90):
            self.interrupted = True
            return 0
        
        if (depth ==0 ):
            return self.evalFunction(chess_board,my_pos,adv_pos,max_step,0)  
        if self.isTerminalState(chess_board, my_pos,adv_pos):
            return self.evalFunction(chess_board,my_pos,adv_pos,max_step,0)
        
        
        if maximizing_player:

            
            value = -(10**9)
            moves = self.getValidMoves(chess_board,my_pos,adv_pos,max_step)
            
            if self.countManhattan(my_pos, adv_pos)>max_step:
                moves= self.filterMoves(my_pos,moves,chess_board,max_step)

            random.shuffle(moves)
            for move in moves:
                (r,c),w = move
                chess_board[r,c,w] = True
                temp = self.alpha_beta((r,c),adv_pos,chess_board,max_step,initial_limit, depth - 1, alpha, beta, False,start_time)
                value = max(value, temp)
                alpha = max(alpha, value)
                chess_board[r,c,w] = False
                if depth == initial_limit:
                    self.maxUtilMoves.append(((move),temp))
                if beta <= alpha:
                    break  # Beta cut-off
            return value
        else:
            value = (10**9)
            moves = self.getValidMoves(chess_board,adv_pos,my_pos,max_step)
            
            if self.countManhattan(my_pos, adv_pos)>max_step:
                moves= self.filterMoves(adv_pos,moves,chess_board,max_step)
            
            random.shuffle(moves)
            for move in moves:
                (r,c),w = move
                chess_board[r,c,w] = True
                value = min(value, self.alpha_beta(my_pos,(r,c),chess_board,max_step,initial_limit, depth - 1, alpha, beta, True,start_time))
                chess_board[r,c,w] = False
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cut-off
            return value

    
    def countManhattan(self,my_pos, adv_pos):
        return abs(my_pos[0]-adv_pos[0]) + abs(my_pos[1]-adv_pos[1])

    def countNumWalls(self,chess_board):
        dim = chess_board.shape[0]

        count = 0
        for r in range(0,dim-1):
            for c in range(0,dim-1):
            
                if chess_board[r,c,1]:
                    count = count +1
                if chess_board[r,c,2]:
                    count = count+1    
        
        for c in range(0,dim-1):
            if chess_board[dim-1,c,1]:
                count = count+1
        for r in range(0,dim-1):
            if chess_board[r,dim-1,2]:
                count = count +1

        return count


    def isTerminalState(self,chess_board, my_pos,adv_pos):
        'Run dfs from my agent to enemy, if cant reach then terminal'
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        q = Queue()
        q.put(my_pos)
        
        chessboardDim = chess_board.shape[0]
        visited =np.zeros((chessboardDim, chessboardDim), dtype=bool)
        visited[my_pos[0],my_pos[1]] = True
        
        while (not q.empty()):
            r,c = q.get()
            for d in range(0,4):
                'If there is a wall or we have already visisted this position'
                if chess_board[r,c,d] or visited[r+moves[d][0],c+moves[d][1]]:
                    continue
                
                'We have reached enemy so its not a terminal state'
                if (r+moves[d][0],c+moves[d][1]) == adv_pos:
                    return False
                
                'Add this position to the queue and mark as visisted'
                q.put((r+moves[d][0],c+moves[d][1]))
                visited[r+moves[d][0],c+moves[d][1]] = True
        'We couldnt reach the enemy so it must be terminal state'
        

        return True
    

    def cellCost(self,chess_board,my_pos,adv_pos,max_step):
        'BFS to calculate the number of moves to reach all available cells'      
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        chessboardDim = chess_board.shape[0]
        visited =np.zeros((chessboardDim, chessboardDim), dtype=bool)
        visited[my_pos[0],my_pos[1]] = True
        
        costs = np.full((chessboardDim, chessboardDim), INF, dtype=int)

        costs[my_pos[0],my_pos[1]] = 0
        
        q = Queue()
        q.put((my_pos,0))

        while(not q.empty()):
            (r,c),i = q.get()
            for d in range(0,4):
                
                if chess_board[r,c,d] or adv_pos == (r+moves[d][0],c+moves[d][1]) or visited[r+moves[d][0],c+moves[d][1]]:
                # chess_board[r,c,d] or visited[r+moves[d][0],c+moves[d][1]]:
                    continue
                q.put(((r+moves[d][0],c+moves[d][1]),i+1))
                visited[r+moves[d][0],c+moves[d][1]] = True
                costs[r+moves[d][0],c+moves[d][1]] = (i+1)//max_step
                if (i+1)%max_step != 0:
                    costs[r+moves[d][0],c+moves[d][1]]  = costs[r+moves[d][0],c+moves[d][1]]+1

        return costs
    
    def evalFunction(self,chess_board,my_pos,adv_pos,max_step,depth):
        
        myCosts = self.cellCost(chess_board, my_pos, adv_pos,max_step)
        advCosts = self.cellCost(chess_board,adv_pos,my_pos,max_step)

        utility =0

        for r in range(0,chess_board.shape[0]):
            for c in range(0,chess_board.shape[1]):

                if (myCosts[r,c] <advCosts[r,c]):
                    utility+=1
                elif (myCosts[r,c] >advCosts[r,c]):
                    utility-=1
        if self.isTerminalState(chess_board, my_pos,adv_pos):
            if (utility > 0):
                return INF+utility
            elif (utility<0):
                return -INF+utility
            return 0
        return utility
    

    def distanceToAdv(self,chess_board, my_pos,adv_pos):
        'Run bfs from my agent to enemy, if cant reach then terminal'
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        q = Queue()
        q.put(my_pos)
        
        chessboardDim = chess_board.shape[0]
        matrix = np.full((chessboardDim, chessboardDim), INF, dtype= int)
        matrix[my_pos[0], my_pos[1]] = 0 
        visited =np.zeros((chessboardDim, chessboardDim), dtype=bool)
        visited[my_pos[0],my_pos[1]] = True
        
        while (not q.empty()):
            r,c = q.get()
            for d in range(0,4):
                'If there is a wall or we have already visisted this position'
                if chess_board[r,c,d] or visited[r+moves[d][0],c+moves[d][1]]:
                    continue
                
                'We have reached enemy so its not a terminal state'
                if (r+moves[d][0],c+moves[d][1]) == adv_pos:
                    return matrix[r,c]+1
                
                'Add this position to the queue and mark as visisted'
                q.put((r+moves[d][0],c+moves[d][1]))
                visited[r+moves[d][0],c+moves[d][1]] = True
                matrix[r+moves[d][0],c+moves[d][1]]  = matrix[r,c] + 1
                
        'We couldnt reach the enemy so it must be terminal state'
        return INF
    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        limit =1
        lastMoves=[]

        self.maxWalls = (chess_board.shape[0]- 1) * ( chess_board.shape[0]) *2 

 
        while(time.time()-start_time<1.90):
            self.maxUtilMoves=[]
            self.alpha_beta(my_pos,adv_pos,chess_board,max_step,limit, limit,-(10**9),10**9, True,start_time)
            if self.interrupted == True:
                self.maxUtilMoves = lastMoves
                #print("depth="+str(limit-1))
                self.interrupted = False
            lastMoves = self.maxUtilMoves
            limit+=1        
        maxIndex = 0
       
        for i in range(1,len(self.maxUtilMoves)):
            if self.maxUtilMoves[i][1]> self.maxUtilMoves[maxIndex][1]:
                maxIndex = i
                
        best_moves = []
        for t in self.maxUtilMoves:
            if (t[1] == self.maxUtilMoves[maxIndex][1]):
                best_moves.append(t)
         
        
        chosenMove = random.choice(best_moves)
        filteredList =[]
        distances = []
        if (chosenMove[1] ==0):
            for move in best_moves:
                chess_board[move[0][0][0],move[0][0][1],move[0][1]] = True
                distance = self.distanceToAdv(chess_board, (move[0][0][0],move[0][0][1]),adv_pos)
                if distance!=INF:
                    if distance <=max_step+1:
                        (r,c),w = move[0]
                        
                        trappedCount =0
                        for d in range(0,4):
                            if chess_board[r,c,w] == True:
                                trappedCount = trappedCount+1
                        if trappedCount ==3:
                            continue
                    filteredList.append(move)
                    distances.append(distance)
                chess_board[move[0][0][0],move[0][0][1],move[0][1]] = False
        
            if len(filteredList)==0:
                return chosenMove[0]
        
            maxIndex = 0
            for i in range(1,len(distances)):
                if distances[i]< distances[maxIndex]:
                    maxIndex = i
            time_taken = time.time()-start_time
            '''print("My AI's turn took ", time_taken, "seconds.")'''
            return filteredList[maxIndex][0]

        
        time_taken = time.time()-start_time
        '''print("My AI's turn took ", time_taken, "seconds.")'''
        
    
        for move in best_moves:
            chess_board[move[0][0][0],move[0][0][1],move[0][1]] = True
            distance = self.distanceToAdv(chess_board, (move[0][0][0],move[0][0][1]),adv_pos)
            if distance <=max_step+1:
                (r,c),w = move[0]
                
                trappedCount =0
                for d in range(0,4):
                    if chess_board[r,c,w] == True:
                        trappedCount = trappedCount+1
                if trappedCount ==3:
                    continue
            filteredList.append(move)
            distances.append(distance)
            chess_board[move[0][0][0],move[0][0][1],move[0][1]] = False
        
        if len(filteredList)==0:
            return chosenMove[0]
        maxIndex = 0
        for i in range(1,len(distances)):
            if distances[i]< distances[maxIndex]:
                maxIndex = i
        return filteredList[maxIndex][0]
    'Returns a list a valid moves for the given parameters'
    def getValidMoves(self,chess_board,my_pos,adv_pos,max_step):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        plays =[]
        q = Queue()
        q.put((my_pos,0))
        
        chessboardDim = chess_board.shape[0]
        visited =np.zeros((chessboardDim, chessboardDim), dtype=bool)
        visited[my_pos[0],my_pos[1]] = True
        move =0
        
        for w in range(0,4):
            if not chess_board[my_pos[0],my_pos[1],w]:
                plays.append(((my_pos[0],my_pos[1]),w))
        
        while (not q.empty() and q.queue[0][1]<max_step):
            (r,c),i = q.get()
            for d in range(0,4):
                if chess_board[r,c,d] or adv_pos == (r+moves[d][0],c+moves[d][1]) or visited[r+moves[d][0],c+moves[d][1]]:
                    continue
                q.put(((r+moves[d][0],c+moves[d][1]),i+1))
                visited[r+moves[d][0],c+moves[d][1]] = True
                for w in range (0,4):
                    'Can place a wall'
                    if not chess_board[r+moves[d][0],c+moves[d][1],w]:
                        'We can move in direction d and place a wall in direction w'
                        plays.append(((r+moves[d][0],c+moves[d][1]),w))

        return plays
class move:
    def __init__(self,pos,direction):
        self.pos = pos
        self.direction = direction
