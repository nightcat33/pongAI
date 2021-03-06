from random import uniform
import math
import numpy as np
import time
import random
import csv

BOARD_SIZE = 12
PADDLE_HEIGHT = 0.2

MOVE_UP = 0
MOVE_DOWN = 1
NOTHING = 2

EXPLORE = 500000
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05

#  two states be considered the same if the ball lies within the same cell in this table.
# PADDLE 

PLAYER1 = 0
PLAYER2 = 1

class State(object):
    def __init__(self):
        # init state
        # left corner is (0,0)
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.velocity_x = 0.03
        self.velocity_y = 0.01
        self.paddle_y = 0.5 - PADDLE_HEIGHT / 2
        self.game_end = False
        self.numberOfBounces = 0
        self.paddle_x = 1

        self.player2_y = 0.5 - PADDLE_HEIGHT / 2
        self.player2_x = 0

        self.win = False


    def movePaddle(self, action, player):
        if player == PLAYER1:
            if action == MOVE_UP:
                self.paddle_y += 0.04
            elif action == MOVE_DOWN:
                self.paddle_y -= 0.04
            elif action == NOTHING:
                self.paddle_y = self.paddle_y

            # off screen check
            if self.paddle_y < 0:
                self.paddle_y = 0
            elif self.paddle_y > 1 - PADDLE_HEIGHT:
                self.paddle_y = 1 - PADDLE_HEIGHT

        elif player == PLAYER2:
            if action == MOVE_UP:
                self.player2_y += 0.02
            elif action == MOVE_DOWN:
                self.player2_y -= 0.02
            elif action == NOTHING:
                self.player2_y = self.player2_y

            # off screen check
            if self.player2_y < 0:
                self.player2_y = 0
            elif self.player2_y > 1 - PADDLE_HEIGHT:
                self.player2_y = 1 - PADDLE_HEIGHT


    def updateState(self):
        # increment speed
        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y
        reward = 0
        if self.ball_y < 0:
            # the ball is off the top of the screen
            self.ball_y = -self.ball_y
            self.velocity_y = -self.velocity_y
        elif self.ball_y > 1:
            # the ball is off the bottom of the screen
            self.ball_y = 2 - self.ball_y
            self.velocity_y = -self.velocity_y

       
        if self.ball_x >= 1 and self.ball_y >= self.paddle_y and self.ball_y <= self.paddle_y + PADDLE_HEIGHT:
            # bounce off the paddle
            self.ball_x = 2 * self.paddle_x - self.ball_x
            # randomize the speed
            while True:
                self.velocity_x = -self.velocity_x + uniform(-0.015, 0.015)
                if abs(self.velocity_x) > 0.03:
                    break

            self.velocity_y += uniform(-0.03, 0.03)
            self.numberOfBounces += 1
            reward = 1
        elif self.ball_x <= 0 and self.ball_y >= self.player2_y and self.ball_y <= self.player2_y + PADDLE_HEIGHT:
            # bounce off the paddle
            self.ball_x = 2 * self.player2_x - self.ball_x
            # randomize the speed
            while True:
                self.velocity_x = -self.velocity_x + uniform(-0.015, 0.015)
                if abs(self.velocity_x) > 0.03:
                    break

            self.velocity_y += uniform(-0.03, 0.03)
            reward = -1

        elif self.ball_x > 1:
            reward = -1
            self.game_end = True
            self.win = False

        elif self.ball_x < 0:
            reward = 2
            self.game_end = True
            self.win = True

        return reward

    def discretizeState(self):

        discretizdBallX = math.floor(self.ball_x * (BOARD_SIZE - 1))
        discretizeBallY = math.floor(self.ball_y * (BOARD_SIZE - 1))

        discretizeVelocityX = None
        discretizeVelocityY = None
        if self.velocity_x < 0:
            discretizeVelocityX = -1
        else:
            discretizeVelocityX = 1

        if self.velocity_y < 0:
            discretizeVelocityY = -1
        else:
            discretizeVelocityY = 1

        if abs(self.velocity_y) < 0.015:
            discretizeVelocityY = 0

        discrete_paddle = math.floor(BOARD_SIZE * self.paddle_y / (1 - PADDLE_HEIGHT))

        if self.paddle_y == 1 - PADDLE_HEIGHT:
            discrete_paddle = 11

        discrete_paddle2 = math.floor(BOARD_SIZE * self.player2_y / (1 - PADDLE_HEIGHT))

        if self.player2_y == 1 - PADDLE_HEIGHT:
            discrete_paddle2 = 11

        return (discretizdBallX, discretizeBallY, discretizeVelocityX, discretizeVelocityY, discrete_paddle, discrete_paddle2)

    def reset(self):
        self.__init__()

    def endOfGame(self):
        return self.game_end

    def getNumberOfBounces(self):
        return self.numberOfBounces

    def isWin(self):
        return self.win


class Agent(object):

    def __init__(self, C, init_epsilon, final_epsilon, discount_factor):

        self.action_utility = [
            [
                [
                    [
                        [
                            {'utility': 0, 'frequency': 0} for i in range(3)
                        ] for i in range(12)
                    ] for i in range(3)
                ] for i in range(2)
            ] for i in range(BOARD_SIZE * BOARD_SIZE)
        ]

        self.C = C
        self.discount_factor = discount_factor
        self.curr_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.init_epsilon = init_epsilon

    def learning_rate_decay(self, N):
        return self.C / (self.C + N)

    def eplison_decay(self):
        if self.curr_epsilon > self.final_epsilon:
            self.curr_epsilon -= (self.init_epsilon - self.final_epsilon) / EXPLORE

    # choose next action from current state
    def strategy(self, currState):
        # epsilon greedy
        self.eplison_decay()
        action = None
        # print("epsilon %f" % self.curr_epsilon)
        if random.random() < self.curr_epsilon:
            # explore
            action = np.random.randint(0,3)

        else:
            # exploit
            (ball_x, ball_y, velocity_x, velocity_y, paddle_y, _) = currState

            (velocity_x_idx, velocity_y_idx) = self.getVelocityIndex(velocity_x, velocity_y)


            # get utility of each action
            utilityUp = self.action_utility[ball_y * BOARD_SIZE + ball_x][velocity_x_idx][velocity_y_idx][paddle_y][MOVE_UP]['utility']
            utilityDown = self.action_utility[ball_y * BOARD_SIZE + ball_x][velocity_x_idx][velocity_y_idx][paddle_y][MOVE_DOWN]['utility']
            utilityNothing = self.action_utility[ball_y * BOARD_SIZE + ball_x][velocity_x_idx][velocity_y_idx][paddle_y][NOTHING]['utility']

            utilityList = [utilityUp, utilityDown, utilityNothing]

            utilityMax = -math.inf
            for i in range(3):
                if utilityList[i] > utilityMax:
                    utilityMax = utilityList[i]
                    action = i

        return action

    def updateUtility(self, currState, nextState, action, reward):
        (ball_x, ball_y, velocity_x, velocity_y, paddle_y, _) = currState
        (velocity_x_idx, velocity_y_idx) = self.getVelocityIndex(velocity_x, velocity_y)

        self.action_utility[ball_y * BOARD_SIZE + ball_x][velocity_x_idx][velocity_y_idx][paddle_y][action]['frequency'] += 1

        nextAction = self.strategy(nextState)

        maxQ = None
        if reward == -1:
            maxQ = 0
        else:
            (next_ball_x, next_ball_y, next_velocity_x, next_velocity_y, next_paddle_y, _) = nextState
            (next_velocity_x_idx, next_velocity_y_idx) = self.getVelocityIndex(next_velocity_x, next_velocity_y)
            maxQ = self.action_utility[next_ball_y * BOARD_SIZE + next_ball_x][next_velocity_x_idx][next_velocity_y_idx][next_paddle_y][nextAction]['utility']

        self.action_utility[ball_y * BOARD_SIZE + ball_x][velocity_x_idx][velocity_y_idx][paddle_y][action]['utility'] += \
            self.learning_rate_decay(self.action_utility[ball_y * BOARD_SIZE + ball_x][velocity_x_idx][velocity_y_idx][paddle_y][action]['frequency']) \
            * (reward + self.discount_factor * maxQ - self.action_utility[ball_y * BOARD_SIZE + ball_x][velocity_x_idx][velocity_y_idx][paddle_y][action]['utility'])

        return

    def getVelocityIndex(self, velocity_x, velocity_y):
        velocity_x_idx = None
        velocity_y_idx = None

        if velocity_x == 1:
            velocity_x_idx = 0
        else:
            velocity_x_idx = 1

        if velocity_y == 1:
            velocity_y_idx = 0
        elif velocity_y == 0:
            velocity_y_idx = 1
        else:
            velocity_y_idx = 2

        return (velocity_x_idx, velocity_y_idx)

    def enableTesting(self):
        self.curr_epsilon = 0.049998

    def stupidStrategy(self, currState):
        (ball_x, ball_y, velocity_x, velocity_y, paddle_y, paddle_y2) = currState

        if paddle_y2 == ball_y:
            return NOTHING
        elif paddle_y2 < ball_y:
            return MOVE_DOWN
        else:
            return MOVE_UP



class stupidSunJiaRun(object):

    def __init__(self, C, init_epsilon, final_epsilon, discount_factor):
        self.state = State()
        self.agent = Agent(C, init_epsilon, final_epsilon, discount_factor)
        self.total = 0
        self.winnings = 0

    def gameByStep(self):

        if self.state.endOfGame():
            # print("Game Over!")
            numberOfBounces = self.state.getNumberOfBounces()

            print("# of bounces: %d." % numberOfBounces)
            self.total += numberOfBounces

            if self.state.isWin():
                self.winnings += 1

            return True
        else:
            currState = self.state.discretizeState()
            currAction = self.agent.strategy(currState)
            currAction_player2 = self.agent.stupidStrategy(currState)

            # move the paddle
            self.state.movePaddle(currAction, PLAYER1)
            self.state.movePaddle(currAction_player2, PLAYER2)

            reward = self.state.updateState()

            nextState = self.state.discretizeState()

            self.agent.updateUtility(currState, nextState, currAction, reward)
            return False


    def train(self, episodes):
        begin = time.time()
        for i in range(episodes):
            if i % 1000 == 0:
                print("Training %d episodes using %f seconds." % (i, time.time()-begin))

            while not self.gameByStep():
                continue
            self.state.reset()

        end_time = time.time()

        time_cost = end_time - begin
        winnings_percentage = self.winnings / float(episodes)
        average = self.total / float(episodes)

        print("Average bounces is %f." % average)
        print("Time cost is %f." % time_cost)
        print("Winning percentage is %f." % winnings_percentage)
        print()
        
        row = [average, winnings_percentage, time_cost, self.agent.C, self.agent.discount_factor]
        with open('multiple_train_results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.total = 0
        self.winnings = 0


    def test(self):
        self.agent.enableTesting()
        
        begin = time.time()
        for i in range(1000):
            while not self.gameByStep():
                continue
            self.state.reset()

        end_time = time.time()

        time_cost = end_time - begin
        average = self.total / float(1000)
        winnings_percentage = self.winnings / float(1000)

        print("Average bounces is %f." % average)
        print("Time cost is %f." % time_cost)
        print("Winning percentage is %f." % winnings_percentage)
        row = [average, winnings_percentage, time_cost, self.agent.C, self.agent.discount_factor]

        with open('multiple_test_results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.total = 0
        self.winnings = 0

if __name__ == "__main__":

    for C in [10, 15, 20, 25, 35, 40]:
        for discount_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            game = stupidSunJiaRun(C, INITIAL_EPSILON, FINAL_EPSILON, discount_factor)
            game.train(100000)
            game.test()






