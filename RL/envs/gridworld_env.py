import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
#import matplotlib
#matplotlib.use("Qt4agg")
import matplotlib.pyplot as plt


#matplotlib.use('Qt5Agg')
from gym.envs.toy_text import discrete
#from skimage import color
#from skimage import io
from itertools import groupby
from operator import itemgetter
# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
#COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
#          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
#          5:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
#          4:[1.0,1.0,0.0]}
COLORS = {0:[0,0,0], 1:[128,128,128], \
          2:[0,0,255], 3:[0,255,0], \
          5:[255,0,0], 6:[255,0,255], \
          4:[255,255,0]}

class GridworldEnv(discrete.DiscreteEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'], #, 'state_pixels'],
        'video.frames_per_second': 1
    }
    num_env = 0
    plan='gridworldPlans/plan0.txt'

    def __init__(self):
        self._make(GridworldEnv.plan,rewards={0:0,3:1,4:1,5:-1,6:-1})

    def setPlan(self,plan,rewards):
        self._make(plan,rewards)

    def _make(self,plan,rewards):
        self.rewards=rewards
        self.nA = 4
        self.actions={0:[1,0],1:[-1,0],2:[0,-1],3:[0,1]}
        self.nbMaxSteps=1000
        # 0:South
        # 1:North
        # 2:West
        # 3:East
        self.action_space = spaces.Discrete(self.nA)

        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, plan)
        self.obs_shape = [128, 128, 3]
        self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = np.copy(self.start_grid_map)  # current grid map
        self.nbSteps=0
        self.rstates = {}
        #self.rewards={}
        #self.done = {}
        self.P=None
        self.nS=0
        self.startPos=self._get_agent_pos(self.current_grid_map)
        self.currentPos=copy.deepcopy(self.startPos)
        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env
        #self.verbose=True
        #self.render()
        #plt.pause(1)


    def getMDP(self):
        if self.P is None:
            self.P={}
            self.states={self.start_grid_map.dumps():0}
            #print("startpos"+str(self.startPos))
            self._getMDP(self.start_grid_map, self.startPos)
        return (self.states,self.P)



    def _getMDP(self,gridmap,state):
        #cur=self.states[str(gridmap)]
        cur = gridmap.dumps()
        succs={0:[],1:[],2:[],3:[]}
        self.P[cur]=succs
        #nstate=copy.deepcopy(state)
        self._exploreDir(gridmap,state,[1,0],0,2,3)
        self._exploreDir(gridmap, state, [-1, 0], 1, 2, 3)
        self._exploreDir(gridmap, state, [0, 1], 3, 0, 1)
        self._exploreDir(gridmap, state, [0, -1], 2, 0, 1)


    def _exploreDir(self,gridmap,state,dir,a,b,c):
        cur=gridmap.dumps()
        gridmap = copy.deepcopy(gridmap)
        succs=self.P[cur]
        nstate = copy.deepcopy(state)
        nstate[0]+=dir[0]
        nstate[1] += dir[1]

        #print("npos"+str(nstate))
        #print("grid"+str(gridmap.dumps()))
        if nstate[0]<gridmap.shape[0] and nstate[0]>=0 and nstate[1]<gridmap.shape[1] and nstate[1]>=0 and gridmap[nstate[0],nstate[1]]!=1:
                oldc=gridmap[nstate[0],nstate[1]]
                #print("oldc"+str(oldc))
                gridmap[state[0],state[1]] = 0
                gridmap[nstate[0],nstate[1]] = 2
                ng=gridmap.dumps()
                done = (oldc == 3 or oldc == 5)
                if ng in self.states:
                    ns=self.states[ng]
                else:
                    ns=len(self.states)
                    self.states[ng]=ns
                    if not done:
                        self._getMDP(gridmap,nstate)
                r=self.rewards[oldc]

                # if oldc==3 or oldc==4:
                #     r=1
                # if oldc == 5 or oldc == 6:
                #     r=-1

                succs[a].append((0.8, ng,r,done))
                succs[b].append((0.1, ng, r, done))
                succs[c].append((0.1, ng, r, done))
        else:
            succs[a].append((0.8,cur,self.rewards[0],False))
            succs[b].append((0.1, cur, self.rewards[0], False))
            succs[c].append((0.1, cur, self.rewards[0], False))




    def _get_agent_pos(self, grid_map):
        state = list(map(
                 lambda x:x[0] if len(x) > 0 else None,
                 np.where(grid_map == 2)
             ))
        return state


    def step(self, action):
        self.nbSteps += 1
        action = int(action)
        #print("state " + str(self.currentPos)+ "action:"+str(action))
        p = np.random.rand()
        if p<0.2:
            p = np.random.rand()
            if action==0 or action==1:
                if p < 0.5:
                    action=2
                else:
                    action=3
            else:
                if p < 0.5:
                    action=0
                else:
                    action=1
            #print("modified action:" + str(action))

        npos = (self.currentPos[0] + self.actions[action][0], self.currentPos[1] + self.actions[action][1])
        rr=-1*(self.nbSteps>self.nbMaxSteps)
        if npos[0] >= self.current_grid_map.shape[0] or npos[0] < 0 or npos[1] >= self.current_grid_map.shape[1] or npos[1] < 0 or self.current_grid_map[npos[0],npos[1]]==1:
            return (self.current_grid_map, self.rewards[0]+rr, self.nbSteps>self.nbMaxSteps, {})


        c=self.current_grid_map[npos]
        r = self.rewards[c]+rr

        done=(c == 3 or c == 5 or self.nbSteps>self.nbMaxSteps)
        # if c==3 or c==4:
        #     r=1
        # if c == 5 or c == 6:
        #     r = -1
        #print(str(self.current_grid_map))
        #print("cur"+str(self.currentPos)+ "npos "+str(npos)+" "+str(self.current_grid_map.shape[0])+" "+str(self.current_grid_map[npos]))

        self.current_grid_map[self.currentPos[0],self.currentPos[1]] = 0
        self.current_grid_map[npos[0],npos[1]] = 2
        self.currentPos = npos


        return (self.current_grid_map,r,done,{})





    def reset(self):
        self.currentPos = copy.deepcopy(self.startPos)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.nbSteps=0
        #self.render()

        return self.current_grid_map

    def _read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
            print(str(grid_map))
        grid_map_array = np.array(
            list(map(
                lambda x: list(map(
                    lambda y: int(y),
                    x.split(' ')
                )),
                grid_map
            ))
        )
        return grid_map_array



    def _gridmap_to_img(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.uint8)
        gs0 = int(observation.shape[0] / grid_map.shape[0])
        gs1 = int(observation.shape[1] / grid_map.shape[1])
        # print("grid="+str(grid_map))
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1] = np.array(COLORS[grid_map[i, j]])
        return observation

    def render(self, pause=0.00001, mode='human', close=False):
        if self.verbose == False:
            return
        img = self._gridmap_to_img(self.current_grid_map)
        #im = color.rgb2gray(img)
        #print(str(img))
        #print(img)
        #img=np.uint8(img*256)
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        # fig.canvas.flush_events()
        # plt.show(block=False)
        plt.pause(pause)
        # time.sleep(1)
        #ret=np.uint8(img)
        #print(ret)
        return img

    def _close_env(self):
        plt.close(1)
        return

    def changeState(self,gridmap):
        self.current_grid_map=gridmap
        self.currentPos=self._get_agent_pos(gridmap)
        self.render()
        #print(str(gridmap))
        #print(str(self.currentPos))

    # def step(self, action):
    #     action = int(action)
    #     print("state " + str(self.s)+ "action:"+str(action))
    #     #print()
    #
    #     olds=self.rstates[self.s]
    #
    #     s, r, d, p=discrete.DiscreteEnv.step(self,action)
    #
    #
    #     nxt_agent_state = self.rstates[s]
    #
    #     #print("nxt state " + str(self.s))
    #
    #     self.current_grid_map[olds] = 0
    #     self.current_grid_map[nxt_agent_state] = 4
    #
    #     self.render()
    #
    #     return (s, r, d, p)

    # def _make(self,plan):
    #     self.nA = 4
    #     # 0:South
    #     # 1:North
    #     # 2:West
    #     # 3:East
    #
    #     this_file_path = os.path.dirname(os.path.realpath(__file__))
    #     self.grid_map_path = os.path.join(this_file_path, plan)
    #     self.obs_shape = [128, 128, 3]
    #     self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
    #     self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
    #     self.states={}
    #     self.rstates = {}
    #     self.rewards={}
    #     self.done = {}
    #     self.P={}
    #     x=0
    #     for i in range(self.start_grid_map.shape[0]):
    #         for j in range(self.start_grid_map.shape[1]):
    #             if self.start_grid_map[i,j]!=1:
    #                 self.states[(i,j)]=x
    #                 self.done[x]=False
    #                 if self.start_grid_map[i,j]==4:
    #                     self.startState=x
    #                     self.rewards[x] = 0
    #                 if self.start_grid_map[i,j]==5:
    #                     self.rewards[x] = -1
    #                 if self.start_grid_map[i,j]==6:
    #                     self.rewards[x] = 1
    #                 if self.start_grid_map[i, j] == 3:
    #                     self.done[x] = True
    #                     self.rewards[x] = 1
    #                 if self.start_grid_map[i, j] == 2:
    #                     self.done[x] = True
    #                     self.rewards[x] = -1
    #                 if self.start_grid_map[i, j] == 0:
    #                     self.rewards[x] = 0
    #                 x += 1
    #     print(self.rewards)
    #     for (i,j) in self.states.keys():
    #         x=self.states[(i,j)]
    #         self.rstates[x]=(i,j)
    #         pi={}
    #         l=[(0.8,(i+1,j)),(0.1,(i,j+1)),(0.1,(i,j-1))]
    #         l=[(self.states[v],p) if v in self.states else (x,p) for (p,v) in l]
    #         l = [(k, list(list(zip(*g))[1])) for k, g in groupby(l, itemgetter(0))]
    #         l = [(sum(p),v,self.rewards[v],self.done[v]) for v,p in l]
    #         pi[0]=l
    #
    #         l = [(0.8, (i - 1, j)), (0.1, (i, j + 1)), (0.1, (i, j - 1))]
    #         l = [(self.states[v], p) if v in self.states else (x, p) for (p, v) in l]
    #         l = [(k, list(list(zip(*g))[1])) for k, g in groupby(l, itemgetter(0))]
    #         l = [(sum(p), v, self.rewards[v], self.done[v]) for v, p in l]
    #         pi[1] = l
    #
    #         l = [(0.8, (i, j-1)), (0.1, (i+1, j)), (0.1, (i-1, j))]
    #         l = [(self.states[v], p) if v in self.states else (x, p) for (p, v) in l]
    #         l = [(k, list(list(zip(*g))[1])) for k, g in groupby(l, itemgetter(0))]
    #         l = [(sum(p), v, self.rewards[v], self.done[v]) for v, p in l]
    #         pi[2] = l
    #
    #         l = [(0.8, (i, j + 1)), (0.1, (i + 1, j)), (0.1, (i - 1, j))]
    #         l = [(self.states[v], p) if v in self.states else (x,p) for (p, v) in l]
    #         l = [(k, list(list(zip(*g))[1])) for k, g in groupby(l, itemgetter(0))]
    #         l = [(sum(p), v, self.rewards[v], self.done[v]) for v, p in l]
    #         pi[3] = l
    #
    #         self.P[x]=pi
    #
    #     self.nS=len(self.states)
    #
    #     self.verbose=True
    #     self.isd = None
    #     self.lastaction = None  # for rendering
    #
    #     self.action_space = spaces.Discrete(self.nA)
    #     self.observation_space = spaces.Discrete(self.nS)
    #
    #     self.s=self.startState
    #
    #     self.seed()
    #
    #     GridworldEnv.num_env += 1
    #     self.this_fig_num = GridworldEnv.num_env
    #     # if self.verbose == True:
    #     #     self.fig = plt.figure(self.this_fig_num)
    #     #     plt.show(block=False)
    #     #     plt.axis('off')
    #     #     self.render()
    #     # self.reset()
    #







        # def __init__(self):
    #
    #     print("init grid world !! ")
    #     self._seed = 0
    #     self.actions = [0, 1, 2, 3]
    #     #0:South
    #     #1:North
    #     #2:West
    #     #3:East
    #
    #     #self.inv_actions = [1, 0, 3, 2]
    #     self.action_space = spaces.Discrete(4)
    #     self.action_pos_dict = {0:[-1, 0], 1:[1,0], 2:[0,-1], 3:[0,1]}
    #
    #     # ''' set observation space '''
    #     self.obs_shape = [128, 128, 3]  # observation space shape
    #     self.observation_space = spaces.Box(low=-1, high=1, shape=self.obs_shape, dtype=np.float32)
    #     #
    #     ''' initialize system state '''
    #     this_file_path = os.path.dirname(os.path.realpath(__file__))
    #     self.grid_map_path = os.path.join(this_file_path, 'plan0.txt')
    #     self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
    #     self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
    #     # self.observation = self._gridmap_to_observation(self.start_grid_map)
    #     self.grid_map_shape = self.start_grid_map.shape
    #
    #     ''' agent state: start, target, current state '''
    #     self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state(self.start_grid_map)
    #     self.agent_state = copy.deepcopy(self.agent_start_state)
    #
    #     ''' set other parameters '''
    #     #self.restart_once_done = False  # restart or not once done
    #     self.verbose = False # to show the environment or not
    #
    #     GridworldEnv.num_env += 1
    #     self.this_fig_num = GridworldEnv.num_env
    #     if self.verbose == True:
    #         self.fig = plt.figure(self.this_fig_num)
    #         plt.show(block=False)
    #         plt.axis('off')
    #         self.render()
    #
    # def step(self, action):
    #     ''' return next observation, reward, finished, success '''
    #     action = int(action)
    #     print("action:"+str(action))
    #     print("state "+str(self.agent_state))
    #     info = {}
    #     info['success'] = False
    #     p=np.random.rand()
    #     if p<0.2:
    #         p = np.random.rand()
    #         if action==0 or action==1:
    #             if p < 0.5:
    #                 action=2
    #             else:
    #                 action=3
    #         else:
    #             if p < 0.5:
    #                 action=0
    #             else:
    #                 action=1
    #         print("modified action:" + str(action))
    #
    #     nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
    #                         self.agent_state[1] + self.action_pos_dict[action][1])
    #
    #     if nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]:
    #         info['success'] = False
    #         return (self.agent_state, 0, False, info)
    #     if nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]:
    #         info['success'] = False
    #         return (self.agent_state, 0, False, info)
    #     # successful behavior
    #     new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
    #
    #     if new_color == 1:  # gray
    #         info['success'] = False
    #         return (self.agent_state, 0, False, info)
    #     info['success'] = True
    #
    #     print("move")
    #     self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
    #     self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
    #     self.agent_state = copy.deepcopy(nxt_agent_state)
    #
    #     self.render()
    #
    #     if new_color == 2: #fail
    #         print(str("crash!"))
    #         return (self.agent_state, -1, True, info)
    #
    #     if new_color == 3:  # win
    #         print(str("win!"))
    #         return (self.agent_state, 1, True, info)
    #
    #     return (self.agent_state, 0, False, info)



    # def _get_agent_start_target_state(self, start_grid_map):
    #     start_state = None
    #     target_state = None
    #     start_state = list(map(
    #         lambda x:x[0] if len(x) > 0 else None,
    #         np.where(start_grid_map == 4)
    #     ))
    #     target_state = list(map(
    #         lambda x:x[0] if len(x) > 0 else None,
    #         np.where(start_grid_map == 3)
    #     ))
    #     if start_state == [None, None] or target_state == [None, None]:
    #         sys.exit('Start or target state not specified')
    #     return start_state, target_state


 
    # def change_start_state(self, sp):
    #     ''' change agent start state '''
    #     ''' Input: sp: new start state '''
    #     if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
    #         _ = self.reset()
    #         return True
    #     elif self.start_grid_map[sp[0], sp[1]] != 0:
    #         return False
    #     else:
    #         s_pos = copy.deepcopy(self.agent_start_state)
    #         self.start_grid_map[s_pos[0], s_pos[1]] = 0
    #         self.start_grid_map[sp[0], sp[1]] = 4
    #         self.current_grid_map = copy.deepcopy(self.start_grid_map)
    #         self.agent_start_state = [sp[0], sp[1]]
    #         self.observation = self._gridmap_to_observation(self.current_grid_map)
    #         self.agent_state = copy.deepcopy(self.agent_start_state)
    #         self.reset()
    #         self._render()
    #     return True
    #
    #
    # def change_target_state(self, tg):
    #     if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
    #         _ = self.reset()
    #         return True
    #     elif self.start_grid_map[tg[0], tg[1]] != 0:
    #         return False
    #     else:
    #         t_pos = copy.deepcopy(self.agent_target_state)
    #         self.start_grid_map[t_pos[0], t_pos[1]] = 0
    #         self.start_grid_map[tg[0], tg[1]] = 3
    #         self.current_grid_map = copy.deepcopy(self.start_grid_map)
    #         self.agent_target_state = [tg[0], tg[1]]
    #         self.observation = self._gridmap_to_observation(self.current_grid_map)
    #         self.agent_state = copy.deepcopy(self.agent_start_state)
    #         self.reset()
    #         self._render()
    #     return True
    
    # def get_agent_state(self):
    #     ''' get current agent state '''
    #     return self.agent_state
    #
    # def get_start_state(self):
    #     ''' get current start state '''
    #     return self.agent_start_state
    #
    # def get_target_state(self):
    #     ''' get current target state '''
    #     return self.agent_target_state

    # def _jump_to_state(self, to_state):
    #     ''' move agent to another state '''
    #     info = {}
    #     info['success'] = True
    #     if self.current_grid_map[to_state[0], to_state[1]] == 0:
    #         if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 4:
    #             self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
    #             self.current_grid_map[to_state[0], to_state[1]] = 4
    #             self.observation = self._gridmap_to_observation(self.current_grid_map)
    #             self.agent_state = [to_state[0], to_state[1]]
    #             self._render()
    #             return (self.observation, 0, False, info)
    #         if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 6:
    #             self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 2
    #             self.current_grid_map[to_state[0], to_state[1]] = 4
    #             self.observation = self._gridmap_to_observation(self.current_grid_map)
    #             self.agent_state = [to_state[0], to_state[1]]
    #             self._render()
    #             return (self.observation, 0, False, info)
    #         if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 7:
    #             self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 3
    #             self.current_grid_map[to_state[0], to_state[1]] = 4
    #             self.observation = self._gridmap_to_observation(self.current_grid_map)
    #             self.agent_state = [to_state[0], to_state[1]]
    #             self._render()
    #             return (self.observation, 0, False, info)
    #     elif self.current_grid_map[to_state[0], to_state[1]] == 4:
    #         return (self.observation, 0, False, info)
    #     elif self.current_grid_map[to_state[0], to_state[1]] == 1:
    #         info['success'] = False
    #         return (self.observation, 0, False, info)
    #     elif self.current_grid_map[to_state[0], to_state[1]] == 3:
    #         self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
    #         self.current_grid_map[to_state[0], to_state[1]] = 7
    #         self.agent_state = [to_state[0], to_state[1]]
    #         self.observation = self._gridmap_to_observation(self.current_grid_map)
    #         self._render()
    #         if self.restart_once_done:
    #             self.observation = self.reset()
    #             return (self.observation, 1, True, info)
    #         return (self.observation, 1, True, info)
    #     else:
    #         info['success'] = False
    #         return (self.observation, 0, False, info)


    
    # def jump_to_state(self, to_state):
    #     a, b, c, d = self._jump_to_state(to_state)
    #     return (a, b, c, d)
