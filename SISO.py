# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:05:02 2022

@author: DHEERAJ
"""


import casadi as cd
import matplotlib.pyplot as plt
import numpy as np 
from time import time
from simulation_code import simulate



T = 0.2
N= 3

x_init = 0.0
y_init = 0.0
theta_init = 0.0

x_target, y_target, theta_target = 1.5, 1.0, 0

v_max = 0.6;
v_min = -v_max;
omega_max = cd.pi/4; 
omega_min = -omega_max;

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = cd.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = cd.horzcat(
        u[:, 1:],
        cd.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())
                    
x = cd.SX.sym('x')
y = cd.SX.sym('y')
theta = cd.SX.sym('theta')

states = cd.vertcat(x,y,theta)
n_states = states.numel()

v = cd.SX.sym('v')
w = cd.SX.sym('w')

controls = cd.vertcat(v,w)
n_controls = controls.numel()

s_dot = cd.vertcat(v*cd.cos(theta),v*cd.sin(theta),w)
# s_dot = cd.vec(s_dot)

f = cd.Function('f', [states,controls],[s_dot])

U = cd.SX.sym('U',n_controls,N)
P = cd.SX.sym('P',n_states+n_states)

#States over the optimization problem
X = cd.SX.sym('X',n_states,N+1)

X[:,0] = P[:3]
for k in range(N):
    st = X[:,k]
    con = U[:,k]
    f_value = f(st,con)
    st_next = st + (T*f_value)
    X[:,k+1] = st_next

ff = cd.Function('ff',[U,P],[X])

obj = 0 # objective function
g= [] # constraints vector

# Q = cd.DM(3,3)
# R = cd.DM(2,2)

Q = np.diag([1,5,0.1])
R = np.diag([0.5,0.05])

for k in range(N):
    st = X[:,k]
    con = U[:,k]
    obj = obj + cd.mtimes([(st-P[3:]).T,Q,(st-P[3:])]) + cd.mtimes([con.T,R,con])

for k in range(N+1):
    g = cd.vertcat(g,X[0,k])
    g = cd.vertcat(g,X[1,k])

# making decision variables one single column vector
OPT_variables = cd.reshape(U,2*N,1)

#Objective function with decision variables and constraints
nlp_prob = {
    'f': obj,
    'x': OPT_variables,
    'g': g,
    'p': P
}

#Options for Optimization
opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}


#Object for NLP solving
solver = cd.nlpsol('solver', 'ipopt', nlp_prob, opts)


lbx = cd.DM.zeros((n_controls*N, 1))
ubx = cd.DM.zeros((n_controls*N, 1))

lbx[0: n_states*(N-1)] = v_min     # X lower bound
lbx[1: n_states*(N-1)] = omega_min     # Y lower bound

ubx[0: n_states*(N-1)] = v_max      # X upper bound
ubx[1: n_states*(N-1)] = omega_max      # Y upper bound


#Constraints
args = {
        'lbx': lbx,
        'ubx': ubx,
        'lbg': -2,
        'ubg': 2,
        'p': [],
        'x0': 0
        }


t0=0
x0 = cd.DM([x_init, y_init, theta_init])                      #initial condition
xs = cd.DM([x_target, y_target, theta_target])     #reference posture
xx = cd.DM(x0)                #contains history of states

t = [t0]
times = np.array([[0]])

u0 = cd.DM(N,2)
sim_time = 50

mpciter = 0

xx1 = cd.repmat(x0, 1, N+1)         # initial state full
xx1 = DM2Arr(xx1)
u_cl = DM2Arr(u0[:, 0])
# xx = 

while(np.linalg.norm((x0-xs),2) > 1e-1 and mpciter < sim_time/T):
    t1 = time()
    args['p'] = cd.vertcat(x0,xs)
    args['x0'] = cd.reshape(u0.T,2*N,1)
    sol = solver(x0=args['x0'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
                p=args['p'])
    
    u =  cd.reshape(sol['x'],2,N)

    ff_value = ff(u,args['p']) #compute optimal solution trajectory

    # u_cl = cd.vertcat(u_cl, u[:,0])
    
    # xx1 = cd.vstack(
    #         xx1,
    #         ff_value
    #     )

    u_cl = np.vstack((
        u_cl,
        DM2Arr(u[:, 0])
    ))

    t = np.vstack((
            t,
            t0))   
    xx1 = np.dstack((
            xx1,
            DM2Arr(ff_value)
        ))
    t0, x0, u0, = shift_timestep(T, t0, x0, u, f)
    
    
    # xx = cd.horzcat(xx,x0) 
    # xx = DM2Arr(xx)
    
    t2 = time()
    
    times = np.vstack((
           times,
           t2-t1
       ))
    
    
    mpciter = mpciter+1

main_loop_time = time()
# ss_error = cd.norm_2(state_init - state_target)

# print('\n\n')
# print('Total time: ', main_loop_time - main_loop)
# print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
# print('final error: ', ss_error)

# simulate
simulate(xx1, u_cl, times, T, N,
          np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=True)


    

