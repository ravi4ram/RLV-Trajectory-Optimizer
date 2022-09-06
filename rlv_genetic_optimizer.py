#
#     Author: ravi_ram
#

import array
import math
import random
import numpy as np
from deap import base, creator, tools, algorithms
from rlv_empirical_data import database


class optimizer:
    # create instance
    atm = database()

    # bounds
    alpha_min, alpha_max = 0, 40

    # set up configuration
    num_gen = 30  # number of generations
    num_ind = 100  # number of individuals in the initial population
    prob_cx = 0.6  # probability of cross-over
    prob_mt = 0.01  # probability of mutation
    eta = 5.0  # eta parameter for mate and mutate functions
    num_hof = 10  # hall of frame size
    
    # calculate constraint functions with c_alpha
    def calculate(self, alpha):
        #print(self.y)
        # [h, v, gamma, lamda, phi, psi] = y
        h = self.y[0];  v = self.y[1];
        # [R, K, T0]    = tem_data
        R = self.tem_data[0]; K = self.tem_data[1];
        # [S, k]        = aero_data
        S = self.aero_data[0]; k = self.aero_data[1];
        # [Rn, m, rho0] = orb_data
        Rn = self.orb_data[0]; m = self.orb_data[1]; rho0 = self.orb_data[2];
        
        # evaluate functions  
        # get atmosphere details (alt in km)
        h_km = (h/1000.0)    
        T, rho, p, m = self.atm.get_atmospheric_data(h_km)

        # get CLa, CD0 for Mach number (from tables) 
        mach = v / math.sqrt( K * R * T)
        CLa, CD0 = self.atm.get_mach_data(mach)   

        # dynamic pressure estimation
        Qp = 1./2. * rho * v**2
        Qp_kpa = (Qp/1000.0)

        # lift, drag estimation        
        CL = CLa * math.radians(alpha) # /rad
        CD = CD0 + k*CL**2
        D = Qp * S * CD       # drag
        L = Qp * S * CL       # lift

        # heat flux estimation
        # move the sign outside to prevent numpy error on negative power
        #q = 18300/np.sqrt(Rn) * np.power(rho,0.5) * np.power(v/1e4,3.05)        
        r_ratio = rho; v_ratio = v/1e4
        r_t = np.sign(r_ratio) * (np.abs(r_ratio)) ** (0.5)
        v_t = np.sign(v_ratio) * (np.abs(v_ratio)) ** (3.05)        
        q = 18300/np.sqrt(Rn) * r_t * v_t

        return [alpha, L, D, Qp_kpa, q] 
    
    # fitness with c_alpha, minimize weighted sum
    def eval(self, x):
        alpha = x[0]
        # get the values for minimize
        alpha, L, D, Qp_kpa, q = self.calculate(alpha)
        # weighted sum average
        weights = [0.8, 0.5]
        functions = [Qp_kpa, q]
        # weighted sum average
        fit = sum(x * y for x, y in zip(weights, functions)) / sum(functions) 
        #fit = sum(weights[i]*f for i, f in enumerate(functions) )
        return fit,

    # constraints
    # (1) Qp < 45 kPa         # Dynamic pressure
    # (2) q < 18.5 W/cm^2     # Heat flux
    def feasible(self, x):
        alpha = x[0]
        alpha, L, D, Qp_kpa, q = self.calculate(alpha)        
        # enforce constraints
        if Qp_kpa > 45.0:        # Dynamic pressure < 45 kPa
            return False
        if q > 18.5:           # Heat flux < 18.5 W/cm^2 
            return False
        return True

    # initalize problem 
    def __init__(self):
        # seed
        random.seed(64)
        
        # constants
        alpha_min, alpha_max = self.alpha_min, self.alpha_max
        eta = self.eta
        # functions
        eval = self.eval
        feasible = self.feasible
        
        # rlv data storage for eval function
        self.y = None
        self.ctrl_data = None; self.orb_data = None;
        self.tem_data = None;  self.aero_data = None
        
        # fitness and toolbox
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        toolbox = self.toolbox
        toolbox.register("alpha_i", random.randint, alpha_min, alpha_max)        
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.alpha_i, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)        
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", eval)        
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[alpha_min], up=[alpha_max], eta=eta )
        toolbox.register("mutate", tools.mutPolynomialBounded, low=[alpha_min], up=[alpha_max], eta=eta, indpb=0.05)
        toolbox.decorate("evaluate", tools.constraint.DeltaPenalty(feasible, -1000))         

    # find optimal solution
    def solve(self):
        toolbox = self.toolbox
        num_ind = self.num_ind; num_hof = self.num_hof; num_gen = self.num_gen;
        prob_cx = self.prob_cx; prob_mt = self.prob_mt;        
        # population
        pop = toolbox.population(n=num_ind)
        hof = tools.HallOfFame(num_hof)
        #         
        algorithms.eaSimple(pop, toolbox, cxpb=prob_cx, mutpb=prob_mt,
                            ngen=num_gen, halloffame=hof, verbose=False)
        return hof

    # get the optimized control value
    def get_values(self, y, ctrl_data, orb_data, tem_data, aero_data):
        # store the data for local processing
        [h, v, gamma, lamda, phi, psi] = y[-1]
        [alpha, L, D] = ctrl_data[-1]
        [Rn, m, rho0] = orb_data
        [R, K, T0]    = tem_data
        [S, k]        = aero_data
        self.y = y[-1]
        self.ctrl_data = ctrl_data[-1]
        self.orb_data = orb_data
        self.tem_data = tem_data
        self.aero_data = aero_data        
        # get the optimized control values
        hof = self.solve()
        # get the L and D values for the alpha 
        alpha = hof[0][0]
        alpha, L, D, Qp_kpa, q = self.calculate(alpha)

        return alpha, L, D


