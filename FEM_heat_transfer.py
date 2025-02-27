import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class FEMSolver_unsteady:
    def __init__(self, nodes, length, k, alpha, hh, Tinf, Q, delt, nt, T_init=None):
        self.nodes = nodes
        self.length = length
        self.k = k
        self.alpha = alpha
        self.hh = hh
        self.Tinf = Tinf
        self.Q = Q
        self.delt = delt
        self.nt = nt
        self.h = length / (nodes - 1)
        self.x = np.linspace(0, length, nodes)
        self.T = np.zeros(nodes) if T_init is None else np.array(T_init)
        self.set_initial_conditions()

    def set_initial_conditions(self):
        if not np.any(self.T):
            self.T[0] = 0
            self.T[1:] = 200

    def stiffness_matrix(self):
        K = np.zeros([self.nodes, self.nodes])
        for j in range(self.nodes):
            for i in range(self.nodes):
                if i == j:
                    K[j, i] = 1 if i == 0 or i == self.nodes - 1 else 2
                elif abs(i - j) == 1:
                    K[j, i] = -1
        return K

    def force_vector(self):
        c = (self.Q * self.h * self.h) / self.k
        f = np.zeros(self.nodes)
        f[1:self.nodes-1] = 1
        f[0] = f[-1] = 0.5
        return f * c

    def mass_matrix(self):
        M = np.zeros([self.nodes, self.nodes])
        for j in range(self.nodes):
            for i in range(self.nodes):
                if i == j:
                    M[j, i] = 2 if i == 0 or i == self.nodes - 1 else 4
                elif abs(i - j) == 1:
                    M[j, i] = 1
        return M * (1/6)

    def force_vector_transient(self):
        f = np.ones(self.nodes)
        f[0] = f[-1] = 0.5
        return f * (self.Q / self.k)

    def crank_nicolson_solver(self):
        T_list = [self.T.copy()]
        K_conv = np.zeros([self.nodes, self.nodes])
        K_conv[-1, -1] = self.hh / self.k
        f_conv = (self.hh * self.Tinf) / self.k
        K = self.stiffness_matrix() / self.h
        M = self.mass_matrix() * (self.h / self.alpha)
        f = self.force_vector_transient() * self.h
        f[-1] += f_conv
        K += K_conv
        A_inv = np.linalg.inv(M + (self.delt / 2) * K)
        B = M - (self.delt / 2) * K

        for _ in range(self.nt):
            c = self.delt * f + np.dot(B, self.T)
            T_new = np.dot(A_inv, c)
            T_new[0] = self.T[0]  # Enforce boundary condition
            T_new[-1] = self.T[-1]  # Enforce boundary condition
            self.T = T_new
            T_list.append(T_new)
        return T_list

    def plot(self, T_list):
        plt.plot(self.x, T_list[-1], linestyle='-', color='b', label='FEM Solution')
        plt.xlabel("Distance")
        plt.ylabel("Temperature")
        plt.legend()
        plt.show()

    def animate(self, T_list):
        fig, ax = plt.subplots()
        line, = ax.plot(self.x, T_list[0], marker='', linestyle='-', color='b')
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature Distribution Over Time")
        ax.set_xlim(0, max(self.x))
        ax.set_ylim(min(map(min, T_list)) - 10, max(map(max, T_list)) + 10)

        def update(frame):
            line.set_ydata(T_list[frame])
            return line,

        ani = FuncAnimation(fig, update, frames=range(0, len(T_list), 10), interval=50, blit=True)
        plt.show()
        return ani

    def solve(self, method='crank_nicolson'):
        if method == 'crank_nicolson':
            return self.crank_nicolson_solver()
        else:
            raise ValueError("Unsupported method")

class FEMSolver_steady:
    def __init__(self, nodes, length=1, k=50, Q=1000, T0=100, T_end=None, hh=0, Tinf=35):
        self.nodes = nodes
        self.length = length
        self.h = length / (nodes - 1)
        self.k = k
        self.Q = Q
        self.T0 = T0
        self.T_end = T_end
        self.hh = hh
        self.Tinf = Tinf
        self.x = np.linspace(0, length, nodes)
        self.t = np.zeros(nodes)
        self.t[0] = T0
        self.K = self.stiffness_matrix()
        self.f = self.force_vector()
        self.apply_boundary_conditions()
    
    def stiffness_matrix(self):
        K = np.zeros((self.nodes, self.nodes))
        for j in range(self.nodes):
            for i in range(self.nodes):
                if i == j:
                    K[j, i] = 1 if (i == 0 or i == self.nodes - 1) else 2
                elif abs(i - j) == 1:
                    K[j, i] = -1
        return K
    
    def force_vector(self):
        c = (self.Q * self.h ** 2) / self.k
        f = np.zeros(self.nodes)
        f[1:-1] = 1
        f[0] = f[-1] = 0.5
        return f * c
    
    def apply_boundary_conditions(self):
        if self.T_end is not None:
            self.t[-1] = self.T_end
            soln_range = (1, self.nodes - 1)
        else:
            soln_range = (1, self.nodes)
        
        a, b = soln_range
        
        # Apply convection terms
        f_c = (self.h * self.hh * self.Tinf) / self.k
        K_c = (self.h * self.hh) / self.k
        self.K[-1, -1] += K_c
        self.f[-1] += f_c
        
        # Solve for unknown temperatures
        self.f -= np.dot(self.K, self.t)
        self.t[a:b] = np.linalg.solve(self.K[a:b, a:b], self.f[a:b])
    
    def plot_solution(self):
        i = np.linspace(0, 1, 100)
        #j = -10 * i ** 2 + 20 * i + 100  # Analytical solution for heat generation case
        
        plt.plot(self.x, self.t, marker='o', linestyle='-', color='b', label="FEM Solution")
        #plt.plot(i, j, color='r', label="Analytical")
        plt.xlabel("Distance")
        plt.ylabel("Temperature")
        plt.legend()
        plt.show()
    
    def solve(self):
        print(self.t)
        self.plot_solution()

        
# Usage
#k = heat transfer coeff
#T0 = temperature at end 0
#T_end = temperature at other end  (L)
# hh = convection coefficient
# Q = Internal heat generation rate
# T_inf = temperature of ambient surrounding
# thermal diffusivity. For steady-state solver, it isnt present

if __name__ == "__main__":
    solve = "unsteady"
    #For unsteady state
    if solve == "unsteady":
        nodes = 100
        length = 1
        k = 100
        alpha = 12.5e-6
        hh = 45
        Tinf = 30
        Q = 0
        delt = 4
        nt = int((60 * 60) / delt)
        T_init = [100] + [0] * (nodes - 2) + [100]  # Example initial condition
    
        
        solver = FEMSolver_unsteady(nodes, length, k, alpha, hh, Tinf, Q, delt, nt, T_init)
        T_list = solver.solve()
        solver.plot(T_list)
        solver.animate(T_list)

    elif solve == "steady":
    #For steady state
        solver = FEMSolver_steady(nodes=10,length=1, k=50, Q=0, T0=100, T_end=20, hh=0, Tinf=35)
        T_list = solver.solve()
        

    else:
        print("ERROR")
        

    
    