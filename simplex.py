import numpy as np
from copy import copy
from typing import Any, List, Tuple, Union
#from typing_extensions import Self
from dataclasses import dataclass


class Simplex:
    def __call__(self, 
                M: np.ndarray, 
                z: np.ndarray, 
                R: List[str], 
                r: List[str]=None, 
                dtypes: List[Union[int, float]]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        M: numpy array with size (m,n)
            Matrix with the variables

        z: numpy vector
            function to minimize

        R: list
            Restriction of each row
        
        r: list of tuples (restriccion: str, value: int/float)
            Restriction of each variable

        dtypes: list
            dtypes of each variables
        """
        self.M = M.astype(float) #matriz de las variables
        self.z = z.astype(float) #la funcion a optimizar
        self.R = R
        self.r = r
        self.dtypes = dtypes

        if len(self.R) != self.M.shape[0]:
            raise TypeError(f"restrictions and rows must have same size but got: restrictions:{len(self.R)} != rows:{self.M.shape[0]}")

        self.M,self.z = self.lower_bounded_variables(M, z, r)

        self.MSimplex = self.make_simplex()

        self.z = np.append(z, np.zeros(self.MSimplex.shape[1] - self.z.shape[0])) 
        if not np.all(self.artificial_variables):
            self.z = -1*self.z
        self.MSimplex = np.append(self.MSimplex, [self.z], axis=0)
        self.pivots = self.is_pivot()
        self.initial_pivots = copy(self.pivots)
        self.MSimplex[-1] = self.del_pivots_z()

        self.w = self.calculate_w()
        self.MSimplex = np.append(self.MSimplex, [self.w], axis=0)
        self.MSimplex = self.simplex()
        self.MSimplex = self.del_w()
        self.MSimplex = self.simplex()
        return self.MSimplex,self.pivots
    

    def lower_bounded_variables(self, M: np.ndarray, z: np.ndarray, r: List[str]) -> np.ndarray:
        for k,i in enumerate(r):
            if i[0] == ">=":
                M[:,-1] = M[:,k]*i[1]*-1 + M[:,-1]
                z[-1] = z[k]*i[1]*-1 + z[-1]
        return M, z


    def upper_bounded_variables(self, M, piv, r) -> Tuple[bool,np.ndarray]:
        if not (self.MSimplex[piv[0],-1] / self.MSimplex[piv[0],piv[1]] <= r[piv[1]][1] and r[piv[1]][0] == "<="):
            M[:,-1] += M[:,piv[1]]*r[piv[1]][1]*-1
            M[:, piv[1]] *= -1
            return False, M
        return True,M
    
    
    def is_pivot(self) -> np.ndarray:
        """
        Determina las columnas y filas pivotes

        Regresa
        -------
        Una matriz de numpy con las coordenadas donde se encuanta los 1's en formato: [[y1,x1], [y2,x2], ...]
        """
        l = self.M.shape[0]
        xs = np.where((np.sum(self.MSimplex[:l] == 0, axis=0) == self.MSimplex[:l].shape[0] - 1 ) & (np.sum(self.MSimplex[:l], axis=0) == 1))[0]
        ys = np.where(self.MSimplex[:l,xs].T == 1)[1]
        return np.column_stack((ys,xs))
    

    def get_pivot(self) -> Tuple[int,int]:
        col_idx = np.argmin(self.MSimplex[-1,:-1])
        a = self.MSimplex[:self.M.shape[0], -1]
        b = self.MSimplex[:self.M.shape[0], col_idx]
        divsp = np.divide(a, b, out=np.full(a.shape, -np.inf), where=b>0)
        #divsn = np.divide(a-self.r[col_idx][1], b, out=np.full(a.shape, np.inf), where=b<0)
        #row_idx = np.argmin([divsp.min(),divsn.min()])
        v_idx = np.where(divsp >= 0)[0]
        row_idx = v_idx[divsp[v_idx].argmin()]
        return row_idx, col_idx


    def del_pivots_z(self) -> np.ndarray:
        z = copy(self.MSimplex[-1])

        if self.stop_condition():
            return z

        for i in self.pivots:
            row = i[0]
            col = i[1]

            if z[col] == 0:
                continue

            z += self.MSimplex[row] * (z[col] / self.MSimplex[row,col] * -1) 

        return z


    def stop_condition(self) -> bool:
        return np.all(self.MSimplex[-1,:-1].round(8) >= 0)
    

    def calculate_w(self) -> np.ndarray:
        piv = self.pivots[self.artificial_variables]
        SR = self.MSimplex[piv[:,0]].sum(axis=0)
        w = np.zeros(self.MSimplex.shape[1])
        w[piv[:,1]] = 1
        return w - SR


    def del_w(self):
        if self.MSimplex[-1,-1].round(8) != 0:
            raise TypeError("Original problem has no feasible solutions")

        M = self.MSimplex[:-1]
        return np.delete(M, self.initial_pivots[self.artificial_variables,1], axis=1)

    
    def make_simplex(self):
        self.artificial_variables = []
        if self.R[0] == "<=":
            MS = np.insert(self.M, [self.M.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
            MS[0,MS.shape[1]-2] = 1
            self.artificial_variables.append(False)
            self.r.append(("<=",np.inf))
        elif self.R[0] == ">=":
            MS = np.insert(self.M, [self.M.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
            MS[0,MS.shape[1]-2] = -1
            MS = np.insert(MS, [MS.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
            MS[0,MS.shape[1]-2] = 1
            self.artificial_variables.append(True)
            self.r.append(("<=",np.inf))
        elif self.R[0] == "=":
            MS = np.insert(self.M, [self.M.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
            MS[0,MS.shape[1]-2] = 1
            self.artificial_variables.append(True)
            self.r.append(("<=",np.inf))
        else:
            raise TypeError(f"Unkown restriction {self.R[0]}")

        for k,i in enumerate(self.R[1:],start=1):
            if i == "<=":
                MS = np.insert(MS, [MS.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
                MS[k,MS.shape[1]-2] = 1
                self.artificial_variables.append(False)
                self.r.append(("<=",np.inf))
            elif i == ">=":
                MS = np.insert(MS, [MS.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
                MS[k,MS.shape[1]-2] = -1
                MS = np.insert(MS, [MS.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
                MS[k,MS.shape[1]-2] = 1
                self.artificial_variables.append(True)
                self.r.append(("<=",np.inf))
            elif i == "=":
                MS = np.insert(MS, [MS.shape[1]-1], np.zeros((self.M.shape[0],1)), axis=1)
                MS[k,MS.shape[1]-2] = 1
                self.artificial_variables.append(True)
                self.r.append(("<=",np.inf))
            else:
                raise TypeError(f"Unkown restriction {i}")
                
        return MS


    def make_zeros(self, row: int, col: int) -> np.ndarray:
        MS = copy(self.MSimplex)
        MS[row] /= MS[row, col]

        for xi in range(MS.shape[0]):
            if xi == row:
                continue
                
            MS[xi] += MS[row] * (MS[xi, col] / MS[row,col] * -1) 

        return MS


    def simplex(self) -> np.ndarray:
        M = copy(self.MSimplex)
        self.pivots = self.is_pivot()

        while not self.stop_condition():
            pivots = self.get_pivot()
            #suc,self.MSimplex = self.upper_bounded_variables(self.MSimplex, pivots, self.r)
            #pivots = self.get_pivot()

            #while not suc:
            #    suc,self.MSimplex = self.upper_bounded_variables(self.MSimplex, pivots, self.r)
            #    pivots = self.get_pivot()

            
            self.MSimplex = self.make_zeros(*pivots)
            self.pivots = self.is_pivot()


        return self.MSimplex



def main(M):
    return None






if __name__ == "__main__":
    #np.set_printoptions(suppress=True)
    #matriz_examen = np.array([[4,2,1,0,0,0,24],[5,3,0,-1,1,0,30],[2,0,0,0,0,1,8],[15,6,0,0,0,0,0]])
    #m1 = np.array([[2,1,1,0,120],[1,1,0,1,90],[-80,-50,0,0,0]], dtype=float)
    #M = np.array([[4,2,1,0,0,0,24],[5,3,0,-1,1,0,30],[2,0,0,0,0,1,8]],dtype=np.float64)
    #z = np.array([15,6,0,0,0,0,0],dtype=np.float64)


    # M = np.array([
    #              [-2, 0,  6,  2,  0, -3, 1, 20],
    #              [-4, 1,  7,  1,  0, -1, 0, 10],
    #              [ 0, 0, -5,  3,  1, -1, 0, 60]],dtype=float)

    # z = np.array([-4, 1, 30, -11, -2,  3, 0, 0],dtype=np.float64)

    # M = np.array([
    #              [3, 2, -1,  0,  0, 1, 0, 0, 60],
    #              [7, 2,  0, -1,  0, 0, 1, 0, 84 ],
    #              [3, 6,  0,  0, -1, 0, 0, 1, 72 ]],dtype=np.float16)

    # z = np.array([10,4,  0,  0,  0, 0, 0, 0, 0],dtype=np.float16)


    # M = np.array([
    #              [-1, 2, 1, 1],
    #              [-1, 0, 2, 4],
    #              [1, -1, 2, 4]],dtype=float)

    # z = np.array([1,1,1],dtype=float)

    # M = np.array([
    #              [4, 2, 24],
    #              [5, 3, 30],
    #              [2, 0, 8]],dtype=float)

    # z = np.array([15,6,0],dtype=float)

    # M = np.array([
    #              [3, 2, 60],
    #              [7, 2, 84],
    #              [3, 6, 72]],dtype=float)

    # z = np.array([10,4,0],dtype=float)


    # M = np.array([
    #              [11, 6, 66],
    #              [5, 50, 225]],dtype=float)

    # z = np.array([1,5,0],dtype=float)

    # M = np.array([
    #              [2.5, 1, 3100],
    #              [1, 1, 2000],
    #              [1, 1, 1700]],dtype=float)

    # z = np.array([22,20,0],dtype=float)

    M = np.array([
                 [2, 2,  1, 2, 5],
                 [1, 2, -3, 4, 5]],dtype=float)

    z = np.array([10,15,-10,25],dtype=float)


    simplex = Simplex()
    #simplex(matriz_examen)
    M,pv = simplex(M,z,["<=","<="],[("<=",1),("<=",1),("<=",1),("<=",1)])


    print(M)
    print(pv)
    print(M[pv[:,0],-1])
    

