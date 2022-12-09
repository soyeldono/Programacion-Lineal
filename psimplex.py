import numpy as np
import math
from copy import copy,deepcopy
from typing import Any, List, Tuple, Union
#from typing_extensions import Self
from dataclasses import dataclass


def insert_in_list(L: List[List[Any]], ob: List[Any], i: int, axis: int=0) -> List[List[Any]]:
    """
    Insert array object 'ob' into the list 'L' in index 'i' position as row or col
    depending on axis option
    """
    l_shape = (len(L),len(L[0]))
    ob_shape = None


    if isinstance(ob[0], list):
        ob_shape = (len(ob), 1)
        ob_shape = (len(ob),len(ob[0]))
    else:
        ob_shape = (1, len(ob))
        ob = [ob]
        

    if (axis and l_shape[0] != ob_shape[0]) or (not axis and l_shape[1] != ob_shape[1]): #axis == 1 -> col, axis == 0 -> row
        raise TypeError(f"Size of ob is not compatible to L, L shape:{l_shape} not compatible with ob shape:{ob_shape}")
    
    if (axis and i > l_shape[1]) or (not axis and i > l_shape[0]):
        raise ValueError(f"Index out of range, recived index {i} but max size of list is {l_shape}")


    if not axis: #row
        for k,o in enumerate(ob):
            L.insert(k+i, o)    
    elif axis: #cols
        for k,o in enumerate(ob):
            for j,oi in enumerate(o):
                L[k].insert(i+j, oi)

    return L


def delete(L: List[Any], idx: Union[List, int], axis: int=0) -> List[List[Any]]:
    if isinstance(idx, int):
        idx = [idx]
    _L = []
    for k,row in enumerate(L):
        if not axis and k not in idx:
            _L.append(row)
        elif axis:
            r = []
            for t,col in enumerate(row):
                if t not in idx:
                    r.append(col)
            _L.append(r)
    return _L
        

def zeros(shape: Union[Tuple, int], add: int=0) -> List[int]:
    if isinstance(shape, int):
        return [0+add for _ in range(shape)]
    
    L = []
    for row in range(shape[0]):
        L.append([0+add for _ in range(shape[1])])
    return L


class Simplex:

    def __call__(self, 
                M: List[List[Union[int, float]]], 
                z: List[Union[int, float]], 
                R: List, dtypes: List[Union[int, float]]) -> Tuple[List, List]:
        """
        M: numpy array with size (m,n)
            Matrix with the variables

        z: numpy vector
            function to minimize

        R: list
            Restriction of each row

        dtypes: list
            dtypes of each variables
        """
        self.M = M #matriz de las variables
        self.z = z #la funcion a optimizar
        self.R = R
        self.dtypes = dtypes

        self.MSimplex = self.make_simplex()

        self.z = z + zeros(len(self.MSimplex[0]) - len(z))
        self.MSimplex.append(self.z)
        self.pivots = self.is_pivot()
        self.initial_pivots = copy(self.pivots)
        self.MSimplex[-1] = self.del_pivots_z()

        self.w = self.calculate_w()
        self.MSimplex.append(self.w)
        self.MSimplex = self.simplex()
        self.MSimplex = self.del_w()
        self.MSimplex = self.simplex()
        return self.MSimplex,self.pivots
        
    
    def is_pivot(self) -> np.ndarray:
        """
        Determina las columnas y filas pivotes

        Regresa
        -------
        Una matriz de numpy con las coordenadas donde se encuanta los 1's en formato: [[y1,x1], [y2,x2], ...]
        """
        l = len(self.M)

        data = []

        for col in range(len(self.MSimplex[0])-1):
            p = True
            s = None
            for k,row in enumerate(self.MSimplex[:l]):
                if row[col] != 0 and row[col] != 1:
                    p = False

                elif row[col] == 1 and p:
                    s = (k,col)
            
            if p:
                data.append(s)
        
        return data
    

    def get_pivot(self) -> Tuple[int,int]:
        col_idx = self.MSimplex[-1][:-1].index(min(self.MSimplex[-1][:-1]))
        a = [i[-1] for i in self.MSimplex[:len(self.M)]]
        b = [i[col_idx] for i in self.MSimplex[:len(self.M)]]
        divs = [ai / bi if bi > 0 else math.inf for ai,bi in zip(a,b)]
        row_idx = divs.index(min(divs))
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

            m = [r*(z[col]/self.MSimplex[row][col])*-1 for r in self.MSimplex[row]]
            z = [zi + mi for zi,mi in zip(z,m)]

        return z


    def stop_condition(self) -> bool:
        return all([True if round(i,8) >=0 else False for i in self.MSimplex[-1][:-1]])
    

    def calculate_w(self) -> np.ndarray:
        pivr = []
        pivc = []
        for i,k in enumerate(self.artificial_variables):
            if k:
                pivr.append(self.pivots[i][0])
                pivc.append(self.pivots[i][1])
        rows = []
        for i in pivr:
            rows.append(self.MSimplex[i])
        
        sum = rows[0]

        for k,i in enumerate(rows[1:]):
            sum = [j + v for j,v in zip(sum,i)]

        w = zeros(len(self.MSimplex[0]))

        for i in pivc:
            w[i] = 1

        return [wi - sri for wi,sri in zip(w,sum)] #w - SR


    def del_w(self):
        if round(self.MSimplex[-1][-1],8) != 0:
            raise TypeError("Original problem has no feasible solutions")

        M = self.MSimplex[:-1]
        pv = []
        for i,k in enumerate(self.artificial_variables):
            if k:
                pv.append(self.initial_pivots[i][1])
        return delete(M, pv, axis=1)

    
    def make_simplex(self):
        self.artificial_variables = []
        if self.R[0] == "<=":
            MS = insert_in_list(deepcopy(self.M), zeros((len(self.M,1))), len(self.M[0])-1, axis=1)
            MS[0][len(MS[0])-2] = 1
            self.artificial_variables.append(False)
        elif self.R[0] == ">=":
            MS = insert_in_list(deepcopy(self.M), zeros((len(self.M),1)), len(self.M[0])-1, axis=1)
            MS[0][len(MS[0])-2] = -1
            MS = insert_in_list(MS, zeros((len(self.M),1)), len(MS[0])-1, axis=1)
            MS[0][len(MS[0])-2] = 1
            self.artificial_variables.append(True)
        elif self.R[0] == "=":
            MS = insert_in_list(deepcopy(self.M), zeros((len(self.M),1)), len(self.M[0])-1, axis=1)
            MS[0][len(MS[0])-2] = 1
            self.artificial_variables.append(True)
        else:
            raise TypeError(f"Unkown restriction {self.R[0]}")

        for k,i in enumerate(self.R[1:],start=1):
            if i == "<=":
                MS = insert_in_list(MS, zeros((len(MS),1)), len(MS[0])-1, axis=1)
                MS[k][len(MS[0])-2] = 1
                self.artificial_variables.append(False)
            elif i == ">=":
                MS = insert_in_list(MS, zeros((len(MS),1)), len(MS[0])-1, axis=1)
                MS[k][len(MS[0])-2] = -1
                MS = insert_in_list(MS, zeros((len(MS),1)), len(MS[0])-1, axis=1)
                MS[k][len(MS[0])-2] = 1
                self.artificial_variables.append(True)
            elif i == "=":
                MS = insert_in_list(MS, zeros((len(MS),1)), len(MS[0])-1, axis=1)
                MS[k][len(MS[0])-2] = 1
                self.artificial_variables.append(True)
            else:
                raise TypeError(f"Unkown restriction {i}")
                
        return MS


    def make_zeros(self, row: int, col: int) -> np.ndarray:
        
        MS = copy(self.MSimplex)
        #MS[row] /= MS[row, col]
        MS[row] = [ri / MS[row][col] for ri in MS[row]]

        for xi in range(len(MS)): #range(MS.shape[0]):
            if xi == row:
                continue

            #MS[xi] += MS[row] * (MS[xi, col] / MS[row,col] * -1) 
            a = [r*(MS[xi][col]/MS[row][col]*-1) for r in MS[row]]
            MS[xi] = [mi + ai for mi,ai in zip(MS[xi],a)]

        return MS


    def simplex(self) -> np.ndarray:
        M = copy(self.MSimplex)
        self.pivots = self.is_pivot()

        while not self.stop_condition():
            #self.MSimplex = self.make_zeros(*self.get_pivot())
            self.MSimplex = self.make_zeros(*self.get_pivot())
            #M = self.make_zeros(*self.get_pivot(), MS=M)
            self.pivots = self.is_pivot()


        return self.MSimplex




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

    M = np.array([
                 [3, 2, 60],
                 [7, 2, 84],
                 [3, 6, 72]],dtype=float)

    z = np.array([10,4,0],dtype=float)



    simplex = Simplex()
    #simplex(matriz_examen)
    M,pv = simplex(M.tolist(),z.tolist(),[">=",">=",">="],[])

    print(np.array(M))
    print(pv)
        

