import numpy as np
from copy import copy
from typing import Any, List, Tuple, Union
#from typing_extensions import Self
from dataclasses import dataclass

# primeros pasos aqui



# class Simplex:
#     """
#     El metodo simplex creado por estudiantes de la clase de PL de la ENES Morelia 2022 c:
#     """

#     def __call__(self, A: np.ndarray, w = False) -> np.ndarray:
#         self.A = A
#         n, m = self.A.shape
#         self.pivotes = []
#         if w:
#             self.__calc_w()
#             self.__simplex()
#             #elimiar algo (W)
        
#         self.__simplex()
#         print(self)


#     def __calc_w(self):
#         """
#         Su funcion para calcular la funcion w aqui
#         """
#         return None

#     def __condicion_de_paro(self):
#         return all(elem >= 0 for elem in self.A[-1, :-1])
#         # return not any(elem < 0 for elem in self.A[-1, :-1])

#     def __es_pivote(self) -> np.ndarray:
#         """
#         Su funcion de poner pivote aqui

#         Regresa
#         -------

#         Una matriz de numpy con las coordenadas donde se encuanta los 1's en formato: [[y1,x1], [y2,x2], ...]
#         """
#         xs = np.where(np.sum(self.A, axis=0) == 1)[0]
#         ys = np.where(self.A[:,xs] == 1)[1]
#         return np.column_stack((ys,xs))

#     def __sacar_pivote(self):
#         # sacar el indice mas pequeño de Z
#         col_index = np.argmin(self.A[-1])
#         # dividr los indices de la col por el RHS
#         divisiones = [
#             rhs / pivote for pivote, rhs in zip(self.A[:-1, col_index], self.A[:-1, -1])
#         ]

#         fila_index = np.argmin(divisiones)

#         return fila_index, col_index

#     def __operaciones_elementales(self, fila_index: int, col_index: int):
#         self.A[fila_index, :] /= self.A[fila_index, col_index]

#         for ix, fila in enumerate(self.A):
#             if ix != fila_index:
#                 ## revisar el signo del elemento que voy a eliminar
#                 ##multiplicar por el signo contratio y sumarlo
#                 signo = 1
#                 if fila[col_index] >= 0:
#                     signo = -1

#                 fila += signo * abs(fila[col_index]) * self.A[fila_index, col_index]

#     def __simplex(self):
#         self.pivotes = self.__es_pivote()
#         while not self.__condicion_de_paro():
#             pivote_i, pivote_j = self.__sacar_pivote()
#             self.__operaciones_elementales(pivote_i, pivote_j)
#             # self.__intercambio(pivote_i, pivote_j)
#             self.pivotes = self.__es_pivote()

#     def __repr__(self) -> str:
#         return " ".join(
#             f"x_{self.pivotes[i][1] + 1} = {self.A[self.pivotes[i][0]][-1]}"
#             for i in range(len(self.pivotes))
#         )


class Simplex:

    def __call__(self, M: np.ndarray, z: np.ndarray):
        self.M = M.astype(float) #matriz de las variables
        self.z = z.astype(float) #la funcion a optimizar
        self.MSimplex = np.append(M, [z], axis=0)

        self.pivots = self.is_pivot()
        self.initial_pivots = copy(self.pivots)
        self.w = self.calculate_w()
        self.MSimplex = np.append(self.MSimplex, [self.w], axis=0)
        self.MSimplex = self.simplex()
        #self.MSimplex = self.del_w()
        return self.MSimplex,self.pivots
        
    
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
        #row_idx = np.argmin(self.MSimplex[:self.M.shape[0], -1] / self.MSimplex[:self.M.shape[0], col_idx])
        col_idx = np.argmin(self.MSimplex[-1])
        a = self.MSimplex[:self.M.shape[0], -1]
        b = self.MSimplex[:self.M.shape[0], col_idx]
        divs = np.divide(a, b, out=np.full(a.shape, -np.inf), where=b>0)
        v_idx = np.where(divs >= 0)[0]
        row_idx = v_idx[divs[v_idx].argmin()]
        return row_idx, col_idx


    def stop_condition(self) -> bool:
        return np.all(self.MSimplex[-1,:-1] >= 0)
    

    def calculate_w(self) -> np.ndarray:
        #self.w = copy(self.MSimplex[-1])
        w = copy(self.MSimplex[-1])

        if self.stop_condition():
            return w

        for i in self.pivots:
            row = i[0]
            col = i[1]

            #if self.w[col] == 0:
            if w[col] == 0:
                continue

            #self.w += self.MSimplex[row] * (self.w[col] / self.MSimplex[row][col] * -1)
            w += self.MSimplex[row] * (w[col] / self.MSimplex[row][col] * -1) 

        return w
    

    def del_w(self):
        M = self.MSimplex[:-1]
        return np.delete(M, self.initial_pivots, axis=1)


    def make_zeros(self, row: int, col: int) -> np.ndarray:
        
        MS = copy(self.MSimplex)
        MS[row] /= MS[row, col]

        # self.MSimplex[row] /= self.MSimplex[row, col]
        #a = self.MSimplex[row]
        #b = self.MSimplex[row, col]
        #self.MSimplex[row] = np.divide(a, b, out=np.full(a.shape, np.inf), where=b!=0)

        for xi in range(MS.shape[0]):
            if xi == row:
                continue
                
            #self.MSimplex[xi] += self.MSimplex[row] * (self.MSimplex[xi, col] / self.MSimplex[row,col] * -1) 
            MS[xi] += MS[row] * (MS[xi, col] / MS[row,col] * -1) 

        return MS


    def simplex(self) -> np.ndarray:
        M = copy(self.MSimplex)
        self.pivots = self.is_pivot()

        while not self.stop_condition():
            self.MSimplex = self.make_zeros(*self.get_pivot())
            #M = self.make_zeros(*self.get_pivot(), MS=M)
            self.pivots = self.is_pivot()


        return self.MSimplex



def main(M):
    return None






if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    #matriz_examen = np.array([[4,2,1,0,0,0,24],[5,3,0,-1,1,0,30],[2,0,0,0,0,1,8],[15,6,0,0,0,0,0]])
    #m1 = np.array([[2,1,1,0,120],[1,1,0,1,90],[-80,-50,0,0,0]], dtype=float)
    M = np.array([[4,2,1,0,0,0,24],[5,3,0,-1,1,0,30],[2,0,0,0,0,1,8]],dtype=np.float64)
    z = np.array([15,6,0,0,0,0,0],dtype=np.float64)


    # M = np.array([
    #              [-2, 0,  6,  2,  0, -3, 1, 20],
    #              [-4, 1,  7,  1,  0, -1, 0, 10],
    #              [ 0, 0, -5,  3,  1, -1, 0, 60]],dtype=float)

    # z = np.array([-4, 1, 30, -11, -2,  3, 0, 0],dtype=np.float64)
    simplex = Simplex()
    #simplex(matriz_examen)
    M,pv = simplex(M,z)

    print(M.round(1))
    print(pv)
    print(M[pv[:,0],-1])
    

