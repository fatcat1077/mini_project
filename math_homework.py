import numpy as np
A=np.array([[1.73, 5.85],[ -1.69, 9.93]])
B=np.array([[2.77, -3.64],[ -3.64, 5.21]])
C=np.array([[-1.39, -2.14],[ 7.53, 1.12]])
AplusB=A+B
print("A+B等於：\n",AplusB)
AminusB=A-B
print("A-B等於：\n",AminusB)
AmatmulB=np.matmul(A, B)
print("A與B的矩陣乘法等於：\n",AmatmulB)
BmatmulA=np.matmul(B, A)
print("B與A的矩陣乘法等於：\n",BmatmulA)
Atranspos=np.transpose(A)
print("A轉置矩陣：\n",Atranspos)
Btranspos=np.transpose(B)
print("B轉置矩陣：\n",Btranspos)
Ainverse=np.linalg.inv(A)
print("A反矩陣：\n",Ainverse)
ABC=np.matmul(A,np.matmul(B, C))
BCA=np.matmul(B,np.matmul(C, A))
CAB=np.matmul(C,np.matmul(A, B))
Atrace=(np.trace(ABC)==np.trace(BCA)==np.trace(CAB))
print("trace一樣嗎?：",Atrace)
eigenvalues, eigenvectors = np.linalg.eig(A)
print("eigenvalues：",eigenvalues)
print("eigenvectors：\n",eigenvectors)
Bcholesky=np.linalg.cholesky(B)
print("B 矩陣cholesky：\n",Bcholesky)