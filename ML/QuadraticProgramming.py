from cvxopt import matrix, solvers

#giải bài toán sau
#Tìm điểm M(x,y) sao cho khoảng cách từ M đến A(10,10) là nhỏ nhất và M thỏa các điều kiện
#x+y<=10
#2x+y<=16
#x+4y<=32
#x,y>=0
G = matrix([[1.,2.,1.,-1.,0.],[1.,1.,4.,0.,-1.]])
h = matrix([10.,16.,32.,0.,0.])
P = matrix([[2.,0],[0.,2.]])
q = matrix([-20.,-20.])
r = 200
solve = solvers.qp(P=P,q=q,G=G,h=h)
print(solve['x'])