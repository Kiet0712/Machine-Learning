from cvxopt import matrix, solvers
#bài toán sách
def lp1():
    c = matrix([5.,10.,15.,4.])
    G = matrix([[1.,0.,-1.,0.,0.,0.],[1.,0.,0.,-1.,0.,0.],[0.,1.,0.,0.,-1.,0.],[0.,1.,0.,0.,0.,-1.]])
    b = matrix([600.,400.])
    A = matrix([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    h = matrix([800.,700.,0,0,0,0])
    solve = solvers.lp(c,G,h,A,b)
    print(solve['x'])
#nghiệm cho ra là x = 600, y=z=0, t = 400
#bài toán canh tác
# tìm giá trị lớn nhất hàm f(x,y)=5x+3y với các điều kiện x+y<=10,2x+y<=16,x+4y<=32,x>=0,y>=0
def lp2():
    c = matrix([-5.,-3.])
    G = matrix([[1.,2.,1.,-1.,0.],[1.,1.,4.,0.,-1.]])
    h = matrix([10.,16.,32.,0.,0.])
    solve = solvers.lp(c,G,h)
    print(solve['x'])
lp2()
#nghiệm là x = 6, y = 4