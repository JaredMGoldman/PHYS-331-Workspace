import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib import rcParams
from scipy.optimize import leastsq
from scipy.interpolate import interp1d


rcParams['font.family']='Skia'


def matrix_mult(m1,m2):
	"""
	INPUTS:
		m1,m2: numpy arrays of shape (m,n), (n,m) respectively where (m == n or m != n)
	OUTPUT:
		matrix product: numpy array of shape (m,n)

	Function that takes two matricies and returns their product.
	"""
	(r1,c1) = m1.shape
	(r2,c2) = m2.shape
	if c1 != r2:
		return "Error, incorect matricies computed. Please reenter matricies with appropriate dimensions."
	newentry=0
	matrix_product = np.zeros((c1,c2))
	for ra in range (0,r1):
		for cb in range (0, c2):
			for ca in range(0,c1):
				newentry = 0
				for rb in range(0,r2):
					newentry += (m1[ra,rb] * m2[rb,cb])
				matrix_product[ra,cb] = newentry			
	return matrix_product


def datafinder():
	"""
	loads the data from the csv document and returns numpy array of experimental values of v and S(v)
	"""
	myvals = np.loadtxt('HW6p1data.csv', delimiter = ',' )
	nicevals = np.transpose(myvals)
	return nicevals



# ---------------------------------------
# Part c


def Jinv():
	"""
	inverse of Jacobian for mutidimensional NR method
	used to estimate the values of c1 and c2 for the best guess
	"""
	return np.array([[380.3974626838066,-222.7529754620734],[-2906.856476518258,19482.28900443848]])
	

def F(x_vec):
	"""
	set of functinos used in multidimensional NR method
	"""
	c1 = x_vec[0,0]
	c2 = x_vec[1,0]
	L_11 = 0.002880503018073279
	L_21 = 3.293456010005426e-05
	L_12 = 0.00042978568133390815
	L_22 = 5.624267451727517e-05
	Sv1 = 0.060596
	Sv2 = 0.01215
	return np.array([[c1*L_11 + c2*L_21-Sv1],[c1*L_12 + c2*L_22-Sv2]])


def rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter):
	"""
	INPUTS:
		F_system:		Main transformation function, takes a 2d numpy array vector.
		Jinv_system:	Inverse of Jacobian of the output of F_system, taked a 2d numpy array vector.
		x_vec0:			2d numpy array vector of the form [[x],[y]].
		tol:			Float, desired accuracy of roots.
		maxiter:		int, iteration limit of root-finding function.
	
	OUTPUT:
		x_vec:			2d numpy array vector of root.
	
	Multi-dimensional root-finding function.
	"""
	x_vec = x_vec0
	for i in range(0,maxiter):
		dx = matrix_mult(Jinv_system(),F_system(x_vec))
		x = F_system(x_vec)[0,0]
		y = F_system(x_vec)[1,0]
		if np.sqrt(np.power(x,2)+np.power(y,2)) <= tol:
			return x_vec 
		else:
			x_vec = np.add(deepcopy(x_vec), -1*dx)
	return "Error: Iteration limit exceeded."


def HarryPlotter(a):
	"""
	"The Chosen function"

	INPUT:
		f: function, this is the function graphed
		xlo: float, lower bound of the graph of the function f
		xhi: float, upper bound of the graph of the function f

	OUTPUT:
		graph of function f over domain xlo-xhi with mesh size dx (definied within the function)
	"""
	x_vals = a[0]                #Determine x and y values
	y_vals= a[1]                 #Plot generator function            
	plt.plot(x_vals,y_vals, label = 'Data')
	plt.xlim(20000,21000)
	plt.grid()
	plt.axhline(color='black')
	plt.axvline(color='black')
	plt.xlabel('Wavenumber (v)')
	plt.ylabel('Optical Intensity (S(v))')
	plt.legend()


def partc1(a):
	myfunc = interp1d(a[0], a[1], kind = 'cubic', bounds_error = False)
	f, xlo, xhi, dx = myfunc, 20000, 21000, 1
	x_vals = np.arange(xlo, xhi + dx, dx)
	horiz = [0.141405, 0.5346]		
	for item in horiz:
		i = 0
		for num in myfunc(x_vals):
			if abs(item-num) <= 0.0001:
				print('At optical intensity', str(item), 'the wavenumber is', x_vals[i])
			i+=1


def main():
	"""
	INPUTS:
		NONE
	OUTPUTS:
		Values of c1 and c2
	"""
	points = [[[50],[50]]]
	for vec in points:
		vec = np.array(vec)
		print(vec)
		F_system, Jinv_system, x_vec0, tol, maxiter = F, Jinv, np.array(vec), 1e-5, 1000
		if type(rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter))== str:
			print(rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter))
		else:
			coordinates = rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter)
			x_val = coordinates[0,0]
			y_val = coordinates[1,0]
			print("For the coordinate pair: (" + str(vec[0,0]) + "," + str(vec[1,0]) + ") the system converges at (" + str(x_val) + "," + str(y_val) + ")" )


def L(v, gamma, v0):
	a = 1/np.pi
	b = np.power(v-v0,2)
	c = gamma*0.5
	return a* c/(b+np.power(c,2))


def cplot(func, xlo, xhi, dx, c1, c2, v01, v02, gamma1, gamma2):                      
	x_vals=np.arange(xlo,xhi+dx,dx)                 			#Determine x and y values
	y_vals= func(c1, c2, v01, v02, gamma1, gamma2, x_vals)		#Plot generator function
	plt.plot(x_vals,y_vals, label = 'Model', linestyle = ':')
	plt.title('Optical Intensiy: Guess vs. Data')
	plt.legend()


def ModelSpectrum(c1, c2, v01, v02, gamma1, gamma2, v):
	return c1 * L(v, gamma1, v01)+ c2 * L(v, gamma2, v02)


a, c1, c2, v01, v02, gamma1, gamma2, xlo, xhi, dx, func= datafinder(), 20.344115996923755, 60.56593635282718, 20350, 20825, 63.48, 37.45, 20000, 21000, 1, ModelSpectrum
cplot(func, xlo, xhi, dx, c1, c2, v01, v02, gamma1, gamma2)
HarryPlotter(a)
plt.show()


#------------------------------
#Part d


x = np.array([c1,c2,v01,v02,gamma1, gamma2])
def ModelSpectrum2(x,v):
	c1 = x[0]
	c2 = x[1]
	v01 = x[2]
	v02 = x[3]
	gamma1 = x[4]
	gamma2 = x[5]
	return c1 * L(v, gamma1, v01)+ c2 * L(v, gamma2, v02)


#-------------------------------
#Part e


v, Sv, func = a[0], a[1], ModelSpectrum2
def Residuals(x, v, Sv):
	return Sv - ModelSpectrum2(x,v)


def residualplot(x, v, Sv,func):
	x_vals = np.arange(20000,21001,1)
	plt.plot(v, Sv, label = "Data")
	plt.scatter(v, Residuals(x,v,Sv), label = "Residuals")
	plt.plot(x_vals, func(x,x_vals),label = "Model")
	plt.title(r"Residuals of $c_1L_1(v)+c_2L_2(v)$", fontdict = {'verticalalignment': 'bottom', 'horizontalalignment':'center'})
	plt.xlabel('Wavenumber (v)')
	plt.ylabel('Optical Intensity (S(v))')
	plt.axhline(color = 'black')
	plt.legend()
	plt.grid()
	plt.xlim(20000,21000)
	plt.show()


residualplot(x,v,Sv,func)


#--------------------------------
#Part f


myres = Residuals


def leastsqplotf(x0, v, Sv, myres):
	goodv = np.arange(20000, 21001, 1)
	res = leastsq(myres, x0, args = (v, Sv))
	x1 = res[0]
	yvals = ModelSpectrum2(x1,goodv)
	plt.plot(goodv, yvals, label = 'Least Squares')
	plt.plot(v, Sv, label = "Data")
	plt.xlabel('Wavenumber (v)')
	plt.ylabel('Optical Intensity (S(v))')
	plt.title('Least Squares Fitting')
	plt.axhline(color = 'black')
	plt.legend()
	plt.grid()
	plt.xlim(20000,21000)
	plt.show()

leastsqplotf(x,v,Sv,myres)


#---------------------------------
# Part g


def leastsqplotg(x0, v, Sv, myres):
	goodv = np.arange(20000, 21001, 1)
	res = leastsq(myres, x0, args = (v, Sv))
	x1 = res[0]
	yvals1 = myres(x1,v,Sv)
	plt.scatter(v, yvals1, label = 'Least Squares Residuals')
	plt.ylim(-2e-6,2e-6)
	plt.xlabel('Wavenumber (v)')
	plt.ylabel('Optical Intensity (S(v))')
	plt.title('Least Squares Fitting Residuals')
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.axhline(color = 'black')
	plt.legend()
	plt.grid()
	plt.xlim(20000,21000)
	plt.show()

#leastsqplotg(x,v,Sv,myres)

#----------------------------------
#Part h

def leastsqploth(x0, v, Sv, myres):
	goodv = np.arange(20000, 21001, 1)
	res = leastsq(myres, x0, args = (v, Sv))
	x1 = res[0]
	yvals = ModelSpectrum2(x1,goodv)
	plt.plot(goodv, yvals, label = 'Least Squares', color = 'violet')
	plt.scatter(v, Sv, label = "Data", color = 'green')
	plt.xlabel('Wavenumber (v)')
	plt.ylabel('Optical Intensity (S(v))')
	plt.title('Least Squares Fitting vs. Original')
	plt.axhline(color = 'black')
	plt.legend()
	plt.grid()
	plt.xlim(20000,21000)
	plt.show()

leastsqploth(x,v,Sv,myres)