import numpy
import theano
import theano.tensor as T

x = T.fmatrix('x')
y = T.lvector('y')

b = theano.shared(numpy.zeros((10,)), name = 'b')
w = theano.shared(numpy.zeros((784,10)), name = 'w')

p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)
print p_y_given_x

get_p_y_given_x = theano.function(inputs=[x], outputs=p_y_given_x)
print get_p_y_given_x
print 'Probability that x is of class %i is %f' % (i, get_p_y_given_x(x_value)[i])
