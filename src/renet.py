#just a single renet layer to simulate a convolution-poll layer
from nn import *
from util import *
import numpy as np
class ReNet:
    def __init__(self, input, w, h, c, wp, hp, d, dir):
        '''
        input: last layer output, or the initial image
        w: input width
        h: input height
        c: input channel num
        wp: width patch length
        hp: height patch length
        d: hidden unit dimension
        dir: direction to apply rnn to get the hidden expression
        After the four direction rnn the output will be size [(w / wp), (h / hp), 2 * d]
        '''
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(d,
                                dtype=theano.config.floatX))
        self.W = theano.shared(name='W',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (wp * hp * c, d))
                                .astype(theano.config.floatX))
        self.U = theano.shared(name='U',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (d, d))
                                .astype(theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))
        self.W_r = theano.shared(name='W_r',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))
        self.U_r = theano.shared(name='U_r',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))
        self.b_r = theano.shared(name='b_r',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))
        self.W_u = theano.shared(name='W_u',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))
        self.U_u = theano.shared(name='U_u',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))
        self.b_u = theano.shared(name='b_u',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))

        # gated recurrent unit
        def recurrence(x_t, h_t_prev):
            u_t = T.nnet.sigmoid(T.dot(self.W_u, x_t) + T.dot(self.U_u, h_t_prev) + self.b_u)
            r_t = T.nnet.sigmoid(T.dot(self.W_r, x_t) + T.dot(self.U_r, h_t_prev) + self.b_r)
            _h_t = T.tanh(T.dot(self.W, x_t) + T.dot(self.U, r_t * h_t_prev) + self.b)
            h_t = (1 - u_t) * h_t_prev + u_t * _h_t
            return [h_t, h_t]

        #construct batch vector
        res = []
        for i in range(w / wp):
            vertical_seq = []
            for j in range(h / hp):
                #construct vector
                tmp = []
                for k in range(0, wp):
                    for o in range(0, hp):
                        x = i * wp + k
                        y = j * hp + o
                        tmp.append(xx[x][y])
                in_x = T.concatenate(tmp)
                vertical_seq.append(in_x)
            vertical_seq = T.stack(vertical_seq)
            res.append(vertical_seq)


        #apply recurrence to get hidden expression
        in_h = []
        for item in res:
            [h, s], _ = theano.scan(fn=recurrence,
                                    sequences=item,
                                    outputs_info=[self.h0, None],
                                    n_steps=item.shape[0])
            in_h.append(h)


        #output the hidden variable
        self.output = T.stack(in_h)

        #test function to check the rightness of the renet
        self.test = theano.function(inputs=[input], outputs=self.output)

if __name__ == '__main__':
    x = T.matrix('x')
    xx = x.reshape((4, 4, 3))
    a = ReNet(xx,4,4,3,2,2,3, 0)
    arr = np.asarray([1]*48, dtype=theano.config.floatX).reshape((4,4,3))
    print(a.test(arr))
