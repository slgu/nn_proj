#just a single renet layer to simulate a convolution-poll layer
from nn import *
from util import *
import numpy as np
class ReNet(object):
    def __init__(self, input, w, h, c, wp, hp, d):
        '''
        input: last layer output, or the initial image
        w: input width
        h: input height
        c: input channel num
        wp: width patch length
        hp: height patch length
        d: hidden unit dimension
        After the four direction rnn the output will be size [(w / wp), (h / hp), 2 * d]
        '''
        l_to_r = ReNetDir(input, w, h, c, wp, hp, d, 0)
        r_to_l = ReNetDir(input, w, h, c, wp, hp, d, 1)
        # stack together
        output1 = T.concatenate([l_to_r.output, r_to_l.output], axis=2)
        u_to_d  = ReNetDir(output1, w / wp, h / hp, 2 * d, 1, 1, d, 2)
        d_to_u = ReNetDir(output1, w / wp, h / hp, 2 * d, 1, 1, d, 3)
        self.output = T.concatenate([u_to_d.output, d_to_u.output], axis=2)
        self.test = theano.function(inputs=[input], outputs=self.output)

class ReNetDir(object):
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
            0: left to right
            1: right to left
            2: up to down
            3: down to up
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
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (wp * hp * c, d))
                                .astype(theano.config.floatX))
        self.U_r = theano.shared(name='W_r',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (d, d))
                                .astype(theano.config.floatX))
        self.b_r = theano.shared(name='b_r',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))
        self.W_u = theano.shared(name='W_u',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (wp * hp * c, d))
                                .astype(theano.config.floatX))
        self.U_u = theano.shared(name='W_u',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (d, d))
                                .astype(theano.config.floatX))
        self.b_u = theano.shared(name='b_u',
                               value=numpy.zeros(d,
                               dtype=theano.config.floatX))

        # gated recurrent unit
        def recurrence(x_t, h_t_prev):
            u_t = T.nnet.sigmoid(T.dot(x_t, self.W_u) + T.dot(h_t_prev, self.U_u) + self.b_u)
            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_r) + T.dot(h_t_prev, self.U_r) + self.b_r)
            _h_t = T.tanh(T.dot(x_t, self.W) + T.dot(r_t * h_t_prev, self.U) + self.b)
            h_t = (1 - u_t) * h_t_prev + u_t * _h_t
            return [h_t, h_t]

        res = []
        # construct batch vector
        if dir == 0 or dir == 1:
            for i in range(w / wp):
                horizontal_seq = []
                for j in range(h / hp):
                    #construct vector
                    tmp = []
                    for k in range(0, wp):
                        for o in range(0, hp):
                            x = i * wp + k
                            y = j * hp + o
                            tmp.append(input[x][y])
                    in_x = T.concatenate(tmp)
                    horizontal_seq.append(in_x)
                #may down to top if dir == 1
                if dir == 1:
                    horizontal_seq = list(reversed(horizontal_seq))
                horizontal_seq = T.stack(horizontal_seq)
                res.append(horizontal_seq)
        elif dir == 2 or dir == 3:
            for j in range(h / hp):
                vertical_seq = []
                for i in range(w / wp):
                    #construct vector
                    tmp = []
                    for k in range(0, wp):
                        for o in range(0, hp):
                            x = i * wp + k
                            y = j * hp + o
                            tmp.append(input[x][y])
                    in_x = T.concatenate(tmp)
                    vertical_seq.append(in_x)
                if dir == 3:
                    vertical_seq = list(reversed(vertical_seq))
                vertical_seq = T.stack(vertical_seq)
                res.append(vertical_seq)
        else:
            print("error input dir")
            exit(0)

        # apply recurrence to get hidden expression
        in_h = []
        for item in res:
            [h, s], _ = theano.scan(fn=recurrence,
                                    sequences=item,
                                    outputs_info=[self.h0, None],
                                    n_steps=item.shape[0])
            in_h.append(h)

        #output the hidden variable
        self.input = input
        self.output = T.stack(in_h)

        #test function to check the rightness of the renet
        self.test = theano.function(inputs=[input], outputs=self.output)

if __name__ == '__main__':
    x = T.matrix('x')
    xx = x.reshape((8, 8, 3))
    a = ReNet(xx,8,8,3,2,2,7)
    arr = np.asarray([1]*192, dtype=theano.config.floatX).reshape((8,8,3))
    print(a.test(arr).shape)
