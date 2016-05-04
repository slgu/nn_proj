#just a single renet layer to simulate a convolution-poll layer
from nn import *
from util import *
import numpy as np
class ReNet(object):
    def __init__(self, input, batch_size, w, h, c, wp, hp, d):
        '''
        input: last layer output, or the initial image
        w: input width
        h: input height
        batch_size: the number of the input sample
        c: input channel num
        wp: width patch length
        hp: height patch length
        d: hidden unit dimension
        After the four direction rnn the output will be size [batch_size, (w / wp), (h / hp), 2 * d]
        '''
        # first get left to right right to left hidden expression then stack
        l_to_r = ReNetDir(input, batch_size, w, h, c, wp, hp, d, 2)
        r_to_l = ReNetDir(input, batch_size, w, h, c, wp, hp, d, 3)
        # stack together
        output1 = T.concatenate([l_to_r.output, r_to_l.output], axis=3)
        # up to down and down to up
        u_to_d = ReNetDir(output1, batch_size, w / wp, h / hp, 2 * d, 1, 1, d, 0)
        d_to_u = ReNetDir(output1, batch_size, w / wp, h / hp, 2 * d, 1, 1, d, 1)
        # get the output
        self.output = T.concatenate([u_to_d.output, d_to_u.output], axis=3)
        self.test = theano.function([input], self.output)
        # get the paramters
        self.params = l_to_r.params + r_to_l.params + u_to_d.params + d_to_u.params

class ReNetDir(object):
    def __init__(self, input, batch_size, w, h, c, wp, hp, d, dir):
        '''
        input: last layer output, or the initial image
        w: input width
        h: input height
        c: input channel num
        batch_size: the number of the input sample
        wp: width patch length
        hp: height patch length
        d: hidden unit dimension
        dir: direction to apply rnn to get the hidden expression
            0: left to right
            1: right to left
            2: up to down
            3: down to up
        After the four direction rnn the output will be size [(batch_size, w / wp), (h / hp), 2 * d]
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

        self.params = [self.h0, self.W, self.U, self.b, self.W_r, self.U_r, self.b_r, self.W_u, self.U_u, self.b_u]
        # gated recurrent unit
        # x can be batch_size * I/J * patch vector dimension
        # h_t_prev batch_size * I/J * h_t_prev
        def gru_recurrence(x_t, h_t_prev):
            u_t = T.nnet.sigmoid(T.dot(x_t, self.W_u) + T.dot(h_t_prev, self.U_u) + self.b_u)
            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_r) + T.dot(h_t_prev, self.U_r) + self.b_r)
            _h_t = T.tanh(T.dot(x_t, self.W) + T.dot(r_t * h_t_prev, self.U) + self.b)
            h_t = (1 - u_t) * h_t_prev + u_t * _h_t
            return [h_t, h_t]


        if dir == 0 or dir == 1:
            # left to right or right to left
            # use reshape and transpoe to get each [wp * hp * c] patch vector
            seq_x = input.reshape((batch_size, w, h / hp, hp * c)).transpose(2,0,1,3).reshape((h/hp, batch_size, w/wp, wp * hp * c))
            # if right to left we need to reverse the data
            if dir == 1:
                seq_x  = seq_x[::-1]
            multi_h0 = T.tile(self.h0, batch_size * w / wp).reshape((batch_size, w / wp, d))
            [seq_h, s], _ = theano.scan(fn=gru_recurrence,
                                    sequences=seq_x,
                                    outputs_info=[multi_h0, None],
                                    n_steps=seq_x.shape[0])
            # reverse back
            if dir == 1:
                seq_h = seq_h[::-1]
            self.output = seq_h.transpose(1,2,0,3)
            print("output done")
        elif dir == 2 or dir == 3:
            # up to down or down to up
            seq_x = input.transpose(0,2,1,3).reshape((batch_size, h, w / wp, wp * c)).transpose(2,0,1,3).reshape((w/hp, batch_size, h/wp, wp * hp * c))
            if dir == 3:
                seq_x  = seq_x[::-1]
            multi_h0 = T.tile(self.h0, batch_size * h / hp).reshape((batch_size, h / hp, d))
            [seq_h, s], _ = theano.scan(fn=gru_recurrence,
                                    sequences=seq_x,
                                    outputs_info=[multi_h0, None],
                                    n_steps=seq_x.shape[0])
            if dir == 3:
                seq_h = seq_h[::-1]
            self.output = seq_h.transpose(1,0,2,3)
            print("output done")

if __name__ == '__main__':
    x = T.matrix('x')
    xx = x.reshape((4, 32, 32, 3))
    a = ReNet(xx,4,32,32,3,2,2,14)
    arr = np.asarray([1]*4*32*32*3, dtype=theano.config.floatX).reshape((4,32,32,3))
    print(a.test(arr).shape)
