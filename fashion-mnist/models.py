import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_units),
            l4=L.Linear(None, n_units),
            l5=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        h = self.activation(self.l3(h))
        h = self.activation(self.l4(h))
        return self.l5(h)


class CNN(chainer.Chain):

    def __init__(self, n_out=10):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(None, 32, ksize=3),
            conv2=L.Convolution2D(None, 32, ksize=3),
            conv3=L.Convolution2D(None, 32, ksize=3),
            fc1=L.Linear(None, 500),
            fc2=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), ksize=2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class BatchConv2D(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize, stride=1, pad=0, activation=F.relu):
        super(BatchConv2D, self).__init__(
            conv=L.Convolution2D(ch_in, ch_out, ksize, stride, pad),
            bn=L.BatchNormalization(ch_out),
        )
        self.activation=activation

    def __call__(self, x):
        h = self.bn(self.conv(x))
        if self.activation is None:
            return h
        return F.relu(h)


class VGG(chainer.Chain):

    def __init__(self, n_out=10):
        super(VGG, self).__init__(
            bconv1_1=BatchConv2D(None, 64, ksize=3, stride=1, pad=1),
            bconv1_2=BatchConv2D(None, 64, ksize=3, stride=1, pad=1),
            bconv2_1=BatchConv2D(None, 128, ksize=3, stride=1, pad=1),
            bconv2_2=BatchConv2D(None, 128, ksize=3, stride=1, pad=1),
            bconv3_1=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            bconv3_2=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            bconv3_3=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            bconv3_4=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            fc4=L.Linear(None, 1024),
            fc5=L.Linear(None, 1024),
            fc6=L.Linear(None, n_out),
        )

    def __call__(self, x):
        h = self.bconv1_1(x)
        h = self.bconv1_2(h)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25)
        h = self.bconv2_1(h)
        h = self.bconv2_2(h)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25)
        h = self.bconv3_1(h)
        h = self.bconv3_2(h)
        h = self.bconv3_3(h)
        h = self.bconv3_4(h)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25)
        h = F.relu(self.fc4(F.dropout(h)))
        h = F.relu(self.fc5(F.dropout(h)))
        return self.fc6(h)
