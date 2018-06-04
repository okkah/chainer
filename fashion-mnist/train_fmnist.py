import argparse
from functools import partial
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extensions

from utils.fmnist_reader import load_dataset
from output_fmnist import output
from preprocess import transform
from models import *


def main():
    parser = argparse.ArgumentParser(description='Chainer example: Fashion-MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of units')
    parser.add_argument('--optimizer', '-op',
                        choices=('SGD', 'MomentumSGD', 'NesterovAG', 'AdaGrad',
                                 'AdaDelta', 'RMSprop', 'Adam'),
                        default='MomentumSGD', help='optimization type')
    parser.add_argument('--model', '-m',
                        choices=('MLP', 'CNN', 'VGG'),
                        default='MLP', help='model type')
    parser.add_argument('--activation', '-a',
                        choices=('sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu'),
                        default='relu')
    parser.add_argument('--random_angle', type=float, default=15.0)
    parser.add_argument('--expand_ratio', type=float, default=1.2)
    parser.add_argument('--crop_size', type=int, nargs='*', default=[28, 28])
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('unit: {}'.format(args.unit))
    print('batch-size: {}'.format(args.batchsize))
    print('epoch: {}'.format(args.epoch))
    print('optimizer: {}'.format(args.optimizer))
    print('model type: {}'.format(args.model))
    print('activation: {}'.format(args.activation))
    print('')

    # Activation
    if args.activation == 'sigmoid':
        activation = F.sigmoid
    elif args.activation == 'tanh':
        activation = F.tanh
    elif args.activation == 'relu':
        activation = F.relu
    elif args.activation == 'leaky_relu':
        activation = F.leaky_relu
    elif args.activation == 'elu':
        activation = F.elu

    # Model
    if args.model == 'MLP':
        model = MLP(args.unit, 10, activation)
    elif args.model == 'CNN':
        model = CNN(10)
    elif args.model == 'VGG':
        model = VGG(10)

    model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Optimizer
    if args.optimizer == 'SGD':
        optimizer = chainer.optimizers.SGD()
    elif args.optimizer == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD()
    elif args.optimizer == 'NesterovAG':
        optimizer = chainer.optimizers.NesterovAG()
    elif args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad()
    elif args.optimizer == 'AdaDelta':
        optimizer = chainer.optimizers.AdaDelta()
    elif args.optimizer == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop()
    elif args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load_Dataset
    train, test = load_dataset()

    # Output_Original_Images
    #output(train=train, file="./result/fmnist_original.png")
    #sys.exit()

    # Preprocess
    mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
    std = np.std([x for x, _ in train], axis=(0, 2, 3))

    train_transform = partial(
        transform, mean=mean, std=std, random_angle=args.random_angle,
        crop_size=args.crop_size, train=True)
    test_transform = partial(transform, mean=mean, std=std, train=False)

    train = TransformDataset(train, train_transform)
    test = TransformDataset(test, test_transform)
    
    # Output_Transformed_Images
    #output(train=train, file="./result/fmnist_transform.png")
    #sys.exit()
    
    # Iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    # Extensions
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run_Training
    trainer.run()

if __name__ == '__main__':
    main()
