# By Shatha Jaradat
# KTH - Royal Institute of Technology
# 2018

# Parts of the code related to finetunning the models was taken from the following link and customised to my needs:
#https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf

import argparse
import os
import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import Processing.Evaluator_MultiClass.Evaluator_MultiClass as evaluator_multiClass
import DataPreprocessing.InstagramDataHelper.InstagramDataHelper as fashionDataset
import DynamicConnections.MultiClass_DynamicPruning.FineTuneModel as multiClass_DynamicPruning
import DynamicLayers.MultiClass_DynamicLayers.FineTuneModel as multiClass_DynamicLayers


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/data/')
parser.add_argument('--pruningOrLayers', help='Model type - dynamic pruning (1) or dynamic layers (2)', default='1')
parser.add_argument('--annotation', metavar='DIR',
    help='path to annotation directory of dataset', default='/data/updated_annotations_eval/Anno/')
parser.add_argument('--eval', metavar='DIR',
                    help='path to evaluation directory of dataset', default='/data/updated_annotations_eval/Eval/')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='fine tune pre-trained model', default='true')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    num_category_classes = 50

    print("num_category_classes = '{}'".format(num_category_classes))
    print("Number of Sub Categories detected in image is Static (2)")
    print("Number of Attributes detected in image is Static (8)")
    print("Number of Style Information detected in image is Static (1)")

    # create model
    if args.finetune:
        print("=> using pre-trained model '{}'".format(args.arch))
        original_model = models.__dict__[args.arch](pretrained=True)
        if args.pruningOrLayers == "1":
            # Dynamic Pruning
            model = multiClass_DynamicPruning(original_model, args.arch)
        else:
            # Dynamic Layers
            model = multiClass_DynamicLayers(original_model, args.arch)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('resnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #cudnn.benchmark = True

    # Data loading code
    transformed_dataset = fashionDataset(args.rootdir,
                                         args.imagesdir,
                                            'train')

    train_loader = DataLoader(transformed_dataset, batch_size=1,
                        shuffle=False, num_workers=4, pin_memory=True)

    validation_dataset = fashionDataset(args.rootdir,
                                         args.imagesdir,
                                          'val')

    val_loader = DataLoader(validation_dataset, batch_size=1,
                        shuffle=False, num_workers=4, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        evaluator_multiClass.validate(val_loader, model, criterion,args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        evaluator_multiClass.adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        evaluator_multiClass.train(train_loader, model, criterion, optimizer, epoch,args.print_freq)

        # evaluate on validation set
        prec1_category,prec1_subcategory, prec1_attributes = evaluator_multiClass.validate(val_loader, model, criterion,args.print_freq)

        # Can be done for attributes as well  = Shathat
        # remember best prec@1 and save checkpoint
        is_best = prec1_category > best_prec1
        best_prec1 = max(prec1_category, best_prec1)
        evaluator_multiClass.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


if __name__ == '__main__':
    main()