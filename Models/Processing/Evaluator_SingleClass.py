import time
import torch
import random
from torch.autograd import Variable
import shutil

###### Single class scenarios -- Categories and Attributes Only with no subcategories
###### Was applied on DeepFashion Dataset
###### Contains methods for training the model, accuracy, saving the model, etc.
class Evaluator_SingleClass(object):

    def train(self,train_loader, model, criterion, optimizer, epoch,print_freq):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_category = AverageMeter()
        losses_attributes= AverageMeter()

        top1_category = AverageMeter()
        top3_category = AverageMeter()
        top5_category = AverageMeter()

        top1_attributes = AverageMeter()
        top3_attributes = AverageMeter()
        top5_attributes = AverageMeter()

        top1_recall_attributes = AverageMeter()
        top3_recall_attributes = AverageMeter()
        top5_recall_attributes = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()

        # image of size 1 * 3 * 224 * 224
        for i, (image, category, attributes, categoryId, lstAttributeIndices) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            input_var = torch.autograd.Variable(image)
            # Float Tensor of size 1 * 50 with value 1 for the correct category and 0 otherwise
            target_category_var = torch.autograd.Variable(category)
            # Float Tensor of size 1 * 1000 with value 1 for the correct attribute and 0 otherwise
            target_attributes_var = torch.autograd.Variable(attributes)

            target_category_var = target_category_var.cuda(async=True)
            target_attributes_var = target_attributes_var.cuda(async=True)

            # Randomly send supporting category or supporting attribute
            categoryItemValue = categoryId
            attributeItemValue = -1
            attributeItem2 = -1
            if len(lstAttributeIndices) > 0:
                attributeItemValue = random.choice(lstAttributeIndices)

            lstItemsToChoose = []
            for item in lstAttributeIndices:
                if item[0] != int(attributeItemValue):
                    lstItemsToChoose.append(item[0])

            if len(lstItemsToChoose) > 0:
                attributeItem2 = random.choice(lstItemsToChoose)


            # compute output
            # send the supporting item type or value
            lstOutputs = model(input_var, categoryItemValue, attributeItemValue, attributeItem2)


            output_category =  lstOutputs[0]
            output_attributes = lstOutputs[1]

            output_category = output_category.cuda(async=True)
            output_attributes = output_attributes.cuda(async=True)

            loss_category = criterion(output_category, target_category_var)
            loss_attributes = criterion(output_attributes, target_attributes_var.squeeze(1))

            # measure accuracy and record loss
            prec1_category, prec3_category, prec5_category = self.accuracy_category(output_category.data, target_category_var, [categoryId], topk=(3,5, 10))
            prec1_attributes, prec3_attributes, prec5_attributes, recall1_attributes, recall3_attributes, recall5_attributes = self.accuracy_multiLabel(output_attributes.data, target_attributes_var,lstAttributeIndices, topk=(3, 5,10))


            losses_category.update(loss_category.data[0], input_var.size(0))
            losses_attributes.update(loss_attributes.data[0], input_var.size(0))

            top1_category.update(prec1_category, input_var.size(0))
            top1_attributes.update(prec1_attributes, input_var.size(0))
            top1_recall_attributes.update(recall1_attributes, input_var.size(0))

            top3_category.update(prec3_category, input_var.size(0))
            top3_attributes.update(prec3_attributes, input_var.size(0))
            top3_recall_attributes.update(recall3_attributes, input_var.size(0))


            top5_category.update(prec5_category, input_var.size(0))
            top5_attributes.update(prec5_attributes, input_var.size(0))
            top5_recall_attributes.update(recall5_attributes, input_var.size(0))


            # compute gradient and do SGD step
            optimizer.zero_grad()

            var_loss = Variable(loss_category.data, requires_grad=True)

            var_loss.backward(retain_graph=True)
            var_loss_attr = Variable(loss_attributes.data, requires_grad=True)
            var_loss_attr.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                with open("/output/results_withtext_singleItem_training.txt", "a") as file_results:

                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                    'Recall@1 Attributes {top1_recall_attributes.val:.3f} ({top1_recall_attributes.avg:.3f})\t'
                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                    'Recall@3 Attributes {top3_recall_attributes.val:.3f} ({top3_recall_attributes.avg:.3f})\t'
                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                    'Recall@5 Attributes {top5_recall_attributes.val:.3f} ({top5_recall_attributes.avg:.3f})\t'
                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss_category =losses_category, loss_attributes = losses_attributes,
                    top1_category =top1_category,top1_attributes = top1_attributes, top1_recall_attributes = top1_recall_attributes,
                    top3_category =top3_category,top3_attributes = top3_attributes, top3_recall_attributes = top3_recall_attributes,
                    top5_category=top5_category , top5_attributes=top5_attributes, top5_recall_attributes = top5_recall_attributes
                    ))
                    file_results.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                    'Recall@1 Attributes {top1_recall_attributes.val:.3f} ({top1_recall_attributes.avg:.3f})\t'
                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                    'Recall@3 Attributes {top3_recall_attributes.val:.3f} ({top3_recall_attributes.avg:.3f})\t'
                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                    'Recall@5 Attributes {top5_recall_attributes.val:.3f} ({top5_recall_attributes.avg:.3f})\t'
                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss_category =losses_category, loss_attributes = losses_attributes,
                    top1_category =top1_category,top1_attributes = top1_attributes , top1_recall_attributes = top1_recall_attributes,
                    top3_category =top3_category,top3_attributes = top3_attributes, top3_recall_attributes = top3_recall_attributes,
                    top5_category=top5_category ,top5_attributes= top5_attributes, top5_recall_attributes = top5_recall_attributes
                    ))
                    file_results.write("\n")

    # lstMultiLabel_GT contains the ground truth multi labels
    def accuracy_multiLabel(self,output, target, lstMultiLabel_GT, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        gt = []
        for item in lstMultiLabel_GT:
            gt.append(int(item))

        res = []

        # Top 1,3,5 multi-label
        lstTop1Pred = [int(pred[0])]
        lstTop3Pred = [int(pred[0]), int(pred[1]), int(pred[2])]
        lstTop5Pred = [int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]), int(pred[4])]

        # intersection for top 1
        intersection_top1 = [v for v in lstTop1Pred if v in gt]
        intersection_top3 = [v for v in lstTop3Pred if v in gt]
        intersection_top5 = [v for v in lstTop5Pred if v in gt]

        val_top1 = float(len(intersection_top1)) / float(len(pred))
        val_top3 = float(len(intersection_top3)) / float(len(pred))
        val_top5 = float(len(intersection_top5)) / float(len(pred))

        val_recall_top1 = float(len(intersection_top1)) / float(len(lstMultiLabel_GT))
        val_recall_top3 = float(len(intersection_top3)) / float(len(lstMultiLabel_GT))
        val_recall_top5 = float(len(intersection_top5)) / float(len(lstMultiLabel_GT))

        res.append(val_top1 * 100)
        res.append(val_top3* 100)
        res.append(val_top5* 100)

        res.append(val_recall_top1* 100)
        res.append(val_recall_top3* 100)
        res.append(val_recall_top5* 100)

        return res

    # lstMultiLabel_GT contains the ground truth multi labels
    def accuracy_category(self,output, target, lstMultiLabel_GT, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        gt = []
        for item in lstMultiLabel_GT:
            gt.append(int(item))

        res = []

        # Top 1,3,5 multi-label
        lstTop1Pred = [int(pred[0])]
        lstTop3Pred = [int(pred[0]), int(pred[1]), int(pred[2])]
        lstTop5Pred = [int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]), int(pred[4])]


        # intersection for top 1
        intersection_top1 = [v for v in lstTop1Pred if v in gt]
        intersection_top3 = [v for v in lstTop3Pred if v in gt]
        intersection_top5 = [v for v in lstTop5Pred if v in gt]


        val_top1 = float(len(intersection_top1)) / float(len(pred))
        val_top3 = float(len(intersection_top3)) / float(len(pred))
        val_top5 = float(len(intersection_top5)) / float(len(pred))

        val_recall_top1 = float(len(intersection_top1)) / float(len(lstMultiLabel_GT))
        val_recall_top3 = float(len(intersection_top3)) / float(len(lstMultiLabel_GT))
        val_recall_top5 = float(len(intersection_top5)) / float(len(lstMultiLabel_GT))


        res.append(val_top1* 100)
        res.append(val_top3* 100)
        res.append(val_top5* 100)

        res.append(val_recall_top1)
        res.append(val_recall_top3)
        res.append(val_recall_top5)

        return res

    def validate(self,val_loader, model, criterion, print_freq):
        batch_time = AverageMeter()

        losses_category= AverageMeter()
        losses_attributes = AverageMeter()

        top1_category= AverageMeter()
        top5_category = AverageMeter()

        top3_category= AverageMeter()
        top3_attributes = AverageMeter()

        top1_attributes = AverageMeter()
        top5_attributes =  AverageMeter()

        top1_recall_attributes = AverageMeter()
        top3_recall_attributes = AverageMeter()
        top5_recall_attributes = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (image, category, attributes, categoryId, lstAttributeIndices) in enumerate(val_loader):        #target = target.cuda(async=True)
            input_var = torch.autograd.Variable(image, volatile=True)

            # Float Tensor of size 1 * 50 with value 1 for the correct category and 0 otherwise
            target_category_var = torch.autograd.Variable(category, volatile=True)
            # Float Tensor of size 1 * 1000 with value 1 for the correct attribute and 0 otherwise
            target_attributes_var = torch.autograd.Variable(attributes)

            target_category_var = target_category_var.cuda(async=True)
            target_attributes_var = target_attributes_var.cuda(async=True)

            attributeItemValue = -1
            categoryItemValue = categoryId
            attributeItem2 = -1
            if len(lstAttributeIndices) > 0:
                attributeItemValue = random.choice(lstAttributeIndices)

            lstItemsToChoose = []
            for item in lstAttributeIndices:
                if item[0] != int(attributeItemValue):
                    lstItemsToChoose.append(item[0])

            if len(lstItemsToChoose) > 0:
                attributeItem2 = random.choice(lstItemsToChoose)


            # compute output
            lstOutputs = model(input_var, categoryItemValue, attributeItemValue, attributeItem2)

            output_category =  lstOutputs[0]
            output_attributes =lstOutputs[1]

            output_category = output_category.cuda(async=True)
            output_attributes = output_attributes.cuda(async=True)

            loss_category = criterion(output_category, target_category_var)
            loss_attributes = criterion(output_attributes, target_attributes_var.squeeze(1))

            # measure accuracy and record loss
            # measure accuracy and record loss
            prec1_category, prec3_category, prec5_category = self.accuracy_category(output_category.data, target_category_var,[categoryId], topk=(3, 5,10))
            prec1_attributes, prec3_attributes, prec5_attributes, recall1_attributes, recall3_attributes, recall5_attributes = self.accuracy_multiLabel(output_attributes.data, target_attributes_var, lstAttributeIndices, topk=(3, 5,10))

            losses_category.update(loss_category.data[0], input_var.size(0))
            losses_attributes.update(loss_attributes.data[0], input_var.size(0))

            top1_category.update(prec1_category, input_var.size(0))
            top1_attributes.update(prec1_attributes, input_var.size(0))
            top1_recall_attributes.update(recall1_attributes, input_var.size(0))


            top3_category.update(prec3_category, input_var.size(0))
            top3_attributes.update(prec3_attributes, input_var.size(0))
            top3_recall_attributes.update(recall3_attributes, input_var.size(0))


            top5_category.update(prec5_category, input_var.size(0))
            top5_attributes.update(prec5_attributes, input_var.size(0))
            top5_recall_attributes.update(recall5_attributes, input_var.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                with open("/output/results_withtext_singleItem_validation.txt", "a") as file_results:

                    print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                    'Recall@1 Attributes {top1_recall_attributes.val:.3f} ({top1_recall_attributes.avg:.3f})\t'
                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                    'Recall@3 Attributes {top3_recall_attributes.val:.3f} ({top3_recall_attributes.avg:.3f})\t'
                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                    'Recall@5 Attributes {top5_recall_attributes.val:.3f} ({top5_recall_attributes.avg:.3f})\t'
                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                        i, len(val_loader),  batch_time=batch_time,
                    data_time=batch_time, loss_category =losses_category, loss_attributes = losses_attributes,
                    top1_category =top1_category,top1_attributes = top1_attributes,top1_recall_attributes = top1_recall_attributes,
                    top3_category =top3_category,top3_attributes = top3_attributes, top3_recall_attributes = top3_recall_attributes,
                    top5_category=top5_category,top5_attributes= top5_attributes, top5_recall_attributes = top5_recall_attributes))
                    file_results.write('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                    'Recall@1 Attributes {top1_recall_attributes.val:.3f} ({top1_recall_attributes.avg:.3f})\t'
                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                    'Recall@3 Attributes {top3_recall_attributes.val:.3f} ({top3_recall_attributes.avg:.3f})\t'
                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                    'Recall@5 Attributes {top5_recall_attributes.val:.3f} ({top5_recall_attributes.avg:.3f})\t'
                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                        i, len(val_loader),  batch_time=batch_time,
                    data_time=batch_time, loss_category =losses_category, loss_attributes = losses_attributes,
                    top1_category =top1_category,top1_attributes = top1_attributes,top1_recall_attributes = top1_recall_attributes,
                    top3_category =top3_category,top3_attributes = top3_attributes,top3_recall_attributes = top3_recall_attributes,
                    top5_category=top5_category,top5_attributes= top5_attributes, top5_recall_attributes = top5_recall_attributes,))
                    file_results.write("\n")

            return top1_category.avg,  top1_attributes.avg, top1_recall_attributes.avg

    def save_checkpoint(self,state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '/output/model_best.pth.tar')

    def adjust_learning_rate(self,optimizer, epoch,lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.1 ** (epoch //30)) #//30
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr


# Helper class
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count