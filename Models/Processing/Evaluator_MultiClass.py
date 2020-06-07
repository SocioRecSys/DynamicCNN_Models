import time
import torch
import random
from torch.autograd import Variable
import shutil

###### Multi class scenarios -- Categories and Attributes and subcategories
###### Was applied on Instagram Dataset
###### Contains methods for training the model, accuracy, saving the model, etc.
class Evaluator_MultiClass(object):

    def train(self,train_loader, model, criterion, optimizer, epoch,print_freq):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_category = AverageMeter()
        losses_subcategory = AverageMeter()
        losses_attributes= AverageMeter()

        top1_category = AverageMeter()
        top3_category = AverageMeter()
        top5_category = AverageMeter()

        top1_subcategory = AverageMeter()
        top3_subcategory = AverageMeter()
        top5_subcategory = AverageMeter()

        top1_attributes = AverageMeter()
        top3_attributes = AverageMeter()
        top5_attributes = AverageMeter()

        top1_recall_category = AverageMeter()
        top3_recall_category = AverageMeter()
        top5_recall_category = AverageMeter()

        top1_recall_subcategory = AverageMeter()
        top3_recall_subcategory = AverageMeter()
        top5_recall_subcategory = AverageMeter()

        top1_recall_attributes = AverageMeter()
        top3_recall_attributes = AverageMeter()
        top5_recall_attributes = AverageMeter()

        top5_precision_blouses = AverageMeter()
        top5_precision_coats = AverageMeter()
        top5_precision_dresses = AverageMeter()
        top5_precision_jeans = AverageMeter()
        top5_precision_jackets = AverageMeter()
        top5_precision_jumpers = AverageMeter()
        top5_precision_skirts = AverageMeter()
        top5_precision_tights = AverageMeter()
        top5_precision_tops = AverageMeter()
        top5_precision_trousers = AverageMeter()
        top5_precision_shoes = AverageMeter()
        top5_precision_bags = AverageMeter()
        top5_precision_accessories = AverageMeter()
        top5_recall_blouses = AverageMeter()
        top5_recall_coats = AverageMeter()
        top5_recall_dresses = AverageMeter()
        top5_recall_jeans = AverageMeter()
        top5_recall_jackets = AverageMeter()
        top5_recall_jumpers = AverageMeter()
        top5_recall_skirts = AverageMeter()
        top5_recall_tights = AverageMeter()
        top5_recall_tops = AverageMeter()
        top5_recall_trousers = AverageMeter()
        top5_recall_shoes = AverageMeter()
        top5_recall_bags = AverageMeter()
        top5_recall_accessories = AverageMeter()

        top5_precisionS_casual = AverageMeter()
        top5_recallS_casual = AverageMeter()
        top5_precisionS_watch = AverageMeter()
        top5_recallS_watch = AverageMeter()
        top5_precisionS_sandal = AverageMeter()
        top5_recallS_sandal = AverageMeter()
        top5_precisionS_bags = AverageMeter()
        top5_recallS_bags = AverageMeter()
        top5_precisionS_tops = AverageMeter()
        top5_recallS_tops = AverageMeter()
        top5_precisionS_shoes = AverageMeter()
        top5_recallS_shoes = AverageMeter()
        top5_precisionS_shorts = AverageMeter()
        top5_recallS_shorts = AverageMeter()
        top5_precisionS_legging = AverageMeter()
        top5_recallS_legging = AverageMeter()
        top5_precisionS_jumpsuit = AverageMeter()
        top5_recallS_jumpsuit = AverageMeter()
        top5_precisionS_skirts = AverageMeter()
        top5_recallS_skirts = AverageMeter()
        top5_precisionS_sweater = AverageMeter()
        top5_recallS_sweater = AverageMeter()


        top5_precisionA_leather = AverageMeter()
        top5_recallA_leather = AverageMeter()
        top5_precisionA_lace = AverageMeter()
        top5_recallA_lace = AverageMeter()
        top5_precisionA_checked = AverageMeter()
        top5_recallA_checked = AverageMeter()
        top5_precisionA_printed = AverageMeter()
        top5_recallA_printed = AverageMeter()
        top5_precisionA_floral = AverageMeter()
        top5_recallA_floral = AverageMeter()
        top5_precisionA_striped = AverageMeter()
        top5_recallA_striped = AverageMeter()
        top5_precisionA_ployester = AverageMeter()
        top5_recallA_ployester = AverageMeter()
        top5_precisionA_colorful = AverageMeter()
        top5_recallA_colorful = AverageMeter()
        top5_precisionA_denim = AverageMeter()
        top5_recallA_denim= AverageMeter()
        top5_precisionA_herringbone = AverageMeter()
        top5_recallA_herringbone = AverageMeter()



        # switch to train mode
        model.train()

        end = time.time()

        # image of size 1 * 3 * 224 * 224
        for i, (image, category,subcategory, attributes, lstCategories,lstSubCategories, lstAttributeIndices) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            input_var = torch.autograd.Variable(image)
            # Float Tensor of size 1 * 13 with value 1 for the correct category and 0 otherwise
            target_category_var = torch.autograd.Variable(category)
            # Float Tensor of size 1 * 124 with value 1 for the correct subcategory and 0 otherwise
            target_subcategory_var = torch.autograd.Variable(subcategory)
            # Float Tensor of size 1 * 128 with value 1 for the correct attribute and 0 otherwise
            target_attributes_var = torch.autograd.Variable(attributes)

            target_category_var = target_category_var.cuda(async=True)
            target_subcategory_var = target_subcategory_var.cuda(async=True)
            target_attributes_var = target_attributes_var.cuda(async=True)

            # Randomly send supporting category or supporting attribute
            categoryItemValue = random.choice(lstCategories)
            subcategoryItemValue = random.choice(lstSubCategories)
            attributeItemValue = random.sample(lstAttributeIndices,2)

            # compute output
            # send the supporting item type or value
            lstOutputs, lstSubCategoriesOutput = model(input_var, categoryItemValue, subcategoryItemValue, attributeItemValue)

            output_category =  Variable(lstOutputs[0])
             #output_subcategory =  Variable(lstOutputs[1])
            output_attributes = Variable(lstOutputs[1])

            output_category = output_category.cuda(async=True)
            #output_subcategory = output_subcategory.cuda(async=True)
            output_attributes = output_attributes.cuda(async=True)

            loss_category = criterion(output_category, target_category_var)
            lstLoss = []
            for lo in range(0,len(lstSubCategoriesOutput)):
                lstLoss = ( criterion(Variable(lstSubCategoriesOutput[lo]).cuda(async=True), target_subcategory_var))
            loss_attributes = criterion(output_attributes, target_attributes_var.squeeze(1))

            # measure accuracy and record loss
            prec1_category, prec3_category, prec5_category, recall1_category, recall3_category, recall5_category = accuracy_multiLabel(output_category.data, target_category_var,lstCategories, topk=(1,3, 5))

            prec1_subcategory, prec3_subcategory, prec5_subcategory, recall1_subcategory, recall3_subcategory, recall5_subcategory = accuracy_multiLabel_subcategories(lstSubCategoriesOutput, target_subcategory_var, lstSubCategories, topk=(1,3, 5))

            prec1_attributes, prec3_attributes, prec5_attributes, recall1_atributes, recall3_attributes, recall5_attributes = accuracy_multiLabel(output_attributes.data, target_attributes_var, lstAttributeIndices, topk=(1, 3,5))

            #prec5_blouses,recall5_blouses,prec5_coats,recall5_coats,prec5_dresses, recall5_dresses,prec5_jeans, recall5_jeans, prec5_jackets, recall5_jackets, prec5_jumpers, recall5_jumpers,prec5_skirts, \
            #recall5_skirts,prec5_tights, recall5_tights, prec5_tops, recall5_tops, prec5_trousers,recall5_trousers,prec5_shoes, recall5_shoes, prec5_bags, recall5_bags, prec5_acc, recall5_acc= accuracy_perCategory(output_category.data, target_category_var,lstCategories, topk=(5))

            #casual_precision, casual_recall, watch_precision, watch_recall, sandal_precision, sandal_recall, bags_precision, bags_recall, \
            #tops_precision, tops_recall, shoes_precision, shoes_recall, shorts_precision, shorts_recall, legging_precision, legging_recall, jumpsuit_precision, jumpsuit_recall, skirts_precision, skirts_recall, sweater_precision, sweater_recall = accuracy_perSubCategory(lstSubCategoriesOutput, target_subcategory_var, lstSubCategories,  topk=(5))

            #leather_precision, leather_recall,lace_precision, lace_recall, checked_precision, checked_recall ,printed_precision, \
            #printed_recall,floral_precision, floral_recall,striped_precision, striped_recall,ployester_precision, ployester_recall,colorful_precision, colorful_recall,denim_precision, denim_recall,herringbone_precision, herringbone_recall = accuracy_perAttributes(output_attributes.data, target_attributes_var, lstAttributeIndices,  topk=(5))

            losses_category.update(loss_category.data[0], input_var.size(0))
            losses_subcategory.update(lstLoss.data[0], input_var.size(0))
            losses_attributes.update(loss_attributes.data[0], input_var.size(0))

            top1_category.update(prec1_category, input_var.size(0))
            top1_subcategory.update(prec1_subcategory, input_var.size(0))
            top1_attributes.update(prec1_attributes, input_var.size(0))

            top3_category.update(prec3_category, input_var.size(0))
            top3_subcategory.update(prec3_subcategory, input_var.size(0))
            top3_attributes.update(prec3_attributes, input_var.size(0))

            top5_category.update(prec5_category, input_var.size(0))
            top5_subcategory.update(prec5_subcategory, input_var.size(0))
            top5_attributes.update(prec5_attributes, input_var.size(0))

            top1_recall_category.update(recall1_category, input_var.size(0))
            top3_recall_category.update(recall3_category, input_var.size(0))
            top5_recall_category.update(recall5_category, input_var.size(0))

            top1_recall_subcategory.update(recall1_subcategory, input_var.size(0))
            top3_recall_subcategory.update(recall3_subcategory, input_var.size(0))
            top5_recall_subcategory.update(recall5_subcategory, input_var.size(0))

            top1_recall_attributes.update(recall1_atributes, input_var.size(0))
            top3_recall_attributes.update(recall3_attributes, input_var.size(0))
            top5_recall_attributes.update(recall5_attributes, input_var.size(0))

            '''
            top5_precision_blouses.update(prec5_blouses, input_var.size(0))
            top5_precision_coats.update(prec5_coats, input_var.size(0))
            top5_precision_dresses.update(prec5_dresses, input_var.size(0))
            top5_precision_jeans.update(prec5_jeans, input_var.size(0))
            top5_precision_jackets.update(prec5_jackets, input_var.size(0))
            top5_precision_jumpers.update(prec5_jumpers, input_var.size(0))
            top5_precision_skirts.update(prec5_skirts, input_var.size(0))
            top5_precision_tights.update(prec5_tights, input_var.size(0))
            top5_precision_tops.update(prec5_tops, input_var.size(0))
            top5_precision_trousers.update(prec5_trousers, input_var.size(0))
            top5_precision_shoes.update(prec5_shoes, input_var.size(0))
            top5_precision_bags.update(prec5_bags, input_var.size(0))
            top5_precision_accessories.update(prec5_acc, input_var.size(0))


            top5_recall_blouses.update(recall5_blouses, input_var.size(0))
            top5_recall_coats.update(recall5_coats, input_var.size(0))
            top5_recall_dresses.update(recall5_dresses, input_var.size(0))
            top5_recall_jeans.update(recall5_jeans, input_var.size(0))
            top5_recall_jackets.update(recall5_jackets, input_var.size(0))
            top5_recall_jumpers.update(recall5_jumpers, input_var.size(0))
            top5_recall_skirts.update(recall5_skirts, input_var.size(0))
            top5_recall_tights.update(recall5_tights, input_var.size(0))
            top5_recall_tops.update(recall5_tops, input_var.size(0))
            top5_recall_trousers.update(recall5_trousers, input_var.size(0))
            top5_recall_shoes.update(recall5_shoes, input_var.size(0))
            top5_recall_bags.update(recall5_bags, input_var.size(0))
            top5_recall_accessories.update(recall5_acc, input_var.size(0))

            top5_precisionS_bags.update(bags_precision,input_var.size(0) )
            top5_precisionS_casual.update(casual_precision,input_var.size(0) )
            top5_precisionS_jumpsuit.update(jumpsuit_precision,input_var.size(0) )
            top5_precisionS_legging.update(legging_precision,input_var.size(0) )
            top5_precisionS_sandal.update(sandal_precision,input_var.size(0) )
            top5_precisionS_shoes.update(shoes_precision,input_var.size(0) )
            top5_precisionS_shorts.update(shorts_precision,input_var.size(0) )
            top5_precisionS_skirts.update(skirts_precision,input_var.size(0) )
            top5_precisionS_sweater.update(sweater_precision,input_var.size(0) )
            top5_precisionS_tops.update(tops_precision,input_var.size(0) )
            top5_precisionS_watch.update(watch_precision,input_var.size(0) )

            top5_recallS_bags.update(bags_recall,input_var.size(0) )
            top5_recallS_casual.update(casual_recall,input_var.size(0) )
            top5_recallS_jumpsuit.update(jumpsuit_recall,input_var.size(0) )
            top5_recallS_legging.update(legging_recall,input_var.size(0) )
            top5_recallS_sandal.update(sandal_recall,input_var.size(0) )
            top5_recallS_shoes.update(shoes_recall,input_var.size(0) )
            top5_recallS_shorts.update(shorts_recall,input_var.size(0) )
            top5_recallS_skirts.update(skirts_recall,input_var.size(0) )
            top5_recallS_sweater.update(sweater_recall,input_var.size(0) )
            top5_recallS_tops.update(tops_recall,input_var.size(0) )
            top5_recallS_watch.update(watch_recall,input_var.size(0) )

            top5_precisionA_checked.update(checked_precision,input_var.size(0) )
            top5_precisionA_colorful.update(colorful_precision,input_var.size(0) )
            top5_precisionA_denim.update(denim_precision,input_var.size(0) )
            top5_precisionA_floral.update(floral_precision,input_var.size(0) )
            top5_precisionA_herringbone.update(herringbone_precision,input_var.size(0) )
            top5_precisionA_lace.update(lace_precision,input_var.size(0) )
            top5_precisionA_leather.update(leather_precision,input_var.size(0) )
            top5_precisionA_ployester.update(ployester_precision,input_var.size(0) )
            top5_precisionA_printed.update(printed_precision,input_var.size(0) )
            top5_precisionA_striped.update(striped_precision,input_var.size(0) )

            top5_recallA_checked.update(checked_recall,input_var.size(0) )
            top5_recallA_colorful.update(colorful_recall,input_var.size(0) )
            top5_recallA_denim.update(denim_recall,input_var.size(0) )
            top5_recallA_floral.update(floral_recall,input_var.size(0) )
            top5_recallA_herringbone.update(herringbone_recall,input_var.size(0) )
            top5_recallA_lace.update(lace_recall,input_var.size(0) )
            top5_recallA_leather.update(leather_recall,input_var.size(0) )
            top5_recallA_ployester.update(ployester_recall,input_var.size(0) )
            top5_recallA_printed.update(printed_recall,input_var.size(0) )
            top5_recallA_striped.update(striped_recall,input_var.size(0) )

            '''''
            # compute gradient and do SGD step
            optimizer.zero_grad()

            var_loss = Variable(loss_category.data, requires_grad=True)
            var_loss.backward(retain_graph=True)

            var_loss_subcategory = Variable(lstLoss.data, requires_grad=True)
            var_loss_subcategory.backward(retain_graph=True)

            var_loss_attr = Variable(loss_attributes.data, requires_grad =True)
            var_loss_attr.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                with open("/output/results_withtext_singleItem_training", "a") as file_results:
                    with open("/output/results_perCategory_training", "a") as file_perCategory_results:
                        with open("/output/results_perSubCategory_training", "a") as file_perSubCategory_results:
                            with open("/output/results_perAttributes_training", "a") as file_perAttributes_results:

                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                                    'Loss_subcategory {loss_subcategory.val:.4f} ({loss_subcategory.avg:.4f})\t'
                                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                                    'Prec@1 Sub Category {top1_subcategory.val:.3f} ({top1_subcategory.avg:.3f})\t'
                                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                                    'Prec@3 Sub Category {top3_subcategory.val:.3f} ({top3_subcategory.avg:.3f})\t'
                                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                                    'Prec@5 Sub Category {top5_subcategory.val:.3f} ({top5_subcategory.avg:.3f})\t'
                                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                                    epoch, i, len(train_loader), batch_time=batch_time,
                                    data_time=data_time, loss_category =losses_category, loss_subcategory = losses_subcategory, loss_attributes = losses_attributes,
                                    top1_category =top1_category, top1_subcategory =top1_subcategory,top1_attributes = top1_attributes,
                                    top3_category =top3_category, top3_subcategory =top3_subcategory,top3_attributes = top3_attributes,
                                    top5_category=top5_category, top5_subcategory=top5_subcategory, top5_attributes=top5_attributes
                                    ))

                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                                    'Loss_subcategory {loss_subcategory.val:.4f} ({loss_subcategory.avg:.4f})\t'
                                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                                    'Recall@1 Category {top1_recall_category.val:.3f} ({top1_recall_category.avg:.3f})\t'
                                    'Recall@3 Category {top3_recall_category.val:.3f} ({top3_recall_category.avg:.3f})\t'
                                    'Recall@5 Category {top5_recall_category.val:.3f} ({top5_recall_category.avg:.3f})\t'
                                    'Recall@1 Sub-Category {top1_recall_subcategory.val:.3f} ({top1_recall_subcategory.avg:.3f})\t'
                                    'Recall@3 Sub-Category {top3_recall_subcategory.val:.3f} ({top3_recall_subcategory.avg:.3f})\t'
                                    'Recall@4 Sub-Category {top5_recall_subcategory.val:.3f} ({top5_recall_subcategory.avg:.3f})\t'
                                    'Recall@1 Attributes {top1_recall_attributes.val:.3f} ({top1_recall_attributes.avg:.3f})\t'
                                    'Recall@3 Attributes {top3_recall_attributes.val:.3f} ({top3_recall_attributes.avg:.3f})\t'
                                    'Recall@5 Attributes {top5_recall_attributes.val:.3f} ({top5_recall_attributes.avg:.3f})'.format(
                                    epoch, i, len(train_loader), batch_time=batch_time,
                                    data_time=data_time, loss_category =losses_category, loss_subcategory = losses_subcategory, loss_attributes = losses_attributes,
                                    top1_recall_category=top1_recall_category, top3_recall_category = top3_recall_category, top5_recall_category = top5_recall_category,
                                    top1_recall_subcategory = top1_recall_subcategory, top3_recall_subcategory = top3_recall_subcategory, top5_recall_subcategory=top5_recall_subcategory,
                                    top1_recall_attributes=top1_recall_attributes,top3_recall_attributes=top3_recall_attributes,top5_recall_attributes=top5_recall_attributes
                                    ))
                                print("\n")
                                print(".........................................................................")
                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Prec@5 Blouses {top5_precision_blouses.val:.3f} ({top5_precision_blouses.avg:.3f})\t'
                                    'Prec@5 Coats {top5_precision_coats.val:.3f} ({top5_precision_coats.avg:.3f})\t'
                                    'Prec@5 Dresses {top5_precision_dresses.val:.3f} ({top5_precision_dresses.avg:.3f})\t'
                                    'Prec@5 Jeans {top5_precision_jeans.val:.3f} ({top5_precision_jeans.avg:.3f})\t'
                                    'Prec@5 Jackets {top5_precision_jackets.val:.3f} ({top5_precision_jackets.avg:.3f})\t'
                                    'Prec@5 Jumpers {top5_precision_jumpers.val:.3f} ({top5_precision_jumpers.avg:.3f})\t'
                                    'Prec@5 Skirts {top5_precision_skirts.val:.3f} ({top5_precision_skirts.avg:.3f})\t'
                                    'Prec@5 Tights {top5_precision_tights.val:.3f} ({top5_precision_tights.avg:.3f})\t'
                                    'Prec@5 Tops {top5_precision_tops.val:.3f} ({top5_precision_tops.avg:.3f})\t'
                                    'Prec@5 Trousers {top5_precision_trousers.val:.3f} ({top5_precision_trousers.avg:.3f})\t'
                                    'Prec@5 Shoes {top5_precision_shoes.val:.3f} ({top5_precision_shoes.avg:.3f})\t'
                                    'Prec@5 Bags {top5_precision_bags.val:.3f} ({top5_precision_bags.avg:.3f})\t'
                                    'Prec@5 Accessories {top5_precision_accessories.val:.3f} ({top5_precision_accessories.avg:.3f})'.format(
                                     epoch, i, len(train_loader), batch_time=batch_time, top5_precision_blouses = top5_precision_blouses,
                                    top5_precision_coats = top5_precision_coats, top5_precision_dresses = top5_precision_dresses, top5_precision_jeans = top5_precision_jeans,
                                    top5_precision_jackets = top5_precision_jackets,  top5_precision_jumpers = top5_precision_jumpers,
                                    top5_precision_skirts = top5_precision_skirts, top5_precision_tights = top5_precision_tights, top5_precision_tops = top5_precision_tops,
                                    top5_precision_trousers = top5_precision_trousers, top5_precision_shoes = top5_precision_shoes, top5_precision_bags = top5_precision_bags,
                                    top5_precision_accessories = top5_precision_accessories
                                     ))
                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Recall@5 Blouses {top5_recall_blouses.val:.3f} ({top5_recall_blouses.avg:.3f})\t'
                                    'Recall@5 Coats {top5_recall_coats.val:.3f} ({top5_recall_coats.avg:.3f})\t'
                                    'Recall@5 Dresses {top5_recall_dresses.val:.3f} ({top5_recall_dresses.avg:.3f})\t'
                                    'Recall@5 Jeans {top5_recall_jeans.val:.3f} ({top5_recall_jeans.avg:.3f})\t'
                                    'Recall@5 Jackets {top5_recall_jackets.val:.3f} ({top5_recall_jackets.avg:.3f})\t'
                                    'Recall@5 Jumpers {top5_recall_jumpers.val:.3f} ({top5_recall_jumpers.avg:.3f})\t'
                                    'Recall@5 Skirts {top5_recall_skirts.val:.3f} ({top5_recall_skirts.avg:.3f})\t'
                                    'Recall@5 Tights {top5_recall_tights.val:.3f} ({top5_recall_tights.avg:.3f})\t'
                                    'Recall@5 Tops {top5_recall_tops.val:.3f} ({top5_recall_tops.avg:.3f})\t'
                                    'Recall@5 Trousers {top5_recall_trousers.val:.3f} ({top5_recall_trousers.avg:.3f})\t'
                                    'Recall@5 Shoes {top5_recall_shoes.val:.3f} ({top5_recall_shoes.avg:.3f})\t'
                                    'Recall@5 Bags {top5_recall_bags.val:.3f} ({top5_recall_bags.avg:.3f})\t'
                                    'Recall@5 Accessories {top5_recall_accessories.val:.3f} ({top5_recall_accessories.avg:.3f})'.format(
                                    epoch, i, len(train_loader), batch_time=batch_time, top5_recall_blouses = top5_recall_blouses,
                                     top5_recall_coats = top5_recall_coats, top5_recall_dresses = top5_recall_dresses, top5_recall_jeans = top5_recall_jeans,
                                    top5_recall_jackets = top5_recall_jackets,  top5_recall_jumpers = top5_recall_jumpers,
                                    top5_recall_skirts = top5_recall_skirts, top5_recall_tights = top5_recall_tights, top5_recall_tops = top5_recall_tops,
                                    top5_recall_trousers = top5_recall_trousers, top5_recall_shoes = top5_recall_shoes, top5_recall_bags = top5_recall_bags,
                                    top5_recall_accessories = top5_recall_accessories
                                    ))
                                print("\n")
                                #casual, watch, sandal, bags, topsides, short, legging, jumpsuit, skirts, sweater
                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Prec@5 Casual {top5_precisionS_casual.val:.3f} ({top5_precisionS_casual.avg:.3f})\t'
                                    'Prec@5 Watch {top5_precisionS_watch.val:.3f} ({top5_precisionS_watch.avg:.3f})\t'
                                    'Prec@5 Sandal {top5_precisionS_sandal.val:.3f} ({top5_precisionS_sandal.avg:.3f})\t'
                                    'Prec@5 Bagas {top5_precisionS_bags.val:.3f} ({top5_precisionS_bags.avg:.3f})\t'
                                    'Prec@5 Tops {top5_precisionS_tops.val:.3f} ({top5_precisionS_tops.avg:.3f})\t'
                                    'Prec@5 Shorts {top5_precisionS_shorts.val:.3f} ({top5_precisionS_shorts.avg:.3f})\t'
                                    'Prec@5 Leggings {top5_precisionS_legging.val:.3f} ({top5_precisionS_legging.avg:.3f})\t'
                                    'Prec@5 Jumpsuits {top5_precisionS_jumpsuit.val:.3f} ({top5_precisionS_jumpsuit.avg:.3f})\t'
                                    'Prec@5 Skirts {top5_precisionS_skirts.val:.3f} ({top5_precisionS_skirts.avg:.3f})\t'
                                    'Prec@5 Sweaters {top5_precisionS_sweater.val:.3f} ({top5_precisionS_sweater.avg:.3f})'.format(
                                     epoch, i, len(train_loader), batch_time=batch_time, top5_precisionS_casual = top5_precisionS_casual,
                                    top5_precisionS_watch = top5_precisionS_watch, top5_precisionS_sandal = top5_precisionS_sandal, top5_precisionS_bags = top5_precisionS_bags,
                                    top5_precisionS_tops = top5_precisionS_tops,  top5_precisionS_shorts = top5_precisionS_shorts,
                                    top5_precisionS_legging = top5_precisionS_legging, top5_precisionS_jumpsuit = top5_precisionS_jumpsuit, top5_precisionS_skirts = top5_precisionS_skirts,
                                    top5_precisionS_sweater = top5_precisionS_sweater
                                     ))
                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Recall@5 Casual {top5_recallS_casual.val:.3f} ({top5_recallS_casual.avg:.3f})\t'
                                    'Recall@5 Watch {top5_recallS_watch.val:.3f} ({top5_recallS_watch.avg:.3f})\t'
                                    'Recall@5 Sandal {top5_recallS_sandal.val:.3f} ({top5_recallS_sandal.avg:.3f})\t'
                                    'Recall@5 Bagas {top5_recallS_bags.val:.3f} ({top5_recallS_bags.avg:.3f})\t'
                                    'Recall@5 Tops {top5_recallS_tops.val:.3f} ({top5_recallS_tops.avg:.3f})\t'
                                    'Recall@5 Shorts {top5_recallS_shorts.val:.3f} ({top5_recallS_shorts.avg:.3f})\t'
                                    'Recall@5 Leggings {top5_recallS_legging.val:.3f} ({top5_recallS_legging.avg:.3f})\t'
                                    'Recall@5 Jumpsuits {top5_recallS_jumpsuits.val:.3f} ({top5_recallS_jumpsuits.avg:.3f})\t'
                                    'Recall@5 Skirts {top5_recallS_skirts.val:.3f} ({top5_recallS_skirts.avg:.3f})\t'
                                    'Recall@5 Sweaters {top5_recallS_sweater.val:.3f} ({top5_recallS_sweater.avg:.3f})'.format(
                                     epoch, i, len(train_loader), batch_time=batch_time, top5_recallS_casual = top5_recallS_casual,
                                    top5_recallS_watch = top5_recallS_watch, top5_recallS_sandal = top5_recallS_sandal, top5_recallS_bags = top5_recallS_bags,
                                    top5_recallS_tops = top5_recallS_tops,  top5_recallS_shorts = top5_recallS_shorts,
                                    top5_recallS_legging = top5_recallS_legging, top5_recallS_jumpsuits = top5_recallS_jumpsuit, top5_recallS_skirts = top5_recallS_skirts,
                                    top5_recallS_sweater = top5_recallS_sweater
                                     ))
                                print("\n")
                                #leather, lace, checked,print, floral,striped,polyester,colourful,denim,herringbone
                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Prec@5 leather {top5_precisionA_leather.val:.3f} ({top5_precisionA_leather.avg:.3f})\t'
                                    'Prec@5 lace {top5_precisionA_lace.val:.3f} ({top5_precisionA_lace.avg:.3f})\t'
                                    'Prec@5 checked {top5_precisionA_checked.val:.3f} ({top5_precisionA_checked.avg:.3f})\t'
                                    'Prec@5 print {top5_precisionA_print.val:.3f} ({top5_precisionA_print.avg:.3f})\t'
                                    'Prec@5 floral {top5_precisionA_floral.val:.3f} ({top5_precisionA_floral.avg:.3f})\t'
                                    'Prec@5 striped {top5_precisionA_striped.val:.3f} ({top5_precisionA_striped.avg:.3f})\t'
                                    'Prec@5 polyester {top5_precisionA_polyester.val:.3f} ({top5_precisionA_polyester.avg:.3f})\t'
                                    'Prec@5 colourful {top5_precisionA_colourful.val:.3f} ({top5_precisionA_colourful.avg:.3f})\t'
                                    'Prec@5 denim {top5_precisionA_denim.val:.3f} ({top5_precisionA_denim.avg:.3f})\t'
                                    'Prec@5 herringbone {top5_precisionA_herringbone.val:.3f} ({top5_precisionA_herringbone.avg:.3f})'.format(
                                     epoch, i, len(train_loader), batch_time=batch_time, top5_precisionA_leather = top5_precisionA_leather,
                                    top5_precisionA_lace = top5_precisionA_lace, top5_precisionA_checked = top5_precisionA_checked, top5_precisionA_print = top5_precisionA_printed,
                                top5_precisionA_floral= top5_precisionA_floral,top5_precisionA_striped=top5_precisionA_striped,top5_precisionA_polyester=top5_precisionA_ployester,
                                top5_precisionA_colourful=top5_precisionA_colorful,top5_precisionA_denim=top5_precisionA_denim,top5_precisionA_herringbone=top5_precisionA_herringbone
                                     ))
                                print('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Recall@5 leather {top5_recallA_leather.val:.3f} ({top5_recallA_leather.avg:.3f})\t'
                                    'Recall@5 lace {top5_recallA_lace.val:.3f} ({top5_recallA_lace.avg:.3f})\t'
                                    'Recall@5 checked {top5_recallA_checked.val:.3f} ({top5_recallA_checked.avg:.3f})\t'
                                    'Recall@5 print {top5_recallA_print.val:.3f} ({top5_recallA_print.avg:.3f})\t'
                                    'Recall@5 floral {top5_recallA_floral.val:.3f} ({top5_recallA_floral.avg:.3f})\t'
                                    'Recall@5 striped {top5_recallA_striped.val:.3f} ({top5_recallA_striped.avg:.3f})\t'
                                    'Recall@5 polyester {top5_recallA_polyester.val:.3f} ({top5_recallA_polyester.avg:.3f})\t'
                                    'Recall@5 colourful {top5_recallA_colourful.val:.3f} ({top5_recallA_colourful.avg:.3f})\t'
                                    'Recall@5 denim {top5_recallA_denim.val:.3f} ({top5_recallA_denim.avg:.3f})\t'
                                    'Recall@5 herringbone {top5_recallA_herringbone.val:.3f} ({top5_recallA_herringbone.avg:.3f})'.format(
                                     epoch, i, len(train_loader), batch_time=batch_time, top5_recallA_leather = top5_recallA_leather,
                                    top5_recallA_lace = top5_recallA_lace, top5_recallA_checked = top5_recallA_checked, top5_recallA_print = top5_recallA_printed,
                                top5_recallA_floral= top5_recallA_floral,top5_recallA_striped=top5_recallA_striped,top5_recallA_polyester=top5_recallA_ployester,
                                top5_recallA_colourful=top5_recallA_colorful,top5_recallA_denim=top5_recallA_denim,top5_recallA_herringbone=top5_recallA_herringbone
                                     ))
                                '''
                                file_results.write('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                                    'Loss_subcategory {loss_subcategory.val:.4f} ({loss_subcategory.avg:.4f})\t'
                                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                                    'Prec@1 Sub Category {top1_subcategory.val:.3f} ({top1_subcategory.avg:.3f})\t'
                                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                                    'Prec@3 Sub Category {top3_subcategory.val:.3f} ({top3_subcategory.avg:.3f})\t'
                                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                                    'Prec@5 Sub Category {top5_subcategory.val:.3f} ({top5_subcategory.avg:.3f})\t'
                                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                                    epoch, i, len(train_loader), batch_time=batch_time,
                                    data_time=data_time, loss_category =losses_category,  loss_subcategory = losses_subcategory,loss_attributes = losses_attributes,
                                    top1_category =top1_category, top1_subcategory =top1_subcategory,top1_attributes = top1_attributes,
                                    top3_category =top3_category,top3_subcategory =top3_subcategory,top3_attributes = top3_attributes,
                                    top5_category=top5_category,top5_subcategory=top5_subcategory,top5_attributes= top5_attributes
                                ))
                                '''''

    def validate(self,val_loader, model, criterion, print_freq):
        batch_time = AverageMeter()

        losses_category= AverageMeter()
        losses_subcategory = AverageMeter()
        losses_attributes = AverageMeter()

        top1_category= AverageMeter()
        top3_category= AverageMeter()
        top5_category = AverageMeter()

        top1_subcategory= AverageMeter()
        top3_subcategory= AverageMeter()
        top5_subcategory = AverageMeter()

        top1_attributes = AverageMeter()
        top3_attributes = AverageMeter()
        top5_attributes = AverageMeter()

        top1_recall_category = AverageMeter()
        top3_recall_category = AverageMeter()
        top5_recall_category = AverageMeter()

        top1_recall_subcategory = AverageMeter()
        top3_recall_subcategory = AverageMeter()
        top5_recall_subcategory = AverageMeter()

        top1_recall_attributes = AverageMeter()
        top3_recall_attributes = AverageMeter()
        top5_recall_attributes = AverageMeter()

        top5_precision_blouses = AverageMeter()
        top5_precision_coats = AverageMeter()
        top5_precision_dresses = AverageMeter()
        top5_precision_jeans = AverageMeter()
        top5_precision_jackets = AverageMeter()
        top5_precision_jumpers = AverageMeter()
        top5_precision_skirts = AverageMeter()
        top5_precision_tights = AverageMeter()
        top5_precision_tops = AverageMeter()
        top5_precision_trousers = AverageMeter()
        top5_precision_shoes = AverageMeter()
        top5_precision_bags = AverageMeter()
        top5_precision_accessories = AverageMeter()
        top5_recall_blouses = AverageMeter()
        top5_recall_coats = AverageMeter()
        top5_recall_dresses = AverageMeter()
        top5_recall_jeans = AverageMeter()
        top5_recall_jackets = AverageMeter()
        top5_recall_jumpers = AverageMeter()
        top5_recall_skirts = AverageMeter()
        top5_recall_tights = AverageMeter()
        top5_recall_tops = AverageMeter()
        top5_recall_trousers = AverageMeter()
        top5_recall_shoes = AverageMeter()
        top5_recall_bags = AverageMeter()
        top5_recall_accessories = AverageMeter()

        top5_precisionS_casual = AverageMeter()
        top5_recallS_casual = AverageMeter()
        top5_precisionS_watch = AverageMeter()
        top5_recallS_watch = AverageMeter()
        top5_precisionS_sandal = AverageMeter()
        top5_recallS_sandal = AverageMeter()
        top5_precisionS_bags = AverageMeter()
        top5_recallS_bags = AverageMeter()
        top5_precisionS_tops = AverageMeter()
        top5_recallS_tops = AverageMeter()
        top5_precisionS_shorts = AverageMeter()
        top5_recallS_shorts = AverageMeter()
        top5_precisionS_shoes = AverageMeter()
        top5_recallS_shoes = AverageMeter()
        top5_precisionS_legging = AverageMeter()
        top5_recallS_legging = AverageMeter()
        top5_precisionS_jumpsuit = AverageMeter()
        top5_recallS_jumpsuit = AverageMeter()
        top5_precisionS_skirts = AverageMeter()
        top5_recallS_skirts = AverageMeter()
        top5_precisionS_sweater = AverageMeter()
        top5_recallS_sweater = AverageMeter()


        top5_precisionA_leather = AverageMeter()
        top5_recallA_leather = AverageMeter()
        top5_precisionA_lace = AverageMeter()
        top5_recallA_lace = AverageMeter()
        top5_precisionA_checked = AverageMeter()
        top5_recallA_checked = AverageMeter()
        top5_precisionA_printed = AverageMeter()
        top5_recallA_printed = AverageMeter()
        top5_precisionA_floral = AverageMeter()
        top5_recallA_floral = AverageMeter()
        top5_precisionA_striped = AverageMeter()
        top5_recallA_striped = AverageMeter()
        top5_precisionA_ployester = AverageMeter()
        top5_recallA_ployester = AverageMeter()
        top5_precisionA_colorful = AverageMeter()
        top5_recallA_colorful = AverageMeter()
        top5_precisionA_denim = AverageMeter()
        top5_recallA_denim= AverageMeter()
        top5_precisionA_herringbone = AverageMeter()
        top5_recallA_herringbone = AverageMeter()


        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (image, category,subcategory, attributes, lstCategories,lstSubCategories, lstAttributeIndices) in enumerate(val_loader):        #target = target.cuda(async=True)
            input_var = torch.autograd.Variable(image, volatile=True)

            # Float Tensor of size 1 * 50 with value 1 for the correct category and 0 otherwise
            target_category_var = torch.autograd.Variable(category, volatile=True)
             # Float Tensor of size 1 * 124 with value 1 for the correct subcategory and 0 otherwise
            target_subcategory_var = torch.autograd.Variable(subcategory)
            # Float Tensor of size 1 * 1000 with value 1 for the correct attribute and 0 otherwise
            target_attributes_var = torch.autograd.Variable(attributes)

            target_category_var = target_category_var.cuda(async=True)
            target_subcategory_var = target_subcategory_var.cuda(async=True)
            target_attributes_var = target_attributes_var.cuda(async=True)

             # Randomly send supporting category or supporting attribute
            categoryItemValue = random.choice(lstCategories)
            subcategoryItemValue = random.choice(lstSubCategories)
            attributeItemValue = random.choice(lstAttributeIndices)

            # compute output
            # send the supporting item type or value
            lstOutputs, lstSubCategoriesOutput = model(input_var, categoryItemValue, subcategoryItemValue, attributeItemValue)

            output_category = Variable(lstOutputs[0])
            #output_subcategory = Variable(lstOutputs[1])
            output_attributes = Variable(lstOutputs[1])

            output_category = output_category.cuda(async=True)
            #output_subcategory = output_subcategory.cuda(async=True)
            output_attributes = output_attributes.cuda(async=True)

            loss_category = criterion(output_category, target_category_var)
            for lo in range(0,len(lstSubCategoriesOutput)):
                loss_subcategory= criterion(Variable(lstSubCategoriesOutput[lo]).cuda(async=True), target_subcategory_var)
            loss_attributes = criterion(output_attributes, target_attributes_var.squeeze(1))

           # measure accuracy and record loss
             # measure accuracy and record loss
            prec1_category, prec3_category, prec5_category, recall1_cateogry, recall3_category, recall5_category = accuracy_multiLabel(output_category.data, target_category_var, lstCategories, topk=(1,3, 5))
            prec1_subcategory, prec3_subcategory, prec5_subcategory, recall1_subcategory, recall3_subcategory, recall5_subcategory = accuracy_multiLabel_subcategories(lstSubCategoriesOutput, target_subcategory_var, lstSubCategories, topk=(1,3, 5))
            prec1_attributes, prec3_attributes, prec5_attributes, recall1_attributes, recall3_attributes, recall5_attributes  = accuracy_multiLabel(output_attributes.data, target_attributes_var,lstAttributeIndices, topk=(1,3, 5))

            #prec5_blouses,recall5_blouses,prec5_coats,recall5_coats,prec5_dresses, recall5_dresses,prec5_jeans, recall5_jeans, prec5_jackets, recall5_jackets, prec5_jumpers, recall5_jumpers,prec5_skirts, \
            #recall5_skirts,prec5_tights, recall5_tights, prec5_tops, recall5_tops, prec5_trousers,recall5_trousers,prec5_shoes, recall5_shoes, prec5_bags, recall5_bags, prec5_acc, recall5_acc= accuracy_perCategory(output_category.data, target_category_var,lstCategories, topk=(5))

            #casual_precision, casual_recall, watch_precision, watch_recall, sandal_precision, sandal_recall, bags_precision, bags_recall, \
            #tops_precision, tops_recall, shoes_precision, shoes_recall, shorts_precision, shorts_recall, legging_precision, legging_recall, jumpsuit_precision, jumpsuit_recall, skirts_precision, skirts_recall, sweater_precision, sweater_recall = accuracy_perSubCategory(lstSubCategoriesOutput, target_subcategory_var, lstSubCategories,  topk=(5))

            #leather_precision, leather_recall,lace_precision, lace_recall, checked_precision, checked_recall ,printed_precision, \
            #    printed_recall,floral_precision, floral_recall,striped_precision, striped_recall,ployester_precision, ployester_recall,colorful_precision, colorful_recall,denim_precision, denim_recall,herringbone_precision, herringbone_recall = accuracy_perAttributes(output_attributes.data, target_attributes_var, lstAttributeIndices, topk=(5))


            losses_category.update(loss_category.data[0], input_var.size(0))
            losses_subcategory.update(loss_subcategory.data[0], input_var.size(0))
            losses_attributes.update(loss_attributes.data[0], input_var.size(0))

            top1_category.update(prec1_category, input_var.size(0))
            top1_subcategory.update(prec1_subcategory, input_var.size(0))
            top1_attributes.update(prec1_attributes, input_var.size(0))

            top3_category.update(prec3_category, input_var.size(0))
            top3_subcategory.update(prec3_subcategory, input_var.size(0))
            top3_attributes.update(prec3_attributes, input_var.size(0))

            top5_category.update(prec5_category, input_var.size(0))
            top5_subcategory.update(prec5_subcategory, input_var.size(0))
            top5_attributes.update(prec5_attributes, input_var.size(0))

            top1_recall_category.update(recall1_cateogry, input_var.size(0))
            top3_recall_category.update(recall3_category, input_var.size(0))
            top5_recall_category.update(recall5_category, input_var.size(0))

            top1_recall_subcategory.update(recall1_subcategory, input_var.size(0))
            top3_recall_subcategory.update(recall3_subcategory, input_var.size(0))
            top5_recall_subcategory.update(recall5_subcategory, input_var.size(0))

            top1_recall_attributes.update(recall1_attributes, input_var.size(0))
            top3_recall_attributes.update(recall3_attributes, input_var.size(0))
            top5_recall_attributes.update(recall5_attributes, input_var.size(0))

            ''''
            top5_precision_blouses.update(prec5_blouses, input_var.size(0))
            top5_precision_coats.update(prec5_coats, input_var.size(0))
            top5_precision_dresses.update(prec5_dresses, input_var.size(0))
            top5_precision_jeans.update(prec5_jeans, input_var.size(0))
            top5_precision_jackets.update(prec5_jackets, input_var.size(0))
            top5_precision_jumpers.update(prec5_jumpers, input_var.size(0))
            top5_precision_skirts.update(prec5_skirts, input_var.size(0))
            top5_precision_tights.update(prec5_tights, input_var.size(0))
            top5_precision_tops.update(prec5_tops, input_var.size(0))
            top5_precision_trousers.update(prec5_trousers, input_var.size(0))
            top5_precision_shoes.update(prec5_shoes, input_var.size(0))
            top5_precision_bags.update(prec5_bags, input_var.size(0))
            top5_precision_accessories.update(prec5_acc, input_var.size(0))


            top5_recall_blouses.update(recall5_blouses, input_var.size(0))
            top5_recall_coats.update(recall5_coats, input_var.size(0))
            top5_recall_dresses.update(recall5_dresses, input_var.size(0))
            top5_recall_jeans.update(recall5_jeans, input_var.size(0))
            top5_recall_jackets.update(recall5_jackets, input_var.size(0))
            top5_recall_jumpers.update(recall5_jumpers, input_var.size(0))
            top5_recall_skirts.update(recall5_skirts, input_var.size(0))
            top5_recall_tights.update(recall5_tights, input_var.size(0))
            top5_recall_tops.update(recall5_tops, input_var.size(0))
            top5_recall_trousers.update(recall5_trousers, input_var.size(0))
            top5_recall_shoes.update(recall5_shoes, input_var.size(0))
            top5_recall_bags.update(recall5_bags, input_var.size(0))
            top5_recall_accessories.update(recall5_acc, input_var.size(0))

            top5_precisionS_bags.update(bags_precision,input_var.size(0) )
            top5_precisionS_casual.update(casual_precision,input_var.size(0) )
            top5_precisionS_jumpsuit.update(jumpsuit_precision,input_var.size(0) )
            top5_precisionS_legging.update(legging_precision,input_var.size(0) )
            top5_precisionS_sandal.update(sandal_precision,input_var.size(0) )
            top5_precisionS_shoes.update(shoes_precision,input_var.size(0) )
            top5_precisionS_shorts.update(shorts_precision,input_var.size(0) )
            top5_precisionS_skirts.update(skirts_precision,input_var.size(0) )
            top5_precisionS_sweater.update(sweater_precision,input_var.size(0) )
            top5_precisionS_tops.update(tops_precision,input_var.size(0) )
            top5_precisionS_watch.update(watch_precision,input_var.size(0) )

            top5_recallS_bags.update(bags_recall,input_var.size(0) )
            top5_recallS_casual.update(casual_recall,input_var.size(0) )
            top5_recallS_jumpsuit.update(jumpsuit_recall,input_var.size(0) )
            top5_recallS_legging.update(legging_recall,input_var.size(0) )
            top5_recallS_sandal.update(sandal_recall,input_var.size(0) )
            top5_recallS_shoes.update(shoes_recall,input_var.size(0) )
            top5_recallS_shorts.update(shorts_recall,input_var.size(0) )
            top5_recallS_skirts.update(skirts_recall,input_var.size(0) )
            top5_recallS_sweater.update(sweater_recall,input_var.size(0) )
            top5_recallS_tops.update(tops_recall,input_var.size(0) )
            top5_recallS_watch.update(watch_recall,input_var.size(0) )

            top5_precisionA_checked.update(checked_precision,input_var.size(0) )
            top5_precisionA_colorful.update(colorful_precision,input_var.size(0) )
            top5_precisionA_denim.update(denim_precision,input_var.size(0) )
            top5_precisionA_floral.update(floral_precision,input_var.size(0) )
            top5_precisionA_herringbone.update(herringbone_precision,input_var.size(0) )
            top5_precisionA_lace.update(lace_precision,input_var.size(0) )
            top5_precisionA_leather.update(leather_precision,input_var.size(0) )
            top5_precisionA_ployester.update(ployester_precision,input_var.size(0) )
            top5_precisionA_printed.update(printed_precision,input_var.size(0) )
            top5_precisionA_striped.update(striped_precision,input_var.size(0) )

            top5_recallA_checked.update(checked_recall,input_var.size(0) )
            top5_recallA_colorful.update(colorful_recall,input_var.size(0) )
            top5_recallA_denim.update(denim_recall,input_var.size(0) )
            top5_recallA_floral.update(floral_recall,input_var.size(0) )
            top5_recallA_herringbone.update(herringbone_recall,input_var.size(0) )
            top5_recallA_lace.update(lace_recall,input_var.size(0) )
            top5_recallA_leather.update(leather_recall,input_var.size(0) )
            top5_recallA_ployester.update(ployester_recall,input_var.size(0) )
            top5_recallA_printed.update(printed_recall,input_var.size(0) )
            top5_recallA_striped.update(striped_recall,input_var.size(0) )

            '''''
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
              with open("/output/results_withtext_singleItem_validation.txt", "a") as file_results:
                with open("/output/results_perCategory_training", "a") as file_perCategory_results:
                        with open("/output/results_perSubCategory_training", "a") as file_perSubCategory_results:
                            with open("/output/results_perAttributes_training", "a") as file_perAttributes_results:

                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                                    'Loss_subcategory {loss_subcategory.val:.4f} ({loss_subcategory.avg:.4f})\t'
                                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                                    'Prec@1 Sub Category {top1_subcategory.val:.3f} ({top1_subcategory.avg:.3f})\t'
                                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                                    'Prec@3 Sub Category {top3_subcategory.val:.3f} ({top3_subcategory.avg:.3f})\t'
                                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                                    'Prec@5 Sub Category {top5_subcategory.val:.3f} ({top5_subcategory.avg:.3f})\t'
                                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                                    i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time, loss_category =losses_category, loss_subcategory = losses_subcategory, loss_attributes = losses_attributes,
                                    top1_category =top1_category, top1_subcategory = top1_subcategory, top1_attributes = top1_attributes,
                                    top3_category =top3_category, top3_subcategory = top3_subcategory, top3_attributes = top3_attributes,
                                    top5_category=top5_category, top5_subcategory = top5_subcategory, top5_attributes= top5_attributes))
                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                                    'Loss_subcategory {loss_subcategory.val:.4f} ({loss_subcategory.avg:.4f})\t'
                                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                                    'Recall@1 Category {top1_recall_category.val:.3f} ({top1_recall_category.avg:.3f})\t'
                                    'Recall@3 Category {top3_recall_category.val:.3f} ({top3_recall_category.avg:.3f})\t'
                                    'Recall@5 Category {top5_recall_category.val:.3f} ({top5_recall_category.avg:.3f})\t'
                                    'Recall@1 Sub-Category {top1_recall_subcategory.val:.3f} ({top1_recall_subcategory.avg:.3f})\t'
                                    'Recall@3 Sub-Category {top3_recall_subcategory.val:.3f} ({top3_recall_subcategory.avg:.3f})\t'
                                    'Recall@4 Sub-Category {top5_recall_subcategory.val:.3f} ({top5_recall_subcategory.avg:.3f})\t'
                                    'Recall@1 Attributes {top1_recall_attributes.val:.3f} ({top1_recall_attributes.avg:.3f})\t'
                                    'Recall@3 Attributes {top3_recall_attributes.val:.3f} ({top3_recall_attributes.avg:.3f})\t'
                                    'Recall@5 Attributes {top5_recall_attributes.val:.3f} ({top5_recall_attributes.avg:.3f})'.format(
                                    i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time, loss_category =losses_category, loss_subcategory = losses_subcategory, loss_attributes = losses_attributes,
                                    top1_recall_category=top1_recall_category, top3_recall_category = top3_recall_category, top5_recall_category = top5_recall_category,
                                    top1_recall_subcategory = top1_recall_subcategory, top3_recall_subcategory = top3_recall_subcategory, top5_recall_subcategory=top5_recall_subcategory,
                                    top1_recall_attributes=top1_recall_attributes,top3_recall_attributes=top3_recall_attributes,top5_recall_attributes=top5_recall_attributes
                                    ))

                                print("\n")
                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Prec@5 Blouses {top5_precision_blouses.val:.3f} ({top5_precision_blouses.avg:.3f})\t'
                                        'Prec@5 Coats {top5_precision_coats.val:.3f} ({top5_precision_coats.avg:.3f})\t'
                                        'Prec@5 Dresses {top5_precision_dresses.val:.3f} ({top5_precision_dresses.avg:.3f})\t'
                                        'Prec@5 Jeans {top5_precision_jeans.val:.3f} ({top5_precision_jeans.avg:.3f})\t'
                                        'Prec@5 Jackets {top5_precision_jackets.val:.3f} ({top5_precision_jackets.avg:.3f})\t'
                                        'Prec@5 Jumpers {top5_precision_jumpers.val:.3f} ({top5_precision_jumpers.avg:.3f})\t'
                                        'Prec@5 Skirts {top5_precision_skirts.val:.3f} ({top5_precision_skirts.avg:.3f})\t'
                                        'Prec@5 Tights {top5_precision_tights.val:.3f} ({top5_precision_tights.avg:.3f})\t'
                                        'Prec@5 Tops {top5_precision_tops.val:.3f} ({top5_precision_tops.avg:.3f})\t'
                                        'Prec@5 Trousers {top5_precision_trousers.val:.3f} ({top5_precision_trousers.avg:.3f})\t'
                                        'Prec@5 Shoes {top5_precision_shoes.val:.3f} ({top5_precision_shoes.avg:.3f})\t'
                                        'Prec@5 Bags {top5_precision_bags.val:.3f} ({top5_precision_bags.avg:.3f})\t'
                                        'Prec@5 Accessories {top5_precision_accessories.val:.3f} ({top5_precision_accessories.avg:.3f})'.format(
                                         i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time,top5_precision_blouses = top5_precision_blouses,
                                        top5_precision_coats = top5_precision_coats, top5_precision_dresses = top5_precision_dresses, top5_precision_jeans = top5_precision_jeans,
                                        top5_precision_jackets = top5_precision_jackets,  top5_precision_jumpers = top5_precision_jumpers,
                                        top5_precision_skirts = top5_precision_skirts, top5_precision_tights = top5_precision_tights, top5_precision_tops = top5_precision_tops,
                                        top5_precision_trousers = top5_precision_trousers, top5_precision_shoes = top5_precision_shoes, top5_precision_bags = top5_precision_bags,
                                        top5_precision_accessories = top5_precision_accessories
                                         ))
                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Recall@5 Blouses {top5_recall_blouses.val:.3f} ({top5_recall_blouses.avg:.3f})\t'
                                        'Recall@5 Coats {top5_recall_coats.val:.3f} ({top5_recall_coats.avg:.3f})\t'
                                        'Recall@5 Dresses {top5_recall_dresses.val:.3f} ({top5_recall_dresses.avg:.3f})\t'
                                        'Recall@5 Jeans {top5_recall_jeans.val:.3f} ({top5_recall_jeans.avg:.3f})\t'
                                        'Recall@5 Jackets {top5_recall_jackets.val:.3f} ({top5_recall_jackets.avg:.3f})\t'
                                        'Recall@5 Jumpers {top5_recall_jumpers.val:.3f} ({top5_recall_jumpers.avg:.3f})\t'
                                        'Recall@5 Skirts {top5_recall_skirts.val:.3f} ({top5_recall_skirts.avg:.3f})\t'
                                        'Recall@5 Tights {top5_recall_tights.val:.3f} ({top5_recall_tights.avg:.3f})\t'
                                        'Recall@5 Tops {top5_recall_tops.val:.3f} ({top5_recall_tops.avg:.3f})\t'
                                        'Recall@5 Trousers {top5_recall_trousers.val:.3f} ({top5_recall_trousers.avg:.3f})\t'
                                        'Recall@5 Shoes {top5_recall_shoes.val:.3f} ({top5_recall_shoes.avg:.3f})\t'
                                        'Recall@5 Bags {top5_recall_bags.val:.3f} ({top5_recall_bags.avg:.3f})\t'
                                        'Recall@5 Accessories {top5_recall_accessories.val:.3f} ({top5_recall_accessories.avg:.3f})'.format(
                                        i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time, top5_recall_blouses = top5_recall_blouses,
                                         top5_recall_coats = top5_recall_coats, top5_recall_dresses = top5_recall_dresses, top5_recall_jeans = top5_recall_jeans,
                                        top5_recall_jackets = top5_recall_jackets,  top5_recall_jumpers = top5_recall_jumpers,
                                        top5_recall_skirts = top5_recall_skirts, top5_recall_tights = top5_recall_tights, top5_recall_tops = top5_recall_tops,
                                        top5_recall_trousers = top5_recall_trousers, top5_recall_shoes = top5_recall_shoes, top5_recall_bags = top5_recall_bags,
                                        top5_recall_accessories = top5_recall_accessories
                                        ))
                                print("\n")
                                #casual, watch, sandal, bags, topsides, short, legging, jumpsuit, skirts, sweater
                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Prec@5 Casual {top5_precisionS_casual.val:.3f} ({top5_precisionS_casual.avg:.3f})\t'
                                        'Prec@5 Watch {top5_precisionS_watch.val:.3f} ({top5_precisionS_watch.avg:.3f})\t'
                                        'Prec@5 Sandal {top5_precisionS_sandal.val:.3f} ({top5_precisionS_sandal.avg:.3f})\t'
                                        'Prec@5 Bagas {top5_precisionS_bags.val:.3f} ({top5_precisionS_bags.avg:.3f})\t'
                                        'Prec@5 Tops {top5_precisionS_tops.val:.3f} ({top5_precisionS_tops.avg:.3f})\t'
                                        'Prec@5 Shorts {top5_precisionS_shorts.val:.3f} ({top5_precisionS_shorts.avg:.3f})\t'
                                        'Prec@5 Leggings {top5_precisionS_legging.val:.3f} ({top5_precisionS_legging.avg:.3f})\t'
                                        'Prec@5 Jumpsuits {top5_precisionS_jumpsuits.val:.3f} ({top5_precisionS_jumpsuits.avg:.3f})\t'
                                        'Prec@5 Skirts {top5_precisionS_skirts.val:.3f} ({top5_precisionS_skirts.avg:.3f})\t'
                                        'Prec@5 Sweaters {top5_precisionS_sweater.val:.3f} ({top5_precisionS_sweater.avg:.3f})'.format(
                                         i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time, top5_precisionS_casual = top5_precisionS_casual,
                                        top5_precisionS_watch = top5_precisionS_watch, top5_precisionS_sandal = top5_precisionS_sandal, top5_precisionS_bags = top5_precisionS_bags,
                                        top5_precisionS_tops = top5_precisionS_tops,  top5_precisionS_shorts = top5_precisionS_shorts,
                                        top5_precisionS_legging = top5_precisionS_legging, top5_precisionS_jumpsuits = top5_precisionS_jumpsuit, top5_precisionS_skirts = top5_precisionS_skirts,
                                        top5_precisionS_sweater = top5_precisionS_sweater
                                         ))
                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Recall@5 Casual {top5_recallS_casual.val:.3f} ({top5_recallS_casual.avg:.3f})\t'
                                        'Recall@5 Watch {top5_recallS_watch.val:.3f} ({top5_recallS_watch.avg:.3f})\t'
                                        'Recall@5 Sandal {top5_recallS_sandal.val:.3f} ({top5_recallS_sandal.avg:.3f})\t'
                                        'Recall@5 Bagas {top5_recallS_bags.val:.3f} ({top5_recallS_bags.avg:.3f})\t'
                                        'Recall@5 Tops {top5_recallS_tops.val:.3f} ({top5_recallS_tops.avg:.3f})\t'
                                        'Recall@5 Shorts {top5_recallS_shorts.val:.3f} ({top5_recallS_shorts.avg:.3f})\t'
                                        'Recall@5 Leggings {top5_recallS_legging.val:.3f} ({top5_recallS_legging.avg:.3f})\t'
                                        'Recall@5 Jumpsuits {top5_recallS_jumpsuits.val:.3f} ({top5_recallS_jumpsuits.avg:.3f})\t'
                                        'Recall@5 Skirts {top5_recallS_skirts.val:.3f} ({top5_recallS_skirts.avg:.3f})\t'
                                        'Recall@5 Sweaters {top5_recallS_sweater.val:.3f} ({top5_recallS_sweater.avg:.3f})'.format(
                                         i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time, top5_recallS_casual = top5_recallS_casual,
                                        top5_recallS_watch = top5_recallS_watch, top5_recallS_sandal = top5_recallS_sandal, top5_recallS_bags = top5_recallS_bags,
                                        top5_recallS_tops = top5_recallS_tops,  top5_recallS_shorts = top5_recallS_shorts,
                                        top5_recallS_legging = top5_recallS_legging, top5_recallS_jumpsuits = top5_recallS_jumpsuit, top5_recallS_skirts = top5_recallS_skirts,
                                        top5_recallS_sweater = top5_recallS_sweater
                                         ))
                                print("\n")
                                #leather, lace, checked,print, floral,striped,polyester,colourful,denim,herringbone
                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Prec@5 leather {top5_precisionA_leather.val:.3f} ({top5_precisionA_leather.avg:.3f})\t'
                                        'Prec@5 lace {top5_precisionA_lace.val:.3f} ({top5_precisionA_lace.avg:.3f})\t'
                                        'Prec@5 checked {top5_precisionA_checked.val:.3f} ({top5_precisionA_checked.avg:.3f})\t'
                                        'Prec@5 print {top5_precisionA_print.val:.3f} ({top5_precisionA_print.avg:.3f})\t'
                                        'Prec@5 floral {top5_precisionA_floral.val:.3f} ({top5_precisionA_floral.avg:.3f})\t'
                                        'Prec@5 striped {top5_precisionA_striped.val:.3f} ({top5_precisionA_striped.avg:.3f})\t'
                                        'Prec@5 polyester {top5_precisionA_polyester.val:.3f} ({top5_precisionA_polyester.avg:.3f})\t'
                                        'Prec@5 colourful {top5_precisionA_colourful.val:.3f} ({top5_precisionA_colourful.avg:.3f})\t'
                                        'Prec@5 denim {top5_precisionA_denim.val:.3f} ({top5_precisionA_denim.avg:.3f})\t'
                                        'Prec@5 herringbone {top5_precisionA_herringbone.val:.3f} ({top5_precisionA_herringbone.avg:.3f})'.format(
                                         i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time, top5_precisionA_leather = top5_precisionA_leather,
                                        top5_precisionA_lace = top5_precisionA_lace, top5_precisionA_checked = top5_precisionA_checked, top5_precisionA_print = top5_precisionA_printed,
                                    top5_precisionA_floral= top5_precisionA_floral,top5_precisionA_striped=top5_precisionA_striped,top5_precisionA_polyester=top5_precisionA_ployester,
                                    top5_precisionA_colourful=top5_precisionA_colorful,top5_precisionA_denim=top5_precisionA_denim,top5_precisionA_herringbone=top5_precisionA_herringbone
                                         ))
                                print('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Recall@5 leather {top5_recallA_leather.val:.3f} ({top5_recallA_leather.avg:.3f})\t'
                                        'Recall@5 lace {top5_recallA_lace.val:.3f} ({top5_recallA_lace.avg:.3f})\t'
                                        'Recall@5 checked {top5_recallA_checked.val:.3f} ({top5_recallA_checked.avg:.3f})\t'
                                        'Recall@5 print {top5_recallA_print.val:.3f} ({top5_recallA_print.avg:.3f})\t'
                                        'Recall@5 floral {top5_recallA_floral.val:.3f} ({top5_recallA_floral.avg:.3f})\t'
                                        'Recall@5 striped {top5_recallA_striped.val:.3f} ({top5_recallA_striped.avg:.3f})\t'
                                        'Recall@5 polyester {top5_recallA_polyester.val:.3f} ({top5_recallA_polyester.avg:.3f})\t'
                                        'Recall@5 colourful {top5_recallA_colourful.val:.3f} ({top5_recallA_colourful.avg:.3f})\t'
                                        'Recall@5 denim {top5_recallA_denim.val:.3f} ({top5_recallA_denim.avg:.3f})\t'
                                        'Recall@5 herringbone {top5_recallA_herringbone.val:.3f} ({top5_recallA_herringbone.avg:.3f})'.format(
                                         i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time,top5_recallA_leather = top5_recallA_leather,
                                        top5_recallA_lace = top5_recallA_lace, top5_recallA_checked = top5_recallA_checked, top5_recallA_print = top5_recallA_printed,
                                    top5_recallA_floral= top5_recallA_floral,top5_recallA_striped=top5_recallA_striped,top5_recallA_polyester=top5_recallA_ployester,
                                    top5_recallA_colourful=top5_recallA_colorful,top5_recallA_denim=top5_recallA_denim,top5_recallA_herringbone=top5_recallA_herringbone
                                         ))
                                ''''
                                file_results.write('Test: [{0}/{1}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Loss_category {loss_category.val:.4f} ({loss_category.avg:.4f})\t'
                                    'Loss_subcategory {loss_subcategory.val:.4f} ({loss_subcategory.avg:.4f})\t'
                                    'Loss_attributes {loss_attributes.val:.3f} ({loss_attributes.avg:.3f})\t'
                                    'Prec@1 Category {top1_category.val:.3f} ({top1_category.avg:.3f})\t'
                                    'Prec@1 Sub Category {top1_subcategory.val:.3f} ({top1_subcategory.avg:.3f})\t'
                                    'Prec@1 Attributes {top1_attributes.val:.3f} ({top1_attributes.avg:.3f})\t'
                                    'Prec@3 Category {top3_category.val:.3f} ({top3_category.avg:.3f})\t'
                                    'Prec@3 Sub Category {top3_subcategory.val:.3f} ({top3_subcategory.avg:.3f})\t'
                                    'Prec@3 Attributes {top3_attributes.val:.3f} ({top3_attributes.avg:.3f})\t'
                                    'Prec@5 Category {top5_category.val:.3f} ({top5_category.avg:.3f})\t'
                                    'Prec@5 Sub Category {top5_subcategory.val:.3f} ({top5_subcategory.avg:.3f})\t'
                                    'Prec@5 Attributes {top5_attributes.val:.3f} ({top5_attributes.avg:.3f})'.format(
                                     i, len(val_loader),  batch_time=batch_time,
                                    data_time=batch_time, loss_category =losses_category, loss_subcategory = losses_subcategory, loss_attributes = losses_attributes,
                                    top1_category =top1_category, top1_subcategory = top1_subcategory, top1_attributes = top1_attributes,
                                    top3_category =top3_category, top3_subcategory = top3_subcategory, top3_attributes = top3_attributes,
                                    top5_category=top5_category, top5_subcategory = top5_subcategory, top5_attributes= top5_attributes))
                                file_results.write("\n")
                                '''''

        return top1_category.avg, top1_subcategory.avg,  top1_attributes.avg

    def save_checkpoint(self,state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '/output/model_best.pth.tar')

    def adjust_learning_rate(self,optimizer, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr

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
    def accuracy_multiLabel_subcategories(self,lstOutput, target, lstMultiLabel_GT, topk=(1,)):
        maxk = max(topk)

        val_top1 = 0.0
        val_top3 = 0.0
        val_top5 = 0.0

        val_recall_top1 = 0.0
        val_recall_top3 = 0.0
        val_recall_top5 = 0.0

        for i in range(0,len(lstOutput)):
            _, pred = lstOutput[i].topk(maxk, 1, True, True)

            pred = pred.t()
            gt = []
            for item in lstMultiLabel_GT:
                gt.append(int(item))

            res = []

            # Top 1,3,5 multi-label
            lstTop1Pred = [int(pred[0])]
            lstTop3Pred = [int(pred[0]), int(pred[1]), int(pred[2])]
            lstTop5Pred = [int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]), int(pred[4])]

            #print("Category predictions ...")
            #print(lstTop5Pred)

            # intersection for top 1
            intersection_top1 = [v for v in lstTop1Pred if v in gt]
            intersection_top3 = [v for v in lstTop3Pred if v in gt]
            intersection_top5 = [v for v in lstTop5Pred if v in gt]

            val_top1 += float(len(intersection_top1)) / float(len(pred))
            val_top3 += float(len(intersection_top3)) / float(len(pred))
            val_top5 += float(len(intersection_top5)) / float(len(pred))

            val_recall_top1 += float(len(intersection_top1)) / float(len(lstMultiLabel_GT))
            val_recall_top3 += float(len(intersection_top3)) / float(len(lstMultiLabel_GT))
            val_recall_top5 += float(len(intersection_top5)) / float(len(lstMultiLabel_GT))

        res.append(val_top1 * 100)
        res.append(val_top3* 100)
        res.append(val_top5* 100)

        res.append(val_recall_top1* 100)
        res.append(val_recall_top3* 100)
        res.append(val_recall_top5* 100)

        return res

# Helper Class
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


