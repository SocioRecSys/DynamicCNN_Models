# By Shatha Jaradat
# KTH - Royal Institute of Technology
# 2018

# Some parts of the code that are related to finetunning models was taken from the following link:
# Then customised according to my needs
#https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf


import torch
import torch.nn as nn
import torch.nn.functional as function
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import Models.Constants_DeepFashion.Helper as helper

######## Dynamic Layers for Single Class Scenarios ######
######## Was applied on Deep Fashion Dataseat ##########
######## Supports VGG16 and ResNet archiectures ########
class FineTuneModel(nn.Module):

    def __init__(self, original_model, arch):
        super(FineTuneModel, self).__init__()
        self.cat = -1
        self.attr = -1
        self.attr2 = -1
        # tensor of categories 1*50
        self.categories_tensor = torch.FloatTensor(1,helper.Num_category_classes).zero_()
        # tensor of attributes 1*1000
        self.attributes_tensor = torch.FloatTensor(1,helper.Num_attribute_classes_general).zero_()
        self.categoryAttributesMappings = {}

        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.linear1 = nn.Linear(512, 512)
            self.generalCategoriesLinear = nn.Linear(512,50)

            # The structure contains the different attributes layers but at runtime
            # only one attribute layer is connected (forward method)
            self.BlousesAttributesLinear = nn.Linear(512,helper.Num_blouses_attributes)
            self.CoatsAttributesLinear = nn.Linear(512,helper.Num_coats_attributes)
            self.DressesAttributesLinear = nn.Linear(512,helper.Num_dresses_attributes)
            self.JeansAttributesLinear = nn.Linear(512,helper.Num_jeans_attributes)
            self.JacketsAttributesLinear = nn.Linear(512,helper.Num_jackets_attributes)
            self.JumpersCardigansAttributesLinear = nn.Linear(512,helper.Num_jumpersCardigans_attributes)
            self.SkirtsAttributesLinear = nn.Linear(512,helper.Num_skirts_attributes)
            self.TightsSocksAttributesLinear = nn.Linear(512,helper.Num_tightsSocks_attributes)
            self.TopsTShirtsAttributesLinear = nn.Linear(512,helper.Num_topsTShirts_attributes)
            self.TrousersShortsAttributesLinear = nn.Linear(512,helper.Num_trousersShorts_attributes)

            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.linear1 = nn.Linear(25088, 4096)
            self.linear2 = nn.Linear(4096, 4096)
            self.generalCategoriesLinear = nn.Linear(4096,50)

            # The structure contains the different attributes layers but at runtime
            # only one attribute layer is connected (forward method)
            self.BlousesAttributesLinear = nn.Linear(4096,helper.Num_blouses_attributes)
            self.CoatsAttributesLinear = nn.Linear(4096,helper.Num_coats_attributes)
            self.DressesAttributesLinear = nn.Linear(4096,helper.Num_dresses_attributes)
            self.JeansAttributesLinear = nn.Linear(4096,helper.Num_jeans_attributes)
            self.JacketsAttributesLinear = nn.Linear(4096,helper.Num_jackets_attributes)
            self.JumpersCardigansAttributesLinear = nn.Linear(4096,helper.Num_jumpersCardigans_attributes)
            self.SkirtsAttributesLinear = nn.Linear(4096,helper.Num_skirts_attributes)
            self.TightsSocksAttributesLinear = nn.Linear(4096,helper.Num_tightsSocks_attributes)
            self.TopsTShirtsAttributesLinear = nn.Linear(4096,helper.Num_topsTShirts_attributes)
            self.TrousersShortsAttributesLinear = nn.Linear(4096,helper.Num_trousersShorts_attributes)

            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x,categoryItem_Value, attributeItem_Value, attribute2_Value):
        f = self.features(x)

        self.cat = int(categoryItem_Value)
        self.attr = int(attributeItem_Value)
        self.attr2 = int(attribute2_Value)

        # output contains the 1 * 50 tensor for categories
        lstOutputs = []

        if self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
            hrelu_linear2 = function.relu(self.linear1(f).clamp(min=0))
            hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear2)))
            self.CreateAttributesTensor(512, hrelu_linear2)

        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
            hrelu_linear1 = function.relu(self.linear1(f).clamp(min=0))
            hrelu_linear2 = function.relu(self.linear2(hrelu_linear1))
            hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear2)))
            self.CreateAttributesTensor(4096, hrelu_linear2)

        index = 0
        for item in range(0,helper.Num_category_classes):
            self.categories_tensor[0][index] = hrelu_generalcategories[0].data[index]
            index += 1
        lstOutputs.append(self.categories_tensor)
        lstOutputs.append(self.attributes_tensor)

        return lstOutputs

    def CreateAttributesTensor(self, number, hrelu_linear2):
        # fill with the mappings of attributes ==> to which category they belong
        with open(helper.categoryAttributesMappingPath,"r") as file_mappings:
            for line in file_mappings:
                self.categoryAttributesMappings[int(line.split(",")[0])] = line.split(",")[1]

        # initializations for the correct scope of attributes based
        # on chosen category
        arr = np.empty(number)
        for i in range(0,number):
            arr[i] = helper.weight_initialisation

        self.CustomiseBlouses(arr, hrelu_linear2, number)
        self.CustomiseCoats(arr,  hrelu_linear2,number)
        self.CustomiseDresses(arr, hrelu_linear2, number)
        self.CustomiseJeans(arr, hrelu_linear2, number)
        self.CustomiseJackets(arr, hrelu_linear2,number)
        self.CustomiseJumpers(arr, hrelu_linear2,number)
        self.CustomiseSkirts(arr, hrelu_linear2, number)
        self.CustomiseLeggings(arr, hrelu_linear2, number)
        self.CustomiseTops(arr, hrelu_linear2, number)
        self.CustomiseTrousers(arr, hrelu_linear2,number)

    # Detect if the attributes of the guiding category belong to trousers attributes
    # Adjust weights of connections of the specific range for the trousers in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseTrousers(self, arr,hrelu_linear2, number):
        # Capris, Chinos, Culottes, Cutoffs, Shorts, SweatPants, SweatShorts, Trunks
        if (self.categoryAttributesMappings[self.cat] == helper.trousersshortsAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.trousersshortsAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_trousersShorts_attributes):
                if dictIndices[counter] == self.attr:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values
                    self.TrousersShortsAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.TrousersShortsAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.TrousersShortsAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.TrousersShortsAttributesLinear.weight.data[counter].requires_grad = True
                    self.TrousersShortsAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_trousersShorts_attributes = function.relu((self.TrousersShortsAttributesLinear(hrelu_linear2)))
            # Fixed size for loss computation
            self.fixMapping_forLoss(hrelu_trousersShorts_attributes,helper.trousersshortsAttributes, helper.Num_trousersShorts_attributes)

    # Detect if the attributes of the guiding category belong to tops and t-shirts attributes
    # Adjust weights of connections of the specific range for the tops and t-shirts  in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseTops(self, arr,hrelu_linear2, number):
        if (self.categoryAttributesMappings[self.cat] == helper.topshortsAttributes):
            dictIndices = {}
            indx = 0
            with open(helper.attributesFolderPath + helper.topshortsAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_topsTShirts_attributes):
                if dictIndices[counter] == self.attr:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.TopsTShirtsAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.TopsTShirtsAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.TopsTShirtsAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.TopsTShirtsAttributesLinear.weight.data[counter].requires_grad = True
                    self.TopsTShirtsAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_topsShirts_attributes = function.relu((self.TopsTShirtsAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_topsShirts_attributes,
                                                   helper.topshortsAttributes, helper.Num_topsTShirts_attributes)


    # Detect if the attributes of the guiding category belong to leggings attributes
    # Adjust weights of connections of the specific range for the leggings in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseLeggings(self, arr,hrelu_linear2, number):
        # Leggings and Joggers
        if (self.categoryAttributesMappings[self.cat] == helper.tightssocksAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.tightssocksAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_tightsSocks_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.TightsSocksAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.TightsSocksAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.TightsSocksAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.TightsSocksAttributesLinear.weight.data[counter].requires_grad = True
                    self.TightsSocksAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_tightsSocks_attributes = function.relu((self.TightsSocksAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_tightsSocks_attributes,
                                                   helper.tightssocksAttributes, helper.Num_tightsSocks_attributes)

    # Detect if the attributes of the guiding category belong to skirts attributes
    # Adjust weights of connections of the specific range for the skirts in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseSkirts(self, arr,hrelu_linear2, number):
        if (self.categoryAttributesMappings[self.cat] == helper.skirtsAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.skirtsAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_skirts_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.SkirtsAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.SkirtsAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.SkirtsAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.SkirtsAttributesLinear.weight.data[counter].requires_grad = True
                    self.SkirtsAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_skirts_attributes = function.relu((self.SkirtsAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_skirts_attributes, helper.skirtsAttributes,
                                                   helper.Num_skirts_attributes)

    # Detect if the attributes of the guiding category belong to jumpers attributes
    # Adjust weights of connections of the specific range for the jumpers in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseJumpers(self, arr,hrelu_linear2, number):
        # Cardigan, Poncho, Gauchos, Caftan, Cape, Coverup, Kaftan, Kimono, Onesie, Robe, Romper
        if (self.categoryAttributesMappings[self.cat] == helper.jumperscardigansAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.jumperscardigansAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_jumpersCardigans_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.JumpersCardigansAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.JumpersCardigansAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.JumpersCardigansAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.JumpersCardigansAttributesLinear.weight.data[counter].requires_grad = True
                    self.JumpersCardigansAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_jumpers_cardigans_attributes = function.relu((self.JumpersCardigansAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_jumpers_cardigans_attributes,
                                                   helper.jumperscardigansAttributes, helper.Num_jumpersCardigans_attributes)

    # Detect if the attributes of the guiding category belong to jackets attributes
    # Adjust weights of connections of the specific range for the jackets in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseJackets(self, arr,hrelu_linear2, number):
        if (self.categoryAttributesMappings[self.cat] == helper.jacketsAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.jacketsAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_jackets_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.JacketsAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.JacketsAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.JacketsAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.JacketsAttributesLinear.weight.data[counter].requires_grad = True
                    self.JacketsAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_jackets_attributes = function.relu((self.JacketsAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_jackets_attributes,
                                                   helper.jacketsAttributes, helper.Num_jackets_attributes)

    # Detect if the attributes of the guiding category belong to jeans attributes
    # Adjust weights of connections of the specific range for the jeans in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseJeans(self, arr,hrelu_linear2, number):
        # Jeans
        if (self.categoryAttributesMappings[self.cat] == helper.jeansAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.jeansAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_jeans_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.JeansAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.JeansAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.JeansAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.JeansAttributesLinear.weight.data[counter].requires_grad = True
                    self.JeansAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_jeans_attributes = function.relu((self.JeansAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_jeans_attributes, helper.jeansAttributes,
                                                   helper.Num_jeans_attributes)

    # Detect if the attributes of the guiding category belong to dresses attributes
    # Adjust weights of connections of the specific range for the dresses in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseDresses(self, arr,hrelu_linear2, number):
        if (self.categoryAttributesMappings[self.cat] == helper.dressesAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.dressesAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_dresses_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.DressesAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.DressesAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.DressesAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.DressesAttributesLinear.weight.data[counter].requires_grad = True
                    self.DressesAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_dresses_attributes = function.relu((self.DressesAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_dresses_attributes,
                                                   helper.dressesAttributes, helper.Num_dresses_attributes)

    # Detect if the attributes of the guiding category belong to Coats attributes
    # Adjust weights of connections of the specific range for the Coats in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseCoats(self, arr,hrelu_linear2, number):
        # Anorak , Parka , PeaCoat, and Coat
        if (self.categoryAttributesMappings[self.cat] == helper.coatsAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.coatsAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1

            for counter in range(0, helper.Num_coats_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.CoatsAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.CoatsAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.CoatsAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.CoatsAttributesLinear.weight.data[counter].requires_grad = True
                    self.CoatsAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_coat_attributes = function.relu((self.CoatsAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_coat_attributes, helper.coatsAttributes,
                                                   helper.Num_coats_attributes)

    # Detect if the attributes of the guiding category belong to blouses attributes
    # Adjust weights of connections of the specific range for the blouses in their layer
    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def CustomiseBlouses(self, arr, hrelu_linear2, number):
        if (self.categoryAttributesMappings[self.cat] == helper.blousestunicsAttributes):
            indx = 0
            dictIndices = {}
            with open(helper.attributesFolderPath + helper.blousestunicsAttributes + ".txt", "r") as file_attributes:
                for line in file_attributes:
                    dictIndices[indx] = (int(line))  # index of the attribute in the 1000 dimension
                    indx += 1
            for counter in range(0, helper.Num_blouses_attributes):
                if dictIndices[counter] == self.attr or dictIndices[counter] == self.attr2:
                    # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values

                    self.BlousesAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.BlousesAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_adjustment
                    self.BlousesAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.BlousesAttributesLinear.weight.data[counter].requires_grad = True
                    self.BlousesAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_blouses_attributes = function.relu((self.BlousesAttributesLinear(hrelu_linear2)))
            self.fixMapping_forLoss(hrelu_blouses_attributes, helper.blousestunicsAttributes, helper.Num_blouses_attributes)

    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def fixMapping_forLoss(self, hrelu_attributes, fileAttributeName, numAttributes):
        lstValues_attributes = [0 for x in range(numAttributes)]

        for vals in hrelu_attributes:
            lstValues_attributes = vals.data

        indx = 0
        with open(helper.attributesFolderPath + fileAttributeName + ".txt" ,"r") as file_attributes:
            for line in file_attributes:
                self.attributes_tensor[0][int(line)] = lstValues_attributes[indx]
                indx += 1


