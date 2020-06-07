# By Shatha Jaradat
# KTH - Royal Institute of Technology
# 2018

# Some parts of the code that are related to finetunning models was taken from the following link:
# Then customised according to my needs
#https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf

import torch
import torch.nn as nn
import torch.nn.functional as function
import Models.Constants_Instagram.Helper as helper
import numpy as np

######## Dynamic Layers for Single Class Scenarios ######
######## Was applied on Deep Fashion Dataseat ##########
######## Supports VGG16 and ResNet archiectures ########
class FineTuneModel(nn.Module):

    def __init__(self, original_model, arch):
        super(FineTuneModel, self).__init__()
        # intialisations of the categories and subcategories dictionaries
        self.FillDictionaries()
        self.initalizeSubCategories()
        self.categories_tensor = torch.FloatTensor(1,helper.Num_category_classes).zero_()
        self.sub_categories_tensor = torch.FloatTensor(1,helper.Num_subcategories).zero_()
        self.attributes_tensor = torch.FloatTensor(1, helper.Num_attributes).zero_()
        self.supportingCategory = -1
        self.supportingSubCategory = -1
        self.supportingAttributes = -1

        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.linear1 = nn.Linear(512, 512)
            self.generalCategoriesLinear = nn.Linear(512,helper.Num_category_classes)

           # Define the structure and the layers that represent the sub-categories of clothing items
            self.BlousesSubCategoriesLinear = nn.Linear(helper.Num_category_classes, helper.Num_blouses_subcategories)
            self.CoatsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_coats_subcategories)
            self.DressesSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_dresses_subcategories)
            self.JeansSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_jeans_subcategories)
            self.JacketsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_jackets_subcategories)
            self.JumpersSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_jumpers_subcategories)
            self.SkirtsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_skirts_subcategories)
            self.TightsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_tights_subcategories)
            self.TopsSubCategoriesLinear  = nn.Linear(helper.Num_category_classes,helper.Num_tops_subcategories)
            self.TrousersSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_trousers_subcategories)
            self.BagsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_bags_subcategories)
            self.AccessoriesSubCategoriesLinear = nn.Linear(helper.Num_category_classes, helper.Num_accessories_subcategories)
            self.ShoesSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_shoes_subcategories)

            # Attributes Classification Layers
            self.generalAttributesLinear = nn.Linear(512,helper.Num_attributes)

            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.linear1 = nn.Linear(25088, 4096)
            self.linear2 = nn.Linear(4096, 4096)
            self.generalCategoriesLinear = nn.Linear(4096,helper.Num_category_classes)

            # Define the structure and the layers that represent the sub-categories of clothing items
            self.BlousesSubCategoriesLinear = nn.Linear(helper.Num_category_classes, helper.Num_blouses_subcategories)
            self.CoatsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_coats_subcategories)
            self.DressesSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_dresses_subcategories)
            self.JeansSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_jeans_subcategories)
            self.JacketsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_jackets_subcategories)
            self.JumpersSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_jumpers_subcategories)
            self.SkirtsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_skirts_subcategories)
            self.TightsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_tights_subcategories)
            self.TopsSubCategoriesLinear  = nn.Linear(helper.Num_category_classes,helper.Num_tops_subcategories)
            self.TrousersSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_trousers_subcategories)
            self.BagsSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_bags_subcategories)
            self.AccessoriesSubCategoriesLinear = nn.Linear(helper.Num_category_classes, helper.Num_accessories_subcategories)
            self.ShoesSubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_shoes_subcategories)

            # Attributes Classification Layers
            self.generalAttributesLinear = nn.Linear(4096,helper.Num_attributes)

            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True

    # This method fills the categories dictionaries in the following way:
    # dictionary["blouses_and_tunics"] = 1  (key = category text, value = category id)
    # And it fills the subcategories mapping to category dictionary as follows:
    # dictionary[1] = "blouses_and_tunics" (key = subcatgory id, value = category text)
    def FillDictionaries(self):
        self.categoriesDict = {}
        self.dictSubCategories_Indices = {}

        with open(helper.categoriesIndiciesFilePath,"r") as file_categories:
            for line in file_categories:
                self.categoriesDict[int(line.split(",")[1])] = line.split(",")[0]
        with open(helper.categoriesSubcategoriesMappings, "r") as file_subcategories:
            for line in file_subcategories:
                self.dictSubCategories_Indices[int(line.split(",")[1])] = line.split(",")[3]

    def forward(self, x,supportingCategory, supportingSubCategory, supportingAttributes):
        f = self.features(x)

        if self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
            hrelu_linear2 = function.relu(self.linear1(f).clamp(min=0))
            number = 512
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
            hrelu_linear1 = function.relu(self.linear1(f).clamp(min=0))
            hrelu_linear2 = function.relu(self.linear2(hrelu_linear1))
            number = 4096


        lstPossibleCategories = []
        # add the first one from the supporting category
        lstPossibleCategories.append(int(supportingCategory))

        self.supportingCategory = (int)(supportingCategory)
        self.supportingSubCategory = (int)(supportingSubCategory)
        self.supportingAttributes = supportingAttributes

        inferredCategory_fromSubCategory = self.GetCategoryFromSubcategory()

        if inferredCategory_fromSubCategory not in self.lstPossibleCategories:
            self.lstPossibleCategories.append(int(inferredCategory_fromSubCategory))

        # Adjust categories tensor
        self.AdjustGeneralCategoriesLayer(number, hrelu_linear2)

        # Adjust sub-categories tensor
        self.AdjustSubCategories()

        # Adjust attributes tensor
        self.AdjustAttributesLayer(number,hrelu_linear2)

        lstOutputs = []
        lstOutputs.append(self.categories_tensor)
        lstOutputs.append(self.attributes_tensor)

        #TODO why not everything at once ?
        return lstOutputs, self.sub_categories_tensor

    def AdjustAttributesLayer(self,number,hrelu_linear2):

        # attributes
        dictAttributesIndices = {}

        # initializations for the correct scope of attributes based
        # on chosen category
        arr = np.empty(number)
        for i in range(0, number):
            arr[i] = helper.initialisation_attributes

        # Blouses
        lstSupportAttributes = []
        for item in self.supportingAttributes:
            lstSupportAttributes.append(int(item))
        indx = 1
        with open(helper.allAttributesFilePath, "r") as file_attributes:
            for line in file_attributes:
                dictAttributesIndices[indx] = (int(line.split(",")[1]))  # index of the attribute in the 1000 dimension
                indx += 1
            for counter in range(1, helper.Num_attributes):
                if dictAttributesIndices[counter] in lstSupportAttributes:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values
                    self.generalAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.generalAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_attributes
                    self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.generalAttributesLinear.weight.data[counter].requires_grad = True
                    self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr)
        hrelu_attributes = function.relu((self.generalAttributesLinear(hrelu_linear2)))
        index = 0
        for item in range(0, helper.Num_attributes):
            self.attributes_tensor[0][index] = hrelu_attributes[0].data[index]
            index += 1

    def AdjustGeneralCategoriesLayer(self, number,hrelu_linear2):
        # configure the weights of connections from the inferred categories
        # initializations for the correct scope of categories based
        # on chosen category
        arr = np.empty(number)
        for i in range(0, number):
            arr[i] = helper.initialisation_categories

        for counter in range(0, helper.Num_category_classes):
            # weights of unrelated categories can be zero
            # required gradient is set to false
            if counter not in self.lstPossibleCategories:
                self.generalCategoriesLinear.weight.data[counter] = torch.zeros(number)
                self.generalCategoriesLinear.weight.data[counter].requires_grad = False
            else:
                 # when a supporting category is sent or inferred
                 # weights of connections can be slightly increased
                 # to see the effect of supporting text values
                 self.generalCategoriesLinear.weight.data[counter].requires_grad = True
                 arr_temp = np.empty(number)
                 for i in range(0, number):
                     arr_temp[i] = self.generalCategoriesLinear.weight.data[counter][i]
                     arr_temp[i] += helper.weight_categories
                 self.generalCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)

        hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear2)))

        index = 0
        for item in range(0, helper.Num_category_classes):
            self.categories_tensor[0][index] = hrelu_generalcategories[0].data[index]
            index += 1


    def AdjustSubCategories(self,hrelu_generalcategories):
        # category that is sent as supporting item
        if self.categoriesDict[self.supportingCategory] == helper.Blouses_and_tunics or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Blouses_and_tunics:
            self.AdjustBlousesSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Coats or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Coats:
            self.AdjustCoatsSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Dresses or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Dresses:
            self.AdjustDressesSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Jeans or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Jeans:
            self.AdjustJeansSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Jackets or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Jackets:
            self.AdjustJacketsSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Jumpers_and_cardigans or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Jumpers_and_cardigans:
            self.AdjustJumpersSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Skirts or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Skirts:
            self.AdjustSkirtsSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Tichts_and_socks or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Tichts_and_socks:
            self.AdjustTightsSubCategoriesLayer(hrelu_generalcategories )
        if self.categoriesDict[self.supportingCategory] == helper.Tops_and_tshirts or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Tops_and_tshirts:
            self.AdjustTopsSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Trouser_and_shorts or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Trouser_and_shorts:
            self.AdjustTrousersSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Shoes or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Shoes:
            self.AdjustShoesSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.Bags or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.Bags:
            self.AdjustBagsSubCategoriesLayer(hrelu_generalcategories)
        if self.categoriesDict[self.supportingCategory] == helper.All_accessories or self.dictSubCategories_Indices[self.supportingSubCategory] == helper.All_accessories:
            self.AdjustAccessoriesSubCategoriesLayer(hrelu_generalcategories)

    # If the detected subcategories for the accessories
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustAccessoriesSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.accessories_subcategoriesFilePath,"r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_bags_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.AccessoriesSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.AccessoriesSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.AccessoriesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.AccessoriesSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.AccessoriesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_accessories_subcategories = function.relu((self.AccessoriesSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories( hrelu_accessories_subcategories,helper.accessories_subcategoriesFilePath, helper.Num_accessories_subcategories)

    # If the detected subcategories for the bags
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustBagsSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.bags_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_bags_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.BagsSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.BagsSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.BagsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.BagsSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.BagsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_bags_subcategories = function.relu((self.BagsSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_bags_subcategories,helper.bags_subcategoriesFilePath, helper.Num_bags_subcategories)

    # If the detected subcategories for the shoes
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustShoesSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.shoes_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_shoes_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.ShoesSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.ShoesSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.ShoesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.ShoesSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.ShoesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_shoes_subcategories = function.relu((self.ShoesSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_shoes_subcategories,helper.shoes_subcategoriesFilePath, helper.Num_shoes_subcategories)

    # If the detected subcategories for the trousers
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustTrousersSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.trousers_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_trousers_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.TrousersSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.TrousersSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.TrousersSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.TrousersSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.TrousersSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_trousers_subcategories = function.relu((self.TrousersSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_trousers_subcategories, helper.trousers_subcategoriesFilePath, helper.Num_trousers_subcategories)

    # If the detected subcategories for the tops
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustTopsSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.tops_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_tops_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.TopsSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.TopsSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.TopsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.TopsSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.TopsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_tops_subcategories = function.relu((self.TightsSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_tops_subcategories,helper.tops_subcategoriesFilePath, helper.Num_tights_subcategories)

    # If the detected subcategories for the tights
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustTightsSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.tights_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_tights_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.TightsSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.TightsSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.TightsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.TightsSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.TightsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_tights_subcategories = function.relu((self.TightsSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_tights_subcategories,helper.tights_subcategoriesFilePath, helper.Num_tights_subcategories)

    # If the detected subcategories for the skirts
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustSkirtsSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.skirts_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_skirts_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.SkirtsSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.SkirtsSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.SkirtsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.SkirtsSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.SkirtsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_skirts_subcategories = function.relu((self.SkirtsSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_skirts_subcategories, helper.skirts_subcategoriesFilePath, helper.Num_skirts_subcategories)

    # If the detected subcategories for the jumpers
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustJumpersSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.jumpers_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_jumpers_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.JumpersSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.JumpersSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.JumpersSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.JumpersSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.JumpersSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_jumpers_subcategories = function.relu((self.JumpersSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_jumpers_subcategories,helper.jumpers_subcategoriesFilePath, helper.Num_jumpers_subcategories)

    # If the detected subcategories for the jackets
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustJacketsSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.jackets_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_jackets_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.JacketsSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.JacketsSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.JacketsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.JacketsSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.JacketsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_jackets_subcategories = function.relu((self.JacketsSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_jackets_subcategories,helper.jackets_subcategoriesFilePath, helper.Num_jackets_subcategories)

    # If the detected subcategories for the jeans
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustJeansSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.jeans_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_jeans_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.JeansSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.JeansSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.JeansSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.JeansSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.JeansSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_jeans_subcategories = function.relu((self.JeansSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_jeans_subcategories,helper.jeans_subcategoriesFilePath, helper.Num_jeans_subcategories)

    # If the detected subcategories for the dresses
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustDressesSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.dresses_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_dresses_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.DressesSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.DressesSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.DressesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.DressesSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.DressesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_dresses_subcategories = function.relu((self.DressesSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_dresses_subcategories,helper.dresses_subcategoriesFilePath, helper.Num_dresses_subcategories)

    # If the detected subcategories for the coats
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustCoatsSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.coats_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_coats_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.CoatsSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.CoatsSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.CoatsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.CoatsSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.CoatsSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_coats_subcategories = function.relu((self.CoatsSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories(hrelu_coats_subcategories,helper.coats_subcategoriesFilePath, helper.Num_coats_subcategories)

    # If the detected subcategories for the blouses
    # Adjust weights of connections of the specific range for that specific sub-category in their layer
    # Map the adjustments to a unified layer for the whole subcategories for the computational loss step
    def AdjustBlousesSubCategoriesLayer(self,hrelu_generalcategories):
        indx = 0
        dictSubCategories_Indices = {}
        with open(helper.blouses_subcategoriesFilePath, "r") as file_subcategories:
            for line in file_subcategories:
                dictSubCategories_Indices[indx] = (int(line))  # index of the subcategory
                indx += 1
        for counter in range(0, helper.Num_blouses_subcategories - 1):
            if dictSubCategories_Indices[counter] == self.supportingSubCategory:  # when a supporting subcategory is sent
                # weights of connections can be slightly increased
                # to see the effect of supporting text values

                self.BlousesSubCategoriesLinear.weight.data[counter].requires_grad = True
                arr_temp = np.empty(helper.Num_category_classes)
                for i in range(0, helper.Num_category_classes):
                    arr_temp[i] = self.BlousesSubCategoriesLinear.weight.data[counter][i]
                    arr_temp[i] += helper.weight_subcategories
                self.BlousesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
            else:
                self.BlousesSubCategoriesLinear.weight.data[counter].requires_grad = True
                self.BlousesSubCategoriesLinear.weight.data[counter] = torch.from_numpy(self.arr_subcategories)

        # connect with the blouses sub category
        hrelu_blouses_subcategories = function.relu((self.BlousesSubCategoriesLinear(hrelu_generalcategories)))
        self.fixMapping_forSubCategories( hrelu_blouses_subcategories,
                                                       helper.blouses_subcategoriesFilePath, helper.Num_blouses_subcategories)

    # The file subcategories_indices contains: subcategory (text), subcategory (id), and its parent category (id)
    # In this method, we have a subcategory id and we want to get the parent category id that it belongs to
    def GetCategoryFromSubcategory(self):

        # Reading from the file to get the parent category id then break
        categoryID = -1
        with open(helper.categoriesSubcategoriesMappings, "r") as file_mappings:
            for line in file_mappings:
                subcategory_id = int(line.split(",")[1])
                parentcategory_id = int(line.split(",")[2])
                if self.supportingSubCategory == subcategory_id:
                    categoryID = parentcategory_id
                    break
        return categoryID

    # initialisation for subcategories
    def initalizeSubCategories(self):
        self.arr_subcategories = np.empty(helper.Num_subcategories)
        for i in range(0,helper.Num_category_classes):
            self.arr_subcategories[i] = helper.intialisation_subcategories

    # Map the adjustments to a unified layer for the whole attributes for the computational loss step
    def fixMapping_forSubCategories(self, hrelu_subcategories,  filesubcategoryPath, numsubcategories):
        lstValues_subcategories = [0 for x in range(numsubcategories)]

        for vals in hrelu_subcategories:
            lstValues_subcategories = vals.data

        # lstValues_all_subcategories = torch.FloatTensor(1,124).zero_()
        indx = 0
        with open(filesubcategoryPath, "r") as file_subcategories:
            for line in file_subcategories:
                self.sub_categories_tensor[0][int(line)] = lstValues_subcategories[indx]
                indx += 1

