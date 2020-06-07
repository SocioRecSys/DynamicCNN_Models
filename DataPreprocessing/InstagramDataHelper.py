# KTH - Royal Institute of Technology
# 2018

import Models.Constants_Instagram.Helper as helper
from PIL import Image
import os
import sys
import numpy as np
from torchvision import transforms
from skimage import io, transform
import torch

# Helper for reading meta-data, partitioning images and other methods
# For Instagram Dataset
# it has categories, sub-categories and attributes
class DataHelper(object):

    # Reading clothing categories and their indices
    def readCategoryIndices(self):
        dict_categories = {}
        with open(helper.category_indices_file, "r") as file_category_indices:
            for line in file_category_indices:
                dict_categories[line.split(",")[0]] = line.split(",")[1]
        return dict_categories

    # Reading clothing styles and their indices
    def readStyleIndices(self):
        dict_styles = {}
        with open(helper.style_indices, "r") as file_style_indices:
            for line in file_style_indices:
                dict_styles[line.split(",")[0]] = line.split(",")[1]
        return dict_styles

    # Reading clothing sub-categories and their indices
    def readSubCategoryIndices(self):
        dict_sub_categories = {}
        counter = 0
        with open(helper.categoriesSubcategoriesMappings, "r") as file_sub_category_indices:
            for line in file_sub_category_indices:
                if counter < 1:
                    counter += 1
                    continue
                dict_sub_categories[line.split(",")[0]] = line.split(",")[1]
        return dict_sub_categories

    # Reading clothing attributes
    def readAttributes(self):
        dict_attributes = {}
        counter = 0
        with open(helper.allAttributesFilePath, "r") as file_attributes_indices:
            for line in file_attributes_indices:
                dict_attributes[line.split(",")[0]] = line.split(",")[1]
        return dict_attributes

    # Partition the images into train and validation
    # partitioning 8000 training
    def partition(self):
        counter = 0
        with open (helper.partitionImgs, "a") as file_partitioning:
            for filename in os.listdir(helper.imgsPath):
                if counter <= helper.imagesNumTraining:
                    file_partitioning.write(filename + ',train')
                    file_partitioning.write("\n")
                else:
                    file_partitioning.write(filename+ ',val')
                    file_partitioning.write("\n")
                    #if (counter == 10000): break
                counter += 1

    # Read the training and validiation images into arrays
    def partitionImages(self):
        with open(helper.partitionImgs, "r") as file_eval_partition:
            self.trainImages = []
            self.valImages = []
            for line in file_eval_partition:
                imagePath =line.split(",")[0]
                imageType = line.split(",")[1]

                if(imageType.strip() == 'train'):
                    self.trainImages.append(imagePath)
                elif(imageType.strip() == 'val'):
                    self.valImages.append(imagePath)

    # Initialisations for methods to fill dictionaries and partition images
    def __init__(self, root_dir_dataset, images_dir, mode='train'):
        self.dict_categories = self.readCategoryIndices()
        self.dict_sub_categories = self.readSubCategoryIndices()
        self.dict_attributes = self.readAttributes()
        self.dict_styles = self.readStyleIndices()
        self.partitionImages()
        self.mode = mode

        self.root_dir_dataset = root_dir_dataset
        self.images_dir = images_dir
        #self.transform = transform

    # Image preprocessing, resize and normalisation
    def __getitem__(self, index):

        # Reading images here is more memory efficient
        # Images are not stored in memory at once, but read as required
        counter = 0

        listImages = []
        if(self.mode == 'train'):
            listImages = self.trainImages
        elif (self.mode == 'val'):
            listImages= self.valImages

        for imagePath in listImages:
            if index != counter:
                counter += 1
                continue
            img_name = os.path.join(self.images_dir,imagePath)
            try:
                image = Image.open(img_name)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print(img_name)
                continue

            normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
            )
            preprocess = transforms.Compose([
                        transforms.ToTensor()
            ])
            image = np.array(image)

            image = transform.resize(image, (224, 224, 3), preserve_range=True, mode='constant')

            image -= np.array([123.68, 116.78, 103.94])

            #image = np.moveaxis(image, 2, 0)

            img_tensor = preprocess(image)

            imagePath = imagePath.replace(".jpg","")
            attributes, lstIndices = self.readImageAttributes(imagePath)
            categoriesIds, categoryTensor = self.readImageCategories(imagePath)
            sub_categoriesIds, sub_categoryTensor = self.readImageSubCategories(imagePath)

            break


        return img_tensor, categoryTensor,sub_categoryTensor, attributes,  categoriesIds,  sub_categoriesIds, lstIndices

    # Read Image categories by Image ID - input from SemCluster component
    def readImageCategories(self, image_id):
        lstCategories = []
        with open(helper.list_images_categories, "r") as file_image_categories:
            for line in file_image_categories:
                if(line.__contains__(image_id)):
                    lstCategories.append(int(line.split(",")[1]))
                    lstCategories.append(int(line.split(",")[2]))
                    lstCategories.append(int(line.split(",")[3]))
                    lstCategories.append(int(line.split(",")[4]))

        categoryTensor = torch.FloatTensor(helper.Num_category_classes).zero_()
        for i in range (0,helper.Num_category_classes):
            if lstCategories.__contains__(i):
                categoryTensor[i] = 1.0
            else:
                categoryTensor[i] = 0.0

        return lstCategories, categoryTensor

    # Read image subcategories by Image ID
    def readImageSubCategories(self, image_id):
        lstSubCategories = []
        with open(helper.list_images_sub_categories, "r") as file_image_sub_categories:
            for line in file_image_sub_categories:
                if(line.__contains__(image_id)):
                    lstSubCategories.append(int(line.split(",")[1]))
                    lstSubCategories.append(int(line.split(",")[2]))
                    lstSubCategories.append(int(line.split(",")[3]))
                    lstSubCategories.append(int(line.split(",")[4]))

            subCategoriesTensor = torch.FloatTensor(helper.Num_subcategories).zero_()
            for i in range (0,helper.Num_subcategories):
                if lstSubCategories.__contains__(i):
                    subCategoriesTensor[i] = 1.0
                else:
                    subCategoriesTensor[i] = 0.0

        return lstSubCategories, subCategoriesTensor

    # Read Image Attributes by Image ID
    def readImageAttributes(self, image_id):
        lstAttributes = []
        with open(helper.list_images_attributes, "r") as file_image_attributes:
            for line in file_image_attributes:
                if(line.__contains__(image_id)):
                    lstAttributes.append(int(line.split(",")[1]))
                    lstAttributes.append(int(line.split(",")[2]))
                    lstAttributes.append(int(line.split(",")[3]))
                    lstAttributes.append(int(line.split(",")[4]))
                    lstAttributes.append(int(line.split(",")[5]))
                    lstAttributes.append(int(line.split(",")[6]))
                    lstAttributes.append(int(line.split(",")[7]))
                    lstAttributes.append(int(line.split(",")[8]))

            attributesTensor = torch.FloatTensor(helper.Num_attributes).zero_()
            for i in range (0,helper.Num_attributes):
                if any(x == i for x in lstAttributes):
                    attributesTensor[i] = 1.0
                else:
                    attributesTensor[i] = 0.0

        return attributesTensor,  lstAttributes