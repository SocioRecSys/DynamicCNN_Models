class Helper(object):

    # Instagram dataset
    Num_category_classes = 13
    Num_subcategories = 125
    Num_attributes = 127

    Num_blouses_subcategories = 3
    Num_coats_subcategories = 8
    Num_dresses_subcategories = 10
    Num_jeans_subcategories = 8
    Num_jackets_subcategories = 14
    Num_jumpers_subcategories = 7
    Num_skirts_subcategories = 10
    Num_tights_subcategories = 6
    Num_tops_subcategories = 10
    Num_trousers_subcategories = 7
    Num_shoes_subcategories = 26
    Num_bags_subcategories = 9
    Num_accessories_subcategories = 9

    Blouses_and_tunics = "blouses_and_tunics"
    Coats = "coats"
    Dresses ="dresses"
    Jeans = "jeans"
    Jackets ="jackets"
    Jumpers_and_cardigans ="jumpers_and_cardigans"
    Skirts ="skirts"
    Tichts_and_socks = "tichts_and_socks"
    Tops_and_tshirts = "tops_and_tshirts"
    Trouser_and_shorts ="trouser_and_shorts"
    Shoes = "shoes"
    Bags = "bags"
    All_accessories = "all_accessories"

    weight_subcategories = 0.1
    weight_attributes = 0.1
    weight_categories = 0.1
    intialisation_subcategories = 0.01
    initialisation_attributes =0.01
    initialisation_categories = 0.01

    # Files Paths
    # Paths to data folders - Paths are set according to the structure of data folders in the Floydhub project
    # However they match the general structure in the Datasets folder in this project
    categoryAttributesFolderPath = "/data/category-attributes/"
    attributesClassificationsPath = "/data/attributes-classifications/"
    categoriesSubcategoriesMappings = "/data/categories-subcategories/subcategories_indices.txt"
    categoriesIndiciesFilePath = "/data/category-subcategories/categories_indices"
    subcategoriesClassifiedPath = "/data/all_subcategories_classified/"
    blouses_subcategoriesFilePath = "/data/all_subcategories_classified/blouses_subcategories"
    accessories_subcategoriesFilePath = "/data/all_subcategories_classified/accessories_subcategories"
    bags_subcategoriesFilePath = "/data/all_subcategories_classified/bags_subcategories"
    coats_subcategoriesFilePath = "/data/all_subcategories_classified/coats_subcategories"
    dresses_subcategoriesFilePath = "/data/all_subcategories_classified/dresses_subcategories"
    jackets_subcategoriesFilePath = "/data/all_subcategories_classified/jackets_subcategories"
    jeans_subcategoriesFilePath = "/data/all_subcategories_classified/jeans_subcategories"
    jumpers_subcategoriesFilePath = "/data/all_subcategories_classified/jumpers_subcategories"
    shoes_subcategoriesFilePath = "/data/all_subcategories_classified/shoes_subcategories"
    skirts_subcategoriesFilePath = "/data/all_subcategories_classified/skirts_subcategories"
    tights_subcategoriesFilePath = "/data/all_subcategories_classified/tights_subcategories"
    tops_subcategoriesFilePath = "/data/all_subcategories_classified/tops_subcategories"
    trousers_subcategoriesFilePath = "/data/all_subcategories_classified/trousers_subcategories"
    allAttributesFilePath = "/data/attributes-classifications/all_attributes"
    stylesIndices = "/data/category-subcategories/style_indices"
    imgsPath = "/data/imgs/"
    partitionImgs ="/data/groundTurth/partitionedImages"

    list_images_categories = "/data/text-analysis/image-categories"
    list_images_sub_categories = "/data/text-analysis/image-sub-categories"
    list_images_attributes = "/data/text-analysis/image-attributes"
    list_images_styles = "/data/text-analysis/image-styles"



    imagesNumTraining = 8000
    imagesNumValidation = 2000
