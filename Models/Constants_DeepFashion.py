class Helper(object):

    Num_category_classes = 50
    Num_attribute_classes_general = 1001
    Num_blouses_attributes = 145
    Num_coats_attributes = 49
    Num_dresses_attributes = 112
    Num_jeans_attributes = 55
    Num_jackets_attributes = 269
    Num_jumpersCardigans_attributes =233
    Num_skirts_attributes = 157
    Num_tightsSocks_attributes =72
    Num_topsTShirts_attributes =441
    Num_trousersShorts_attributes =256
    
    categoryAttributesMappingPath = "/data/all_attributes_classified/categoryAttributesMappings"
    attributesFolderPath = "/data/all_attributes_classified/"
    accessoriesAttributes = "accessories_attributes"
    bagsAttributes= "bags_attributes"
    blousestunicsAttributes = "blouses_tunics_attributes"
    coatsAttributes = "coats_attributes"
    dressesAttributes = "dresses_attributes"
    jacketsAttributes = "jackets_attributes"
    jeansAttributes = "jeans_attributes"
    jumperscardigansAttributes = "jumpers_cardigans_attributes"
    shoesAttributes = "shoes_attributes"
    skirtsAttributes = "skirts_attributes"
    tightssocksAttributes = "tights_socks_attributes"
    topshortsAttributes = "tops_shorts_attributes"
    trousersshortsAttributes = "trousers_shorts_attributes"

    category_file_path = "/data/deepfashion/list_category_cloth.txt"
    attribute_category_file_path = "/data/deepfashion/list_attr_cloth.txt"
    category_images_file_path ="/data/deepfashion/list_category_img.txt"
    attribute_images_file_path = "/data/deepfashion/list_attr_img.txt"
    eval_partition_file_path = "/data/deepfashion/list_eval_partition.txt"


    weight_intialisation = 0.01
    weight_adjustment = 0.1