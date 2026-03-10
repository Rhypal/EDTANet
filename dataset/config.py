# 数据集配置信息, 完整类别列表
DATASETS = {
    "PVi": {        # 38 class
        "path": "/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture/PlantVillage",
        "classes": [
            'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Blueberry_Healthy',
            'Cherry_Healthy', 'Cherry_Powdery_mildew', 'Citrus_Greening', 'Corn_Common_rust', 'Corn_Gray_leaf_spot',
            'Corn_Healthy', 'Corn_Northern_leaf_blight', 'Grape_Black_measles', 'Grape_Black_rot', 'Grape_Healthy',
            'Grape_Leaf_blight', 'Peach_Bacterial_spot', 'Peach_Healthy', 'Pepper_Bacterial_spot', 'Pepper_Healthy',
            'Potato_Early_blight', 'Potato_Healthy', 'Potato_Late_blight', 'Raspberry_Healthy', 'Soybean_Healthy',
            'Squash_Powdery_mildew', 'Strawberry_Healthy', 'Strawberry_Leaf_scorch', 'Tomato_Bacterial_spot',
            'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_mold', 'Tomato_Mosaic_virus',
            'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_spot', 'Tomato_Yellow_leaf_curl_virus'
        ],
        "data_dtype": ['jpg']
    },
    "PDc": {        # 27 class
        "path": "/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture/PlantDoc",
        "classes": [
            'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Blueberry_Healthy', 'Cherry_Healthy',
            'Corn_Common_rust', 'Corn_Gray_leaf_spot', 'Corn_Northern_leaf_blight', 'Grape_Black_rot', 'Grape_Healthy',
            'Peach_Healthy', 'Pepper_Bacterial_spot', 'Pepper_Healthy', 'Potato_Early_blight', 'Potato_Late_blight',
            'Raspberry_Healthy', 'Soybean_Healthy', 'Squash_Powdery_mildew', 'Strawberry_Healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_mold',
            'Tomato_Mosaic_virus', 'Tomato_Septoria_leaf_spot', 'Tomato_Yellow_leaf_curl_virus'
        ],
        "data_dtype": ['jpg']
    },
    "AIC": {        # 37 class
        "path": "/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture/AI Challenger",
        "classes": [
            'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Cherry_Healthy',
            'Cherry_Powdery_mildew', 'Citrus_Greening', 'Citrus_Healthy', 'Corn_Common_rust', 'Corn_Gray_leaf_spot',
            'Corn_Healthy', 'Corn_Northern_leaf_blight', 'Grape_Black_measles', 'Grape_Black_rot', 'Grape_Healthy',
            'Grape_Leaf_blight', 'Peach_Bacterial_spot', 'Peach_Healthy', 'Pepper_Bacterial_spot', 'Pepper_Healthy',
            'Potato_Early_blight', 'Potato_Healthy', 'Potato_Late_blight', 'Soybean_Healthy', 'Squash_Powdery_mildew',
            'Strawberry_Healthy', 'Strawberry_Leaf_scorch', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
            'Tomato_Healthy',  'Tomato_Late_blight', 'Tomato_Leaf_mold', 'Tomato_Mosaic_virus',
            'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_spot', 'Tomato_Yellow_leaf_curl_virus'
        ],
        "data_dtype": ['jpg']
    },
    "PDs": {        # 27 class
        "path": "/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/Agriculture/PlantDiseases",
        "classes": [
            'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Blueberry_Healthy', 'Cherry_Healthy',
            'Corn_Common_rust', 'Corn_Gray_leaf_spot', 'Corn_Northern_leaf_blight', 'Grape_Black_rot', 'Grape_Healthy',
            'Peach_Healthy', 'Pepper_Bacterial_spot', 'Pepper_Healthy', 'Potato_Early_blight', 'Potato_Late_blight',
            'Raspberry_Healthy', 'Soybean_Healthy', 'Squash_Powdery_mildew', 'Strawberry_Healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_mold',
            'Tomato_Mosaic_virus', 'Tomato_Septoria_leaf_spot', 'Tomato_Yellow_leaf_curl_virus'
        ],
        "data_dtype": ['jpg']
    },
}



# PVi-PDc 公共类 27
PVi_PDc_common_classes = {
    'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Blueberry_Healthy', 'Cherry_Healthy',
    'Corn_Common_rust', 'Corn_Gray_leaf_spot', 'Corn_Northern_leaf_blight', 'Grape_Black_rot', 'Grape_Healthy',
    'Peach_Healthy', 'Pepper_Bacterial_spot', 'Pepper_Healthy', 'Potato_Early_blight', 'Potato_Late_blight',
    'Raspberry_Healthy', 'Soybean_Healthy', 'Squash_Powdery_mildew', 'Strawberry_Healthy', 'Tomato_Bacterial_spot',
    'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_mold', 'Tomato_Mosaic_virus',
    'Tomato_Septoria_leaf_spot', 'Tomato_Yellow_leaf_curl_virus'
}

# AIC-PDc 公共类 25
AIC_PDc_common_classes = {
    'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Cherry_Healthy', 'Corn_Common_rust',
    'Corn_Gray_leaf_spot', 'Corn_Northern_leaf_blight', 'Grape_Black_rot', 'Grape_Healthy', 'Peach_Healthy',
    'Pepper_Bacterial_spot', 'Pepper_Healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Soybean_Healthy',
    'Squash_Powdery_mildew', 'Strawberry_Healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy',
    'Tomato_Late_blight', 'Tomato_Leaf_mold', 'Tomato_Mosaic_virus', 'Tomato_Septoria_leaf_spot',
    'Tomato_Yellow_leaf_curl_virus'
}


# AIC-PVi 公共类 36
PVi_AIC_common_classes = {
    'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Cherry_Healthy',
    'Cherry_Powdery_mildew', 'Citrus_Greening', 'Corn_Common_rust', 'Corn_Gray_leaf_spot', 'Corn_Healthy',
    'Corn_Northern_leaf_blight', 'Grape_Black_measles', 'Grape_Black_rot', 'Grape_Healthy', 'Grape_Leaf_blight',
    'Peach_Bacterial_spot', 'Peach_Healthy', 'Pepper_Bacterial_spot', 'Pepper_Healthy', 'Potato_Early_blight',
    'Potato_Healthy', 'Potato_Late_blight', 'Soybean_Healthy', 'Squash_Powdery_mildew', 'Strawberry_Healthy',
    'Strawberry_Leaf_scorch', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight',
    'Tomato_Leaf_mold', 'Tomato_Mosaic_virus', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_spot',
    'Tomato_Yellow_leaf_curl_virus'
}


# PDc-PDs 公共类 27
PDc_PDs_common_classes = {
    'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Blueberry_Healthy', 'Cherry_Healthy',
    'Corn_Common_rust', 'Corn_Gray_leaf_spot', 'Corn_Northern_leaf_blight', 'Grape_Black_rot', 'Grape_Healthy',
    'Peach_Healthy', 'Pepper_Bacterial_spot', 'Pepper_Healthy', 'Potato_Early_blight', 'Potato_Late_blight',
    'Raspberry_Healthy', 'Soybean_Healthy', 'Squash_Powdery_mildew', 'Strawberry_Healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_mold',
    'Tomato_Mosaic_virus', 'Tomato_Septoria_leaf_spot', 'Tomato_Yellow_leaf_curl_virus'
}


# PVi-PP 公共类 3  Plant Pathology
PVi_PP_common_classes = {'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab'}

# AIC-PP 公共类 3  Plant Pathology
AIC_PP_common_classes = {'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab'}


# 39
All_classes = {
    'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_Healthy', 'Apple_Scab', 'Blueberry_Healthy',
    'Cherry_Healthy', 'Cherry_Powdery_mildew', 'Citrus_Greening', 'Citrus_Healthy', 'Corn_Common_rust',
    'Corn_Gray_leaf_spot', 'Corn_Healthy', 'Corn_Northern_leaf_blight', 'Grape_Black_measles', 'Grape_Black_rot',
    'Grape_Healthy', 'Grape_Leaf_blight', 'Peach_Bacterial_spot', 'Peach_Healthy', 'Pepper_Bacterial_spot',
    'Pepper_Healthy', 'Potato_Early_blight', 'Potato_Healthy', 'Potato_Late_blight', 'Raspberry_Healthy',
    'Soybean_Healthy', 'Squash_Powdery_mildew', 'Strawberry_Healthy', 'Strawberry_Leaf_scorch',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_mold',
    'Tomato_Mosaic_virus', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_spot',
    'Tomato_Yellow_leaf_curl_virus'
}


