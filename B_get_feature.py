# -*- coding:utf-8 -*-
from B_feature_relation_function import *

if __name__ == '__main__':
    #提取数据特征
    '''
    img_dir = '../data/200.jpg'
    img_feature_shuffleNetV2 = get_img_feature_shuffleNetV2_model(img_dir)
    img_feature_vgg16 = get_img_feature_vgg16(img_dir)
    img_feature_resnet18 = get_img_feature_resnet18(img_dir)
    print(len(img_feature_shuffleNetV2),len(img_feature_vgg16),len(img_feature_resnet18))
    '''
    for filepath in ['./mydataset_paths.txt']:#,'./UCMerced_LandUse_paths.txt'
        #filepath = './Holidays_paths.txt'
        name = filepath.split('/')[1]
        Datasets_name = name.split('_')[0]

        Result = get_Datasets_feature(filepath)
        [all_feature_shuffleNetV2,all_feature_vgg16,all_feature_resnet18,all_feature_densenet121,
        all_feature_Alexnet,all_feature_convnext,all_feature_squeezenet,
        all_feature_googlenet,all_feature_efficientnet,all_feature_mnasnet,
        all_feature_regnet,all_feature_mobilenet,all_feature_color,all_feature_lbp,
        all_feature_GIST,all_feature_Harris,all_feature_HOG] = Result


        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_shuffleNetV2'.split('_')[2]+'.npy', all_feature_shuffleNetV2)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_vgg16'.split('_')[2]+'.npy', all_feature_vgg16)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_resnet18'.split('_')[2]+'.npy', all_feature_resnet18)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_densenet121'.split('_')[2]+'.npy', all_feature_densenet121)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_Alexnet'.split('_')[2]+'.npy', all_feature_Alexnet)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_convnext'.split('_')[2]+'.npy', all_feature_convnext)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_squeezenet'.split('_')[2]+'.npy', all_feature_squeezenet)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_googlenet'.split('_')[2]+'.npy', all_feature_googlenet)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_efficientnet'.split('_')[2]+'.npy', all_feature_efficientnet)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_mnasnet'.split('_')[2]+'.npy', all_feature_mnasnet)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_regnet'.split('_')[2]+'.npy', all_feature_regnet)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_mobilenet'.split('_')[2]+'.npy', all_feature_mobilenet)

        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_color'.split('_')[2]+'.npy', all_feature_color)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_lbp'.split('_')[2]+'.npy', all_feature_lbp)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_GIST'.split('_')[2]+'.npy', all_feature_GIST)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_Harris'.split('_')[2]+'.npy', all_feature_Harris)
        np.save('./features/'+Datasets_name+'_feature_'+'all_feature_HOG'.split('_')[2]+'.npy', all_feature_HOG)
 