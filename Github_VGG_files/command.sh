


### 03/2023 Please modify the command below according to your settings if necessary
python main_10fold.py --save-dir=./result/model/save_BrainGlu_T1_original_MN152Affine_WB_3T_DSx2_multiscale_stride16_Batchx5_T1mean_lr1e-5_20221110_811_V-T_fold2-9_Adam --batch-size=3 --lr=1e-5 --cuda-idx=0 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=std --val-folder='fold2' --test-folder='fold9' |& tee -a ./result/log/log_save_Multiscale_T1_MN152Affine_WB_3T_DSx2_patch_32x32x32_stride16_Batchx5_T1mean_lr1e-5_20211107_811_V-T_fold2-9_Adam














