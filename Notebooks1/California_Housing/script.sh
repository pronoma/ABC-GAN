#Run Baseline Models  
for runs in 1 2 3 4 5 6 7 8 9 10
do 
    papermill BaselineModels.ipynb ./BaseLine_Model_Output/BaselineModels_output_${runs}.ipynb

done 

# #GAN 
# # for runs in 1 2 3 4 5 6 7 8 9 10
# # do
# #     papermill GAN.ipynb ./GAN_Output/GAN_output_${runs}_${i}.ipynb  
# # done

# # #ABC-GAN - Stats  
# for runs in 1 2 3 4 5 6 7 8 9 10
# do
#     i=0
#     for V in 1 0.1 0.01 
#     do 
#         papermill ABC_GAN-Stats.ipynb ./ABC_GAN_Stats/ABC-GAN_output_${runs}_${i}.ipynb -p variance ${V} 
#         ((i=i+1))
#     done 
# done

# # #ABC-GAN - Catboost 
# for runs in 1 2 3 4 5 6 7 8 9 10
# do
#     i=0
#     for V in 1 0.1 0.01 
#     do 
#         papermill ABC_GAN-Catboost.ipynb ./ABC_GAN_Catboost/ABC-GAN_output_${runs}_${i}.ipynb -p variance ${V} 
#         ((i=i+1))
#     done 
# done

# # #Analysis 
# papermill Analysis.ipynb California.ipynb 
# jupyter nbconvert California.ipynb --to pdf

