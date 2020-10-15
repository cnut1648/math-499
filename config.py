EPOCHS = 60
BATCH_SIZE = 32
ensemble_num = 15

train_for_all = True

# dir that contains label csv
# eg. label_dir/tranch1_labels.csv
label_dir = "/content/drive/My Drive/M499/" 


# dir that contains images folders,
# imgs from tranch t must be in subfolder img_dir/tranch<t>
# imgs for all tranchs must be in subfolder img_dir/allTranch
# note that after unzip the inner directory hierarchy have to be flatten    
img_dir = "/content/drive/My Drive/M499/" 
                                          
# base dir to which model would be saved
# saved model checkpoints in save_dir/<tranch>/<model type>-<ensemble num>.<epoch>-<val acc>.h5
# saved log in save_dir/<tranch>/<model type>-<ensemble num>.log
# saved plot in save_dir/<tranch>/<model type>-<ensemble num>.jpg
# final saved model in save_dir/<tranch>/<model type>-<ensemble num>:::final--<day>:<hour>.h5
# saved model for all tranchs in save_dir/all/....
save_dir = "/content/drive/My Drive/model" 