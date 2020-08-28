Basic running:

python train-BinaryDNN.py -t <0 or 1> -s <suffix_for_output_dir>

0 = make plots from pre-trained model in training directory.
1 = train new model from scratch.

Code should be fairly well commented.

Most code is in the python script above or in 'plotting' package.
Inside train-BinaryDNN.py:
'load_data' function will make a new dataframe from the input ntuples if one doesn't already exists in the training 
directory. If you want to make a new dataset in the same directory, you will need to delete the old 
output_dataframe.csv before reruning the script.
'classbalance' variable sets the balance and therefore the focus of the DNN between background(s) and signal. Be 
careful, I have currently set this so signal weights == 1 and background is scaled to the same effective number of 
events. You will need to change this for your specific case. 

'baseline_model' function is where the network architecture is defined. You should be able to see quite easily how 
thhe hyperparameters can be changed.

Inside the main function, the selection used on the fly to create the dataframe is defined by the variable 'selection_criteria'
The input data is defined by 'inputs_file_path'.

The rest of the model is defined in the models '.fit' function.

Most of the rest is fairly straightforward if you read the comments.
