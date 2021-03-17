import sys
import os
import optparse, json, argparse
import numpy as np
import pandas as pd
import ROOT
sys.path.insert(0, '/afs/cern.ch/work/j/jthomasw/private/IHEP/HH/HHWWyy/all_had/')
from application.apply_DNN import apply_DNN
from root_numpy import tree2array
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.image import NonUniformImage

def main():
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-p', '--processName', dest='processName', help='Process name.', default=[], type=str, nargs='+')
    args = parser.parse_args()
    processes = args.processName

    process_filename = {
    'HHWWgg' : ('GluGluToHHTo2G4Q_node_cHHH1_2018'),
    'DYJetsToLL_M-50': ('DYJetsToLL_M-50_TuneCP5_13TeV'),
    'WW_TuneCP5':('WW_TuneCP5_13TeV-pythia8'),
    'DiPhotonJetsBox_M40_80': ('DiPhotonJetsBox_M40_80'),
    'DiPhoton_80plus':  ('DiPhotonJetsBox_MGG-80toInf_13TeV'),
    'QCD_Pt-30toInf':('QCD_Pt-30toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_13TeV'),
    'QCD_Pt-30to40':('QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV'),
    'QCD_Pt-40toInf':('QCD_Pt-40toInf_DoubleEMEnriched_MGG-80toInf'),
    'GJet_Pt-20to40' : ('GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV'),
    'GJet_Pt-40toInf' : ('GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf') ,
    'TTGJets' : ('TTGJets_TuneCP5_13TeV'),
    'TTGG' : ('TTGG_0Jets_TuneCP5_13TeV'),
    'TTplusJets' : ('TTJets_TuneCP5_13TeV'),
    'ttHJetToGG' : ('ttHJetToGG_M125_13TeV'),
    'VBFHToGG':('VBFHToGG_M125_13TeV'),
    'GluGluHToGG':('GluGluHToGG_M125_TuneCP5_13TeV'),
    'VHToGG':('VHToGG_M125_13TeV')
    }

    for process in processes:
        print('Running process: ', process)
        inputs_file_path = '/eos/user/r/rasharma/post_doc_ihep/double-higgs/ntuples/January_2021_Production/DNN/samples_w_DNN/HHWWyyDNN_binary_10March_SHiggs_BalanceYields/'
        print('Getting input files from directory: ', inputs_file_path)
        current_sample_name = process_filename.get(process)

        # Open and load input variable .json
        input_var_jsonFile = ''
        input_var_jsonFile = open('input_variables.json','r')
        variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()
        column_headers = []
        for key,var in variable_list:
            column_headers.append(key)
        column_headers.append('DNN_evaluation')
        column_headers.append('event')
        column_headers.append('weight')


        outputdataframe_name = 'dataframes/output_dataframe_%s.csv' %(current_sample_name)
        if os.path.isfile(outputdataframe_name):
            """Load dataset or not

            If one changes the input training variables then we have to reload dataset.
            Don't use the previous .csv file if you update the list of input variables.
            """
            print('Loading data .csv from: %s . . . . ' % (outputdataframe_name))
            data = pd.read_csv(outputdataframe_name)
        else:
            print('Making new data .csv from: %s . . . . ' % (outputdataframe_name))
            data = load_data(inputs_file_path,column_headers,'',current_sample_name)
            data = data.mask(data<-25., -9.)
            data.to_csv(outputdataframe_name, index=False)
            data = pd.read_csv(outputdataframe_name)

        #for poi,var in variable_list[:1]:
        for poi,var in variable_list:
            DNN_output_values = data['DNN_evaluation']
            poi_input_values = data[poi]
            fig, ax1 = plt.subplots(ncols=1, figsize=(10,10))

            #ax1.plot(DNN_output_values, poi_input_values, '.', markersize=3)
            plt.xlabel('DNN output',fontsize=25)
            if 'goodJets_0_bDiscriminator' in poi:
                poi = 'jet_0_bDisc'
            if 'goodJets_1_bDiscriminator' in poi:
                poi = 'jet_1_bDisc'
            if 'goodJets_2_bDiscriminator' in poi:
                poi = 'jet_2_bDisc'
            if 'goodJets_3_bDiscriminator' in poi:
                poi = 'jet_3_bDisc'

            y_axis_upper = max(poi_input_values)
            y_axis_lower = min(poi_input_values)
            x_axis_upper = max(DNN_output_values)
            x_axis_lower = min(DNN_output_values)
            #print(poi)
            if 'Candidate_M' in poi:
                y_axis_upper = 2000.
            if 'Candidate_pt'in poi:
                y_axis_upper = 400.
            if 'Subleading_Jet_pt'in poi:
                y_axis_upper = 50.
            if 'Leading_Jet_pt'in poi:
                y_axis_upper = 300.
            if 'Leading_Photon_E/CMS_hgg_mass'in poi:
                y_axis_upper = 4.
            if 'Subleading_Photon_E/CMS_hgg_mass'in poi:
                y_axis_upper = 2.
            if 'Leading_Photon_pt/CMS_hgg_mass'in poi:
                y_axis_upper = 3.
            if 'Subleading_Photon_pt/CMS_hgg_mass'in poi:
                y_axis_upper = 1.
            if 'Photon_MVA'in poi:
                y_axis_upper = 1.
                y_axis_lower = 0.8
            if 'Photon_initE'in poi:
                y_axis_upper = 200.
            if 'Leading_Photon_r9'in poi:
                y_axis_upper = 1.
                y_axis_lower = 0.9
            if 'Subleading_Photon_r9'in poi:
                y_axis_upper = 1.
                y_axis_lower = 0.7
            if 'Photon_full5x5_r9'in poi:
                y_axis_upper = 1.
                y_axis_lower = 0.9
            if 'cosThetaHH'in poi:
                y_axis_upper = 1.
                y_axis_lower = 0.8
            if 'cosThetaWW'in poi:
                y_axis_upper = 1.
                y_axis_lower = 0.8
            if '_bDisc'in poi:
                y_axis_upper = 0.1
                y_axis_lower = 0.
            if 'N_goodJets'in poi:
                y_axis_upper = 8.

            axis_range = [[x_axis_lower,x_axis_upper],[y_axis_lower,y_axis_upper]]

            # Create heat map smeared using a Gaussian kernel
            img, extent = myplot(DNN_output_values, poi_input_values, 8, axis_range)
            corr_heatmap = plt.imshow(img, extent=extent, origin='lower', cmap=cm.coolwarm, aspect='auto')

            ax1.set_title("Correlation w. "+poi,fontsize=25)
            plt.ylabel(poi,fontsize=25)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            cb = plt.colorbar(corr_heatmap)
            cb.set_label('Number of entries')
            #if '/' in process:
            #process = process.replace('/','_over_')
            poi = poi.replace('/','_over_')
            #ax1.set_title("Input var. DNN Correlation"+poi,fontsize=25)
            fig.savefig('correlation_plots/%s_corr_DNN_%s.png' %(process,poi))
            plt.close()

        #print(DNN_output_values)
    exit(0)

def myplot(x, y, s, axis_range, bins=50):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[bins,bins], range=axis_range)
    #heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

def load_data(inputPath, variables, criteria, process):
    my_cols_list=variables
    data = pd.DataFrame(columns=my_cols_list)
    if 'GluGluToHHTo2G4Q_node_cHHH1_2018' in process:
        sampleNames=process
        fileNames = [process]
        target=1
    else:
        sampleNames=process
        fileNames = [process]
        target=0

    treename=['output_tree']
    filename_fullpath = inputPath+"/"+process+".root"
    print("Input file: ", filename_fullpath)
    print("my_cols_list: \n",my_cols_list)
    #print("my_cols_list[:-1]: \n",my_cols_list[:-1])
    print("criteria: ", criteria)
    tfile = ROOT.TFile(filename_fullpath)
    for tname in treename:
        print('TTree: ', tname)
        ch_0 = tfile.Get(tname)
        if ch_0 is not None :
            print('# Events in original sample: ' , ch_0.GetEntries())
            chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-1], selection=criteria)
            chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list)
            chunk_df['key']=process
            chunk_df['target']=target
            chunk_df['weight']=chunk_df["weight"]
            # =============== Weights ==================
            # WARNING! 'sample_weight' will overide 'class_weight'
            # ==========================================
            if sampleNames=='HH':
                # Reweight classes
                chunk_df['classbalance'] = 1.0
            if sampleNames=='bckg':
                chunk_df['classbalance'] = 1.0
            data = data.append(chunk_df, ignore_index=True)
            #print('Dataframe info = ', data.info() )
        else:
            print("TTree == None")
        ch_0.Delete()
    tfile.Close()
    return data

main()
