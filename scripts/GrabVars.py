# -*- coding: utf-8 -*-
# @Author: Ram Krishna Sharma
# @Date:   2021-04-13
# @Last Modified by:   Ram Krishna Sharma
# @Last Modified time: 2021-04-13
#
# Leading_Photon_MVA,Subleading_Photon_MVA,Leading_Photon_SC_eta,Leading_Photon_r9,Leading_Photon_passElectronVeto,Leading_Photon_hasPixelSeed,Subleading_Photon_r9,Subleading_Photon_passElectronVeto,Subleading_Photon_hasPixelSeed,Leading_Photon_E,Leading_Photon_pt,Leading_Photon_eta,Leading_Photon_phi,Subleading_Photon_E,Subleading_Photon_pt,Subleading_Photon_eta,Subleading_Photon_phi,PhotonID_min,PhotonID_max,New_Leading_Jet_E,New_Leading_Jet_pt,New_Leading_Jet_px,New_Leading_Jet_py,New_Leading_Jet_pz,New_Leading_Jet_eta,New_Leading_Jet_phi,New_Subleading_Jet_E,New_Subleading_Jet_pt,New_Subleading_Jet_px,New_Subleading_Jet_py,New_Subleading_Jet_pz,New_Subleading_Jet_eta,New_Subleading_Jet_phi,New_Sub2leading_Jet_E,New_Sub2leading_Jet_pt,New_Sub2leading_Jet_px,New_Sub2leading_Jet_py,New_Sub2leading_Jet_pz,New_Sub2leading_Jet_eta,New_Sub2leading_Jet_phi,New_Sub3leading_Jet_E,New_Sub3leading_Jet_pt,New_Sub3leading_Jet_px,New_Sub3leading_Jet_py,New_Sub3leading_Jet_pz,New_Sub3leading_Jet_eta,New_Sub3leading_Jet_phi,New_OnShellW_LeadingJet_bDis,New_OnShellW_SubLeadingJet_bDis,New_OffShellW_LeadingJet_bDis,New_OffShellW_SubLeadingJet_bDis,New_OnShellW_E,New_OnShellW_Mass,New_OnShellW_pt,New_OnShellW_px,New_OnShellW_py,New_OnShellW_pz,New_OnShellW_eta,New_OnShellW_phi,New_OffShellW_E,New_OffShellW_Mass,New_OffShellW_pt,New_OffShellW_px,New_OffShellW_py,New_OffShellW_pz,New_OffShellW_eta,New_OffShellW_phi,New_HWW_E,New_HWW_Mass,New_HWW_pt,New_HWW_px,New_HWW_py,New_HWW_pz,New_HWW_eta,New_HWW_phi,New_dR_Hgg_Jet1,New_dR_Hgg_Jet2,New_dR_Hgg_Jet3,New_dR_Hgg_Jet4,New_dPhi_Hgg_Jet1,New_dPhi_Hgg_Jet2,New_dPhi_Hgg_Jet3,New_dPhi_Hgg_Jet4,New_DPhi_gg,New_DR_gg,New_DPhi_HH,New_DR_HH,New_minDeltaR_gg4j,New_maxDeltaR_gg4j,New_minDeltaR_4j,New_maxDeltaR_4j,a_costheta1,a_costheta2,a_costhetastar,a_Phi,a_Phi1,CosThetaStar_CS,CosThetaStar_CS_old,HelicityCostheta1,HelicityCostheta2,weight,unweighted,target,key,classweight,process_ID

FileList = {
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_BalanceYields": "All.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar1_BalanceYields": "SelVar1.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar2_BalanceYields": "SelVar2.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar2DeCorrMgg_BalanceYields": "SelVar2DeCorrMgg.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar3_BalanceYields": "SelVar3.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar3DeCorrMgg_BalanceYields": "SelVar3DeCorrMgg.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar4DeCorrMgg_BalanceYields": "SelVar4DeCorrMgg.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar5DeCorrMgg_BalanceYields": "SelVar5DeCorrMgg.json",
    "HHWWyyDNN_binary_Apr12_VarVsROC_E300_LR10em4_B200_Adam_SelVar6DeCorrMgg_BalanceYields": "SelVar6DeCorrMgg.json"
}



# for j,dirList in enumerate(FileList):
for key in FileList:
    print key,FileList[key]
    with open(key+'/output_dataframe.csv',"r") as f:
        first_line = f.readline().strip()


    first_line = first_line.split(",")

    # print(first_line)

    first_line = first_line[:len(first_line)-6]
    # print(first_line)

    #  "Leading_Photon_hasPixelSeed": "\"F\"",

    file1 = open(FileList[key],"w")
    file1.write('{\n')
    lengthOfList = len(first_line)
    for i,vars in enumerate(first_line):
        if (i==lengthOfList-1):
            file1.write('\t"'+vars+'": '+ '"\\"F\\""\n')
        else:
            file1.write('\t"'+vars+'": '+ '"\\"F\\"",\n')
    file1.write('}\n')
