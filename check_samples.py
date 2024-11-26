import os
mass = [300, 400, 500, 1500, 2000, 3000]

path = "/eos/user/a/avijay/HZZ_mergedrootfiles/"

ggh = "GluGluHToZZTo2L2Nu_M{mass}_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8.root"
vbf = "VBF_HToZZTo2L2Nu_M{mass}_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8.root"

for m in mass:
    # check if file exists
    if os.path.exists(path + ggh.format(mass=m)):
        print("File exists")
    else:
        print("File does not exist")
        print(ggh.format(mass=m))

    if os.path.exists(path + vbf.format(mass=m)):
        print("File exists")
    else:
        print("File does not exist")
        print(vbf.format(mass=m))
