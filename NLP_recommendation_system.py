import pandas as pd
import numpy as np
import sys
import os
import operator
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from string import digits
from gensim import corpora
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.lancaster import LancasterStemmer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

def getFileSize(path,file):
    fileStats = os.stat(path+file)
    print(str(round(fileStats.st_size/(1024*1024*1024),2))+' GB')

def lineGenerator(path,file):
    """ This Function is to create a generator to read a large file line by line from the raw file rather than reading whole file.
    This function yields one line at a time"""
    f = open(path+file,'rb')
    while True:
        line = f.readline()
        if not line:
            break
        yield line

def readFileInToList(path,file,recList,n):
    """This Function is to read given number of lines from raw file. Lines are read using generator"""
    retrieveLine = lineGenerator(path,file)
    try:
      while len(recList)<n:
        line = next(retrieveLine).decode(errors='ignore')
        if not line:
            break
        else:
            recList.append(line)
    except StopIteration:
        print('Total Lines read from the raw file :',len(recList))

def readNRowsFileInToList(path,file,recList,startLine,endLine):
    """This Function is to read given number of lines from raw file.It has option to specify start line and end line Lines are read using generator"""
    retrieveLine = lineGenerator(path,file)
    count=0
    while count<=endLine:
        if count<startLine:
            next(retrieveLine)
            count+=1
        else:
            try:
                line = next(retrieveLine).decode(errors='ignore')
                if not line:
                     break
                else:
                     recList.append(line)
                     count+=1
            except StopIteration:
                print('Total Lines read from the raw file :',len(recList))
    print('Total Lines read from the raw file :',len(recList))

def getUniqueCustIDs(path,file,custID,cc):
    """ This function is to return a set of unique customer IDs which is to be used in further
    Data cleaning process"""
    retrieveLine = lineGenerator(path,file)
    try:
        while True:
            line = next(retrieveLine).decode(errors='ignore')
            if not line:
                break
            else:
                ID = line[0:line.find('\t')]
                try:
                    cC = line.split('\t')[1] in cc
                except IndexError:
                    cC=False
                if ID.isdigit() & cC:
                    custID.append(int(ID))
    except StopIteration:
        print('All lines are read. No more lines to read from file.....')
    print('Total unique customers in the file :',len(set(custID)))
    return set(custID)

def fixNewLineIssue(records,custID,cc):
    """There are two issues with the dataset
    1. New lines in Job Summary are treated as new records which is incorrect
    2. Number of Tabs in each line varies because of presence of additional tabs in Job Summary
    This function is to fix issue 1 and form a complete record
    New Line is identified by the presence of customer ID at the start of the line. If the start of the line
    does not contain customer ID, then it is identified as part of previous line that has customer ID at the
    start
    """
    # Identify lines that does not contain customer ID at the start
    startingCustID = []
    for i in records:
        try:
            if(int(i[0:i.find('\t')]) in custID) & (i.split('\t')[1] in cc):
                startingCustID.append(True)
            else:
                startingCustID.append(False)
        except ValueError:
            startingCustID.append(False)
        except IndexError:
            startingCustID.append(False)
    # Combine the split up lines to a single valid line
    for i in np.arange(0,len(records)):
        moveBack=0;
        if startingCustID[i]==False:
            for r in np.arange(i,0,-1):
                if startingCustID[r]:
                    break
                else:
                    moveBack+=1
            records[i-moveBack] = records[i-moveBack]+' '+records[i]
        else:
            pass
    records = [records[i] for i in np.arange(1,len(records)) if startingCustID[i]]
    return(records)

def fixTabIssue(records):
    """ There are 78 columns in the dataset which means presence of 77 tabs.
    """
    records_77 = list(filter(lambda x:x.count('\t')==77,records))
    records_fix = list(filter(lambda x:x.count('\t')>77,records))
    # there might be few last or first records where number of fields might be incorrect due to availability of
    # only half data. Other half will be next batch which is yet to be processed
    # It should be processed in next batch of records
    last_record = list(filter(lambda x:x.count('\t')<77,records))
    for i,j in enumerate(records_fix):
        tabIndex = pd.Series([val.start() for val in re.finditer('\t',j)])
        startVal = tabIndex.sort_values(ascending=True)[15]
        lastVal = tabIndex.sort_values(ascending=False).reset_index(drop=True)[60]
        records_fix[i]=j[:startVal+1]+j[startVal+1:lastVal+1].replace('\t','')+'\t'+j[lastVal+1:]
        records_fix[i].replace('\n','')
    records=records_77+records_fix
    return(records)

def dfRecords(records,n_rec):
    for i in records:
        yield i.split('\t')

def listToDataFrame(records,headers):
    # merging by 100k records at a time to avoid memory error
    merge_by = 100000
    n = len(records)//merge_by
    df = pd.DataFrame([])
    for i in np.arange(n):
        df = pd.concat([df,pd.DataFrame([i.split('\t') for i in records[i*merge_by:i*merge_by+merge_by]])])

    # appending last remaining records
    df = pd.concat([df,pd.DataFrame([i.split('\t') for i in records[n*merge_by:len(records)]])])
    df.columns = headers.split('\t')
    return(df)

def topicmodel(dataframe):
    doc_complete = dataframe['JobSummary'].values.T.tolist()
    doc_clean = [clean(doc).split() for doc in doc_complete]
    return doc_clean

def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stemmer = LancasterStemmer()
    if type(doc) != str:
        doc=doc.to_string()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stemmed = " ".join(stemmer.stem(word) for word in normalized.split())
    texted = " ".join(word.translate(str.maketrans('','','1234567890')) for word in stemmed.split())
    return texted

def word_feats(words):
    return dict([(word, True) for word in words])

def MRARestagger(doc_clean):
    #kitchen
    refrigerator_repairs = clean('Refrigerator Repairs').split()
    refrigerator_notcold = clean('Refrigerator Not Cold Enough').split()
    freezer_repairs = clean('Freezer Repairs').split()
    dishwash_repairs = clean('Dishwasher Repairs').split()
    oven = clean('Ovens, Stove Tops & Ranges').split()
    ice = clean('Ice Machine Repairs').split()
    garbage = clean('Garbage Disposal Repairs').split()
    microwave = clean('Microwave Oven Repairs').split()
    vent = clean('Vent Hoods').split()
    wine = clean('Wine Coolers').split()
    trash = clean('Trash Compactors').split()
    out = clean('Outdoor Kitchens').split()

    #laundry
    wash = clean('Washing Machine Repair').split()
    dry = clean('Dryer Repair').split()
    dryclean = clean('Dryer Vent Cleaning').split()

    #Residential Appliance Parts
    respart = clean('Residential Appliance Parts').split()

    RefrigeratorRepairs_features = [(word_feats(RefrigeratorRepairs), 'RefrigeratorRepairs') for RefrigeratorRepairs in refrigerator_repairs]
    RefrigeratorNotColdEnough_features = [(word_feats(RefrigeratorNotColdEnough), 'RefrigeratorNotColdEnough') for RefrigeratorNotColdEnough in refrigerator_notcold]
    FreezerRepairs_features = [(word_feats(FreezerRepairs), 'FreezerRepairs') for FreezerRepairs in freezer_repairs]
    DishwasherRepairs_features = [(word_feats(DishwasherRepairs), 'DishwasherRepairs') for DishwasherRepairs in dishwash_repairs]
    OvensStoveTopsRanges_features = [(word_feats(OvensStoveTopsRanges), 'OvensStoveTopsRanges') for OvensStoveTopsRanges in oven]
    IceMachineRepairs_features = [(word_feats(IceMachineRepairs), 'IceMachineRepairs') for IceMachineRepairs in ice]
    GarbageDisposalRepairs_features = [(word_feats(GarbageDisposalRepairs), 'GarbageDisposalRepairs') for GarbageDisposalRepairs in garbage]
    MicrowaveOvenRepairs_features = [(word_feats(MicrowaveOvenRepairs), 'MicrowaveOvenRepairs') for MicrowaveOvenRepairs in microwave]
    VentHoods_features = [(word_feats(VentHoods), 'VentHoods') for VentHoods in vent]
    WineCoolers_features = [(word_feats(WineCoolers), 'WineCoolers') for WineCoolers in wine]
    TrashCompactors_features = [(word_feats(TrashCompactors), 'TrashCompactors') for TrashCompactors in trash]
    OutdoorKitchens_features = [(word_feats(OutdoorKitchens), 'OutdoorKitchens') for OutdoorKitchens in out]

    WashingMachineRepair_features = [(word_feats(WashingMachineRepair), 'WashingMachineRepair') for WashingMachineRepair in wash]
    DryerRepair_features = [(word_feats(DryerRepair), 'DryerRepair') for DryerRepair in dry]
    DryerVentCleaning_features = [(word_feats(DryerVentCleaning), 'DryerVentCleaning') for DryerVentCleaning in dryclean]

    ResidentialApplianceParts_features = [(word_feats(ResidentialApplianceParts), 'ResidentialApplianceParts') for ResidentialApplianceParts in respart]

    train_set = RefrigeratorRepairs_features +RefrigeratorNotColdEnough_features +FreezerRepairs_features +DishwasherRepairs_features +OvensStoveTopsRanges_features +IceMachineRepairs_features +GarbageDisposalRepairs_features +MicrowaveOvenRepairs_features +VentHoods_features +WineCoolers_features +TrashCompactors_features +OutdoorKitchens_features +WashingMachineRepair_features +DryerRepair_features +DryerVentCleaning_features +ResidentialApplianceParts_features

    classifier = NaiveBayesClassifier.train(train_set)
    # Predict
    RefrigeratorRepairs=0
    RefrigeratorNotColdEnough=0
    FreezerRepairs=0
    DishwasherRepairs=0
    OvensStoveTopsRanges=0
    IceMachineRepairs=0
    GarbageDisposalRepairs=0
    MicrowaveOvenRepairs=0
    VentHoods=0
    WineCoolers=0
    TrashCompactors=0
    OutdoorKitchens=0
    WashingMachineRepair=0
    DryerRepair=0
    DryerVentCleaning=0
    ResidentialApplianceParts=0

    words = clean(doc_clean).split()
    for word in words:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'RefrigeratorRepairs':
            RefrigeratorRepairs = RefrigeratorRepairs + 1
        if classResult == 'RefrigeratorNotColdEnough':
            RefrigeratorNotColdEnough = RefrigeratorNotColdEnough + 1
        if classResult == 'FreezerRepairs':
            FreezerRepairs = FreezerRepairs + 1
        if classResult == 'DishwasherRepairs':
            DishwasherRepairs = DishwasherRepairs + 1
        if classResult == 'OvensStoveTopsRanges':
            OvensStoveTopsRanges = OvensStoveTopsRanges + 1
        if classResult == 'IceMachineRepairs':
            IceMachineRepairs = IceMachineRepairs + 1
        if classResult == 'GarbageDisposalRepairs':
            GarbageDisposalRepairs = GarbageDisposalRepairs + 1
        if classResult == 'MicrowaveOvenRepairs':
            MicrowaveOvenRepairs = MicrowaveOvenRepairs + 1
        if classResult == 'VentHoods':
            VentHoods = VentHoods + 1
        if classResult == 'WineCoolers':
            WineCoolers = WineCoolers + 1
        if classResult == 'TrashCompactors':
            TrashCompactors = TrashCompactors + 1
        if classResult == 'OutdoorKitchens':
            OutdoorKitchens = OutdoorKitchens + 1
        if classResult == 'WashingMachineRepair':
            WashingMachineRepair = WashingMachineRepair + 1
        if classResult == 'DryerRepair':
            DryerRepair = DryerRepair + 1
        if classResult == 'DryerVentCleaning':
            DryerVentCleaning = DryerVentCleaning + 1
        if classResult == 'ResidentialApplianceParts':
            ResidentialApplianceParts = ResidentialApplianceParts + 1
    votedservice = 'Data Not Available'
    if(len(words)>0):
        servicetypes = {'Refrigerator Repairs':float(RefrigeratorRepairs)/len(words),'Refrigerator Not Cold Enough':float(RefrigeratorNotColdEnough)/len(words),'Freezer Repairs':float(FreezerRepairs)/len(words),'Dishwasher Repairs':float(DishwasherRepairs)/len(words),'Ovens,Stove Tops & Ranges':float(OvensStoveTopsRanges)/len(words),'Ice Machine Repairs':float(IceMachineRepairs)/len(words),'Garbage Disposal Repairs':float(GarbageDisposalRepairs)/len(words),'Microwave Oven Repairs':float(MicrowaveOvenRepairs)/len(words),'Vent Hoods':float(VentHoods)/len(words),'Wine Coolers':float(WineCoolers)/len(words),'Trash Compactors':float(TrashCompactors)/len(words),'Outdoor Kitchens':float(OutdoorKitchens)/len(words),'Washing Machine Repair':float(WashingMachineRepair)/len(words),'Dryer Repair':float(DryerRepair)/len(words),'Dryer Vent Cleaning':float(DryerVentCleaning)/len(words),'Residential Appliance Parts':float(ResidentialApplianceParts)/len(words)}
        votedservice = max(servicetypes.items(), key=operator.itemgetter(1))[0]
    return votedservice

def MRAComtagger(doc_clean):
    #kitchen
    refrigerators = clean('Refrigerators').split()
    freezer = clean('Freezers').split()
    oven = clean('Ovens, Stove Tops & Ranges').split()
    ice = clean('Ice Machines').split()
    mixer = clean('Mixers').split()
    pizza = clean('Pizza Tables').split()
    sandwich = clean('Sandwich Prep Tables').split()
    steam = clean('Steam Tables').split()
    wfreeze = clean('Walk-In Freezers').split()
    fryer = clean('Deep Fryers').split()
    cooler = clean('Bar Coolers').split()

    #laundry
    wash = clean('Washing Machine Repair').split()
    dry = clean('Dryer Repair').split()
    dryclean = clean('Dryer Vent Cleaning').split()

    #Commercial Appliance Parts
    compart = clean('Commercial Appliance Parts').split()

    Refrigerators_features = [(word_feats(Refrigerators), 'Refrigerators') for Refrigerators in refrigerators]
    Freezers_features = [(word_feats(Freezers), 'Freezers') for Freezers in freezer]
    OvensStoveTopsRanges_features = [(word_feats(OvensStoveTopsRanges), 'OvensStoveTopsRanges') for OvensStoveTopsRanges in oven]
    IceMachines_features = [(word_feats(IceMachines), 'Ice Machines') for IceMachines in ice]
    Mixers_features = [(word_feats(Mixers), 'Mixers') for Mixers in mixer]
    PizzaTables_features = [(word_feats(PizzaTables), 'Pizza Tables') for PizzaTables in pizza]
    SandwichPrepTables_features = [(word_feats(SandwichPrepTables), 'Sandwich Prep Tables') for SandwichPrepTables in sandwich]
    SteamTables_features = [(word_feats(SteamTables), 'Steam Tables') for SteamTables in steam]
    WalkInFreezers_features = [(word_feats(WalkInFreezers), 'Walk-In Freezers') for WalkInFreezers in wfreeze]
    DeepFryers_features = [(word_feats(DeepFryers), 'Deep Fryers') for DeepFryers in fryer]
    BarCoolers_features = [(word_feats(BarCoolers), 'Bar Coolers') for BarCoolers in cooler]

    WashingMachineRepair_features = [(word_feats(WashingMachineRepair), 'WashingMachineRepair') for WashingMachineRepair in wash]
    DryerRepair_features = [(word_feats(DryerRepair), 'DryerRepair') for DryerRepair in dry]
    DryerVentCleaning_features = [(word_feats(DryerVentCleaning), 'DryerVentCleaning') for DryerVentCleaning in dryclean]

    CommercialApplianceParts_features = [(word_feats(CommercialApplianceParts), 'CommercialApplianceParts') for CommercialApplianceParts in compart]

    train_set = Refrigerators_features+Freezers_features+OvensStoveTopsRanges_features+IceMachines_features+Mixers_features+PizzaTables_features+SandwichPrepTables_features+SteamTables_features+WalkInFreezers_features+DeepFryers_features+BarCoolers_features+WashingMachineRepair_features+DryerRepair_features+DryerVentCleaning_features+CommercialApplianceParts_features

    classifier = NaiveBayesClassifier.train(train_set)
    # Predict
    Refrigerators=0
    Freezers=0
    OvensStoveTopsRanges=0
    IceMachines=0
    Mixers=0
    PizzaTables=0
    SandwichPrepTables=0
    SteamTables=0
    WalkInFreezers=0
    DeepFryers=0
    BarCoolers=0
    WashingMachineRepair=0
    DryerRepair=0
    DryerVentCleaning=0
    CommercialApplianceParts=0

    words = clean(doc_clean).split()
    for word in words:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'Refrigerators':
            Refrigerators = Refrigerators + 1
        if classResult == 'Freezers':
            Freezers = Freezers + 1
        if classResult == 'OvensStoveTopsRanges':
            OvensStoveTopsRanges = OvensStoveTopsRanges + 1
        if classResult == 'IceMachines':
            IceMachines = IceMachines + 1
        if classResult == 'Mixers':
            Mixers = Mixers + 1
        if classResult == 'PizzaTables':
            PizzaTables = PizzaTables + 1
        if classResult == 'SandwichPrepTables':
            SandwichPrepTables = SandwichPrepTables + 1
        if classResult == 'SteamTables':
            SteamTables = SteamTables + 1
        if classResult == 'WalkInFreezers':
            WalkInFreezers = WalkInFreezers + 1
        if classResult == 'DeepFryers':
            DeepFryers = DeepFryers + 1
        if classResult == 'BarCoolers':
            BarCoolers = BarCoolers + 1
        if classResult == 'WashingMachineRepair':
            WashingMachineRepair = WashingMachineRepair + 1
        if classResult == 'DryerRepair':
            DryerRepair = DryerRepair + 1
        if classResult == 'DryerVentCleaning':
            DryerVentCleaning = DryerVentCleaning + 1
        if classResult == 'CommercialApplianceParts':
            CommercialApplianceParts = CommercialApplianceParts + 1
    votedservice = 'Data Not Available'
    if(len(words)>0):
        servicetypes = {'Refrigerators':float(Refrigerators)/len(words),'Freezers':float(Freezers)/len(words),'Ovens, Stove Tops & Ranges':float(OvensStoveTopsRanges)/len(words),'Ice Machines':float(IceMachines)/len(words),'Mixers':float(Mixers)/len(words),'Pizza Tables':float(PizzaTables)/len(words),'Sandwich Prep Tables':float(SandwichPrepTables)/len(words),'Steam Tables':float(SteamTables)/len(words),'Walk-In Freezers':float(WalkInFreezers)/len(words),'Deep Fryers':float(DeepFryers)/len(words),'Bar Coolers':float(BarCoolers)/len(words),'Washing Machine Repair':float(WashingMachineRepair)/len(words),'Dryer Repair':float(DryerRepair)/len(words),'Dryer Vent Cleaning':float(DryerVentCleaning)/len(words),'Commercial Appliance Parts':float(CommercialApplianceParts)/len(words)}
        votedservice = max(servicetypes.items(), key=operator.itemgetter(1))[0]
    return votedservice

def MRRRestagger(doc_clean):
    #kitchen
    draincleaning = clean('Drain Cleaning').split()
    hydroscrub = clean('HydroScrub - Jetting').split()
    sewersystem = clean('Sewer System Backups').split()
    sewerline = clean('Sewer Line Repair & Replacements').split()
    trenchlesssewer = clean('Trenchless Sewer Line Repair').split()
    plumbingvideo = clean('Plumbing Video Camera Inspection').split()
    plumbingdiagnosis = clean('Plumbing Diagnosis & Inspection').split()
    plumbingrepairs = clean('Plumbing Repairs').split()
    plumbingreplacement = clean('Plumbing Replacement & Installations').split()
    waterlinerepairs = clean('Water Line Repairs').split()
    waterlinereplacement = clean('Water Line Replacement & Installations').split()
    frozenpipes = clean('Frozen Pipes').split()
    leakingpipes = clean('Leaking Pipes').split()
    waterheaterrepair = clean('Water Heater Repair & Replacements').split()
    plumbingsystem = clean('Plumbing System Maintenance').split()
    emergencyservice = clean('Emergency Service').split()
    wellpumpservice = clean('Well Pump Service').split()
    wellpump = clean('Well Pump Installation & Replacement').split()
    sumppumps = clean('Sump Pumps').split()
    cloggeddrains = clean('Clogged Drains').split()

    DrainCleaning_features = [(word_feats(DrainCleaning), 'DrainCleaning') for DrainCleaning in draincleaning]
    HydroScrub_features = [(word_feats(HydroScrub), 'HydroScrub') for HydroScrub in hydroscrub]
    SewerSystem_features = [(word_feats(SewerSystem), 'SewerSystem') for SewerSystem in sewersystem]
    SewerLine_features = [(word_feats(SewerLine), 'SewerLine') for SewerLine in sewerline]
    TrenchlessSewer_features = [(word_feats(TrenchlessSewer), 'TrenchlessSewer') for TrenchlessSewer in trenchlesssewer]
    PlumbingVideo_features = [(word_feats(PlumbingVideo), 'PlumbingVideo') for PlumbingVideo in plumbingvideo]
    PlumbingDiagnosis_features = [(word_feats(PlumbingDiagnosis), 'PlumbingDiagnosis') for PlumbingDiagnosis in plumbingdiagnosis]
    PlumbingRepairs_features = [(word_feats(PlumbingRepairs), 'PlumbingRepairs') for PlumbingRepairs in plumbingrepairs]
    PlumbingReplacement_features = [(word_feats(PlumbingReplacement), 'PlumbingReplacement') for PlumbingReplacement in plumbingreplacement]
    WaterLineRepairs_features = [(word_feats(WaterLineRepairs), 'WaterLineRepairs') for WaterLineRepairs in waterlinerepairs]
    WaterLineReplacement_features = [(word_feats(WaterLineReplacement), 'WaterLineReplacement') for WaterLineReplacement in waterlinereplacement]
    FrozenPipes_features = [(word_feats(FrozenPipes), 'FrozenPipes') for FrozenPipes in frozenpipes]
    LeakingPipes_features = [(word_feats(LeakingPipes), 'LeakingPipes') for LeakingPipes in leakingpipes]
    WaterHeaterRepair_features = [(word_feats(WaterHeaterRepair), 'WaterHeaterRepair') for WaterHeaterRepair in waterheaterrepair]
    PlumbingSystem_features = [(word_feats(PlumbingSystem), 'PlumbingSystem') for PlumbingSystem in plumbingsystem]
    EmergencyService_features = [(word_feats(EmergencyService), 'EmergencyService') for EmergencyService in emergencyservice]
    WellPumpService_features = [(word_feats(WellPumpService), 'WellPumpService') for WellPumpService in wellpumpservice]
    WellPump_features = [(word_feats(WellPump), 'WellPump') for WellPump in wellpump]
    SumpPumps_features = [(word_feats(SumpPumps), 'SumpPumps') for SumpPumps in sumppumps]
    CloggedDrains_features = [(word_feats(CloggedDrains), 'CloggedDrains') for CloggedDrains in cloggeddrains]

    train_set = DrainCleaning_features +HydroScrub_features +SewerSystem_features +SewerLine_features +TrenchlessSewer_features +PlumbingVideo_features +PlumbingDiagnosis_features +PlumbingRepairs_features +PlumbingReplacement_features +WaterLineRepairs_features +WaterLineReplacement_features +FrozenPipes_features +LeakingPipes_features +WaterHeaterRepair_features +PlumbingSystem_features +EmergencyService_features +WellPumpService_features +WellPump_features +SumpPumps_features +CloggedDrains_features

    classifier = NaiveBayesClassifier.train(train_set)
    # Predict
    DrainCleaning=0
    HydroScrub=0
    SewerSystem=0
    SewerLine=0
    TrenchlessSewer=0
    PlumbingVideo=0
    PlumbingDiagnosis=0
    PlumbingRepairs=0
    PlumbingReplacement=0
    WaterLineRepairs=0
    WaterLineReplacement=0
    FrozenPipes=0
    LeakingPipes=0
    WaterHeaterRepair=0
    PlumbingSystem=0
    EmergencyService=0
    WellPumpService=0
    WellPump=0
    SumpPumps=0
    CloggedDrains=0

    words = clean(doc_clean).split()
    for word in words:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'DrainCleaning':
            DrainCleaning = DrainCleaning + 1
        if classResult == 'HydroScrub':
            HydroScrub = HydroScrub + 1
        if classResult == 'SewerSystem':
            SewerSystem = SewerSystem + 1
        if classResult == 'SewerLine':
            SewerLine = SewerLine + 1
        if classResult == 'TrenchlessSewer':
            TrenchlessSewer = TrenchlessSewer + 1
        if classResult == 'PlumbingVideo':
            PlumbingVideo = PlumbingVideo + 1
        if classResult == 'PlumbingDiagnosis':
            PlumbingDiagnosis = PlumbingDiagnosis + 1
        if classResult == 'PlumbingRepairs':
            PlumbingRepairs = PlumbingRepairs + 1
        if classResult == 'PlumbingReplacement':
            PlumbingReplacement = PlumbingReplacement + 1
        if classResult == 'WaterLineRepairs':
            WaterLineRepairs = WaterLineRepairs + 1
        if classResult == 'WaterLineReplacement':
            WaterLineReplacement = WaterLineReplacement + 1
        if classResult == 'FrozenPipes':
            FrozenPipes = FrozenPipes + 1
        if classResult == 'LeakingPipes':
            LeakingPipes = LeakingPipes + 1
        if classResult == 'WaterHeaterRepair':
            WaterHeaterRepair = WaterHeaterRepair + 1
        if classResult == 'PlumbingSystem':
            PlumbingSystem = PlumbingSystem + 1
        if classResult == 'EmergencyService':
            EmergencyService = EmergencyService + 1
        if classResult == 'WellPumpService':
            WellPumpService = WellPumpService + 1
        if classResult == 'WellPump':
            WellPump = WellPump + 1
        if classResult == 'SumpPumps':
            SumpPumps = SumpPumps + 1
        if classResult == 'CloggedDrains':
            CloggedDrains = CloggedDrains + 1
    votedservice = 'Data Not Available'
    if(len(words)>0):
        servicetypes = {'Drain Cleaning':float(DrainCleaning)/len(words),'HydroScrub - Jetting':float(HydroScrub)/len(words),'Sewer System Backups':float(SewerSystem)/len(words),'Sewer Line Repair & Replacements':float(SewerLine)/len(words),'Trenchless Sewer Line Repair':float(TrenchlessSewer)/len(words),'Plumbing Video Camera Inspection':float(PlumbingVideo)/len(words),'Plumbing Diagnosis & Inspection':float(PlumbingDiagnosis)/len(words),'Plumbing Repairs':float(PlumbingRepairs)/len(words),'Plumbing Replacement & Installations':float(PlumbingReplacement)/len(words),'Water Line Repairs':float(WaterLineRepairs)/len(words),'Water Line Replacement & Installations':float(WaterLineReplacement)/len(words),'Frozen Pipes':float(FrozenPipes)/len(words),'Leaking Pipes':float(LeakingPipes)/len(words),'Water Heater Repair & Replacements':float(WaterHeaterRepair)/len(words),'Plumbing System Maintenance':float(PlumbingSystem)/len(words),'Emergency Service':float(EmergencyService)/len(words),'Well Pump Service':float(WellPumpService)/len(words),'Well Pump Installation & Replacement':float(WellPump)/len(words),'Sump Pumps':float(SumpPumps)/len(words),'Clogged Drains':float(CloggedDrains)/len(words)}
        votedservice = max(servicetypes.items(), key=operator.itemgetter(1))[0]
    return votedservice

def MRRComtagger(doc_clean):
    draincleaning = clean('Drain Cleaning').split()
    hydroscrub = clean('HydroScrub - Jetting').split()
    sewersystem = clean('Sewer System Backups').split()
    sewerline = clean('Sewer Line Repair & Replacements').split()
    trenchlesssewer = clean('Trenchless Sewer Line Repair').split()
    plumbingvideo = clean('Plumbing Video Camera Inspection').split()
    plumbingdiagnosis = clean('Plumbing Diagnosis & Inspection').split()
    plumbingrepairs = clean('Plumbing Repairs').split()
    plumbingreplacement = clean('Plumbing Replacement & Installations').split()
    waterlinerepairs = clean('Water Line Repairs').split()
    waterlinereplacement = clean('Water Line Replacement & Installations').split()
    frozenpipes = clean('Frozen Pipes').split()
    leakingpipes = clean('Leaking Pipes').split()
    waterheaterrepair = clean('Water Heater Repair & Replacements').split()
    plumbingsystem = clean('Plumbing System Maintenance').split()
    emergencyservice = clean('Emergency Service').split()
    cloggeddrains = clean('Clogged Drains').split()

    DrainCleaning_features = [(word_feats(DrainCleaning), 'DrainCleaning') for DrainCleaning in draincleaning]
    HydroScrub_features = [(word_feats(HydroScrub), 'HydroScrub') for HydroScrub in hydroscrub]
    SewerSystem_features = [(word_feats(SewerSystem), 'SewerSystem') for SewerSystem in sewersystem]
    SewerLine_features = [(word_feats(SewerLine), 'SewerLine') for SewerLine in sewerline]
    TrenchlessSewer_features = [(word_feats(TrenchlessSewer), 'TrenchlessSewer') for TrenchlessSewer in trenchlesssewer]
    PlumbingVideo_features = [(word_feats(PlumbingVideo), 'PlumbingVideo') for PlumbingVideo in plumbingvideo]
    PlumbingDiagnosis_features = [(word_feats(PlumbingDiagnosis), 'PlumbingDiagnosis') for PlumbingDiagnosis in plumbingdiagnosis]
    PlumbingRepairs_features = [(word_feats(PlumbingRepairs), 'PlumbingRepairs') for PlumbingRepairs in plumbingrepairs]
    PlumbingReplacement_features = [(word_feats(PlumbingReplacement), 'PlumbingReplacement') for PlumbingReplacement in plumbingreplacement]
    WaterLineRepairs_features = [(word_feats(WaterLineRepairs), 'WaterLineRepairs') for WaterLineRepairs in waterlinerepairs]
    WaterLineReplacement_features = [(word_feats(WaterLineReplacement), 'WaterLineReplacement') for WaterLineReplacement in waterlinereplacement]
    FrozenPipes_features = [(word_feats(FrozenPipes), 'FrozenPipes') for FrozenPipes in frozenpipes]
    LeakingPipes_features = [(word_feats(LeakingPipes), 'LeakingPipes') for LeakingPipes in leakingpipes]
    WaterHeaterRepair_features = [(word_feats(WaterHeaterRepair), 'WaterHeaterRepair') for WaterHeaterRepair in waterheaterrepair]
    PlumbingSystem_features = [(word_feats(PlumbingSystem), 'PlumbingSystem') for PlumbingSystem in plumbingsystem]
    EmergencyService_features = [(word_feats(EmergencyService), 'EmergencyService') for EmergencyService in emergencyservice]
    CloggedDrains_features = [(word_feats(CloggedDrains), 'CloggedDrains') for CloggedDrains in cloggeddrains]

    train_set = DrainCleaning_features +HydroScrub_features +SewerSystem_features +SewerLine_features +TrenchlessSewer_features +PlumbingVideo_features +PlumbingDiagnosis_features +PlumbingRepairs_features +PlumbingReplacement_features +WaterLineRepairs_features +WaterLineReplacement_features +FrozenPipes_features +LeakingPipes_features +WaterHeaterRepair_features +PlumbingSystem_features +EmergencyService_features  +CloggedDrains_features

    classifier = NaiveBayesClassifier.train(train_set)
    # Predict
    DrainCleaning=0
    HydroScrub=0
    SewerSystem=0
    SewerLine=0
    TrenchlessSewer=0
    PlumbingVideo=0
    PlumbingDiagnosis=0
    PlumbingRepairs=0
    PlumbingReplacement=0
    WaterLineRepairs=0
    WaterLineReplacement=0
    FrozenPipes=0
    LeakingPipes=0
    WaterHeaterRepair=0
    PlumbingSystem=0
    EmergencyService=0
    CloggedDrains=0

    words = clean(doc_clean).split()
    for word in words:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'DrainCleaning':
            DrainCleaning = DrainCleaning + 1
        if classResult == 'HydroScrub':
            HydroScrub = HydroScrub + 1
        if classResult == 'SewerSystem':
            SewerSystem = SewerSystem + 1
        if classResult == 'SewerLine':
            SewerLine = SewerLine + 1
        if classResult == 'TrenchlessSewer':
            TrenchlessSewer = TrenchlessSewer + 1
        if classResult == 'PlumbingVideo':
            PlumbingVideo = PlumbingVideo + 1
        if classResult == 'PlumbingDiagnosis':
            PlumbingDiagnosis = PlumbingDiagnosis + 1
        if classResult == 'PlumbingRepairs':
            PlumbingRepairs = PlumbingRepairs + 1
        if classResult == 'PlumbingReplacement':
            PlumbingReplacement = PlumbingReplacement + 1
        if classResult == 'WaterLineRepairs':
            WaterLineRepairs = WaterLineRepairs + 1
        if classResult == 'WaterLineReplacement':
            WaterLineReplacement = WaterLineReplacement + 1
        if classResult == 'FrozenPipes':
            FrozenPipes = FrozenPipes + 1
        if classResult == 'LeakingPipes':
            LeakingPipes = LeakingPipes + 1
        if classResult == 'WaterHeaterRepair':
            WaterHeaterRepair = WaterHeaterRepair + 1
        if classResult == 'PlumbingSystem':
            PlumbingSystem = PlumbingSystem + 1
        if classResult == 'EmergencyService':
            EmergencyService = EmergencyService + 1
        if classResult == 'CloggedDrains':
            CloggedDrains = CloggedDrains + 1
    votedservice = 'Data Not Available'
    if(len(words)>0):
        servicetypes = {'Drain Cleaning':float(DrainCleaning)/len(words),'HydroScrub - Jetting':float(HydroScrub)/len(words),'Sewer System Backups':float(SewerSystem)/len(words),'Sewer Line Repair & Replacements':float(SewerLine)/len(words),'Trenchless Sewer Line Repair':float(TrenchlessSewer)/len(words),'Plumbing Video Camera Inspection':float(PlumbingVideo)/len(words),'Plumbing Diagnosis & Inspection':float(PlumbingDiagnosis)/len(words),'Plumbing Repairs':float(PlumbingRepairs)/len(words),'Plumbing Replacement & Installations':float(PlumbingReplacement)/len(words),'Water Line Repairs':float(WaterLineRepairs)/len(words),'Water Line Replacement & Installations':float(WaterLineReplacement)/len(words),'Frozen Pipes':float(FrozenPipes)/len(words),'Leaking Pipes':float(LeakingPipes)/len(words),'Water Heater Repair & Replacements':float(WaterHeaterRepair)/len(words),'Plumbing System Maintenance':float(PlumbingSystem)/len(words),'Emergency Service':float(EmergencyService)/len(words),'Clogged Drains':float(CloggedDrains)/len(words)}
        votedservice = max(servicetypes.items(), key=operator.itemgetter(1))[0]
    return votedservice

def MDGTagger(doc_clean):
    #homeglass
    windowrepair = clean('Window Repair').split()
    showerdoors = clean('Shower Doors').split()
    doublepanewindows = clean('Double Pane Windows').split()
    customglasssolutions = clean('Custom Glass Solutions').split()
    emergencyservice = clean('Emergency Service').split()
    #auto glass
    windshieldrepair = clean('Windshield Repair').split()
    carwindowreplacement = clean('Car Window Replacement').split()
    autoglasscare = clean('Auto Glass Care').split()
    g12roadhazardguarantee = clean('G12® Road Hazard Guarantee').split()
    #business glass
    advancemeasurement = clean('Advance Measurement').split()
    nationalaccounts = clean('National Accounts').split()
    industrysolutions = clean('Industry Solutions').split()
    busemergencyservice = clean('Emergency Service').split()

    WindowRepair_features = [(word_feats(WindowRepair), 'WindowRepair') for WindowRepair in windowrepair]
    ShowerDoors_features = [(word_feats(ShowerDoors), 'ShowerDoors') for ShowerDoors in showerdoors]
    DoublePaneWindows_features = [(word_feats(DoublePaneWindows), 'DoublePaneWindows') for DoublePaneWindows in doublepanewindows]
    CustomGlassSolutions_features = [(word_feats(CustomGlassSolutions), 'CustomGlassSolutions') for CustomGlassSolutions in customglasssolutions]
    EmergencyService_features = [(word_feats(EmergencyService), 'EmergencyService') for EmergencyService in emergencyservice]
    WindshieldRepair_features = [(word_feats(WindshieldRepair), 'WindshieldRepair') for WindshieldRepair in windshieldrepair]
    CarWindowReplacement_features = [(word_feats(CarWindowReplacement), 'CarWindowReplacement') for CarWindowReplacement in carwindowreplacement]
    AutoGlassCare_features = [(word_feats(AutoGlassCare), 'AutoGlassCare') for AutoGlassCare in autoglasscare]
    G12RoadHazardGuarantee_features = [(word_feats(G12RoadHazardGuarantee), 'G12®RoadHazardGuarantee') for G12RoadHazardGuarantee in g12roadhazardguarantee]
    AdvanceMeasurement_features = [(word_feats(AdvanceMeasurement), 'AdvanceMeasurement') for AdvanceMeasurement in advancemeasurement]
    NationalAccounts_features = [(word_feats(NationalAccounts), 'NationalAccounts') for NationalAccounts in nationalaccounts]
    IndustrySolutions_features = [(word_feats(IndustrySolutions), 'IndustrySolutions') for IndustrySolutions in industrysolutions]
    BusEmergencyService_features = [(word_feats(EmergencyService), 'EmergencyService') for EmergencyService in busemergencyservice]


    train_set = WindowRepair_features +ShowerDoors_features +DoublePaneWindows_features +CustomGlassSolutions_features +EmergencyService_features +WindshieldRepair_features +CarWindowReplacement_features +AutoGlassCare_features +G12RoadHazardGuarantee_features +AdvanceMeasurement_features +NationalAccounts_features +IndustrySolutions_features +BusEmergencyService_features

    classifier = NaiveBayesClassifier.train(train_set)
    # Predict
    WindowRepair=0
    ShowerDoors=0
    DoublePaneWindows=0
    CustomGlassSolutions=0
    EmergencyService=0
    WindshieldRepair=0
    CarWindowReplacement=0
    AutoGlassCare=0
    G12RoadHazardGuarantee=0
    AdvanceMeasurement=0
    NationalAccounts=0
    IndustrySolutions=0
    BusEmergencyService=0

    words = clean(doc_clean).split()
    for word in words:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'WindowRepair':
            WindowRepair = WindowRepair + 1
        if classResult == 'ShowerDoors':
            ShowerDoors = ShowerDoors + 1
        if classResult == 'DoublePaneWindows':
            DoublePaneWindows = DoublePaneWindows + 1
        if classResult == 'CustomGlassSolutions':
            CustomGlassSolutions = CustomGlassSolutions + 1
        if classResult == 'EmergencyService':
            EmergencyService = EmergencyService + 1
        if classResult == 'WindshieldRepair':
            WindshieldRepair = WindshieldRepair + 1
        if classResult == 'CarWindowReplacement':
            CarWindowReplacement = CarWindowReplacement + 1
        if classResult == 'AutoGlassCare':
            AutoGlassCare = AutoGlassCare + 1
        if classResult == 'G12®RoadHazardGuarantee':
            G12RoadHazardGuarantee = G12RoadHazardGuarantee + 1
        if classResult == 'AdvanceMeasurement':
            AdvanceMeasurement = AdvanceMeasurement + 1
        if classResult == 'NationalAccounts':
            NationalAccounts = NationalAccounts + 1
        if classResult == 'IndustrySolutions':
            IndustrySolutions = IndustrySolutions + 1
        if classResult == 'EmergencyService':
            BusEmergencyService = BusEmergencyService + 1
    votedservice = 'Data Not Available'
    if(len(words)>0):
        servicetypes = {'Window Repair':float(WindowRepair)/len(words),'Shower Doors':float(ShowerDoors)/len(words),'Double Pane Windows':float(DoublePaneWindows)/len(words),'Custom Glass Solutions':float(CustomGlassSolutions)/len(words),'Emergency Service':float(EmergencyService)/len(words),'Windshield Repair':float(WindshieldRepair)/len(words),'Car Window Replacement':float(CarWindowReplacement)/len(words),'Auto Glass Care':float(AutoGlassCare)/len(words),'G12 Road Hazard Guarantee':float(G12RoadHazardGuarantee)/len(words),'Advance Measurement':float(AdvanceMeasurement)/len(words),'National Accounts':float(NationalAccounts)/len(words),'Industry Solutions':float(IndustrySolutions)/len(words),'Business Emergency Service':float(BusEmergencyService)/len(words)}
        votedservice = str(max(servicetypes.items(), key=operator.itemgetter(1))[0])
    return votedservice

def masterTagger(df):
    df['ServiceTypeTag'] = np.nan
    df['ServiceTypeTag'] = df.ServiceTypeTag.astype(str)
    for i in range(len(df)):
        if(df.at[i,'JobSummary']):
            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Residential" ):
                df.at[i,'ServiceTypeTag']=MRARestagger(df.at[i,'JobSummary'])
            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Commercial" ):
                df.at[i,'ServiceTypeTag']=MRAComtagger(df.at[i,'JobSummary'])
            if(df.at[i,'ConceptCode']=="MRR" and df.at[i,'SyncCustomerType']== "Residential" ):
                df.at[i,'ServiceTypeTag']=MRRRestagger(df.at[i,'JobSummary'])
            if(df.at[i,'ConceptCode']=="MRR" and df.at[i,'SyncCustomerType']== "Commercial" ):
                df.at[i,'ServiceTypeTag']=MRRComtagger(df.at[i,'JobSummary'])
            if(df.at[i,'ConceptCode']=="MDG" ):
                df.at[i,'ServiceTypeTag']=MDGTagger(df.at[i,'JobSummary'])
        else:
            df.at[i,'ServiceTypeTag']='Data Not Available'
        print(df.at[i,'ServiceTypeTag'])
    print('Service Tagging Process Complete')

def serviceclassTagger(df):
    df['ServiceClass'] = np.nan
    df['ServiceClass'] = df.ServiceTypeTag.astype(str)
    for i in range(len(df)):
        if(df.at[i,'ServiceTypeTag']!='Data Not Available'):
            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Residential" and  df.at[i,'ServiceTypeTag'] in ['Refrigerator Repairs', 'Refrigerator Not Cold Enough', 'Freezer Repairs', 'Dishwasher Repairs', 'Ovens,Stove Tops & Ranges', 'Ice Machine Repairs', 'Garbage Disposal Repairs', 'Microwave Oven Repairs', 'Vent Hoods', 'Wine Coolers', 'Trash Compactors', 'Outdoor Kitchens'] ):
                df.at[i,'ServiceClass']='Kitchen'
            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Residential" and  df.at[i,'ServiceTypeTag'] in ['Washing Machine Repair', 'Dryer Repair', 'Dryer Vent Cleaning'] ):
                df.at[i,'ServiceClass']='Laundry'
            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Residential" and  df.at[i,'ServiceTypeTag'] in ['Residential Appliance Parts'] ):
                df.at[i,'ServiceClass']='Residential Appliance Parts'

            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Commercial" and  df.at[i,'ServiceTypeTag'] in ['Refrigerators', 'Freezers', 'Ovens,Stove Tops & Ranges', 'Ice Machines', 'Mixers', 'Pizza Tables', 'Sandwich Prep Tables', 'Steam Tables', 'Walk-In Freezers', 'Deep Fryers', 'Bar Coolers'] ):
                df.at[i,'ServiceClass']='Kitchen'
            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Commercial" and  df.at[i,'ServiceTypeTag'] in ['Washing Machine Repair','Dryer Repair','Dryer Vent Cleaning'] ):
                df.at[i,'ServiceClass']='Laundry'
            if(df.at[i,'ConceptCode']=="MRA" and df.at[i,'SyncCustomerType']== "Commercial" and  df.at[i,'ServiceTypeTag'] in ['Commercial Appliance Parts'] ):
                df.at[i,'ServiceClass']='Commercial Appliance Parts'

            if(df.at[i,'ConceptCode']=="MRR" and df.at[i,'SyncCustomerType']== "Residential" and  df.at[i,'ServiceTypeTag'] in ['Drain Cleaning', 'HydroScrub - Jetting', 'Sewer System Backups', 'Sewer Line Repair & Replacements', 'Trenchless Sewer Line Repair', 'Plumbing Video Camera Inspection', 'Plumbing Diagnosis & Inspection', 'Plumbing Repairs', 'Plumbing Replacement & Installations', 'Water Line Repairs', 'Water Line Replacement & Installations', 'Frozen Pipes', 'Leaking Pipes', 'Water Heater Repair & Replacements', 'Plumbing System Maintenance', 'Emergency Service', 'Well Pump Service', 'Well Pump Installation & Replacement', 'Sump Pumps', 'Clogged Drains'] ):
                df.at[i,'ServiceClass']='Residential'

            if(df.at[i,'ConceptCode']=="MRR" and df.at[i,'SyncCustomerType']== "Commercial" and  df.at[i,'ServiceTypeTag'] in ['Clogged Drains', 'Drain Cleaning', 'HydroScrub - Jetting', 'Sewer System Backups', 'Sewer Line Repair & Replacements', 'Trenchless Sewer Line Repair', 'Plumbing Video Camera Inspection', 'Plumbing Diagnosis & Inspection', 'Plumbing Replacement & Installations', 'Water Line Repairs', 'Water Line Replacement & Installations', 'Plumbing Repairs', 'Frozen Pipes', 'Leaking Pipes', 'Water Heater Repair & Replacements', 'Plumbing System Maintenance', 'Emergency Service'] ):
                df.at[i,'ServiceClass']='Commercial'

            if(df.at[i,'ConceptCode']=="MDG" and  df.at[i,'ServiceTypeTag'] in ['Window Repair','Shower Doors','Double Pane Windows','Custom Glass Solutions','Emergency Service'] ):
                df.at[i,'ServiceClass']='Home Glass'
            if(df.at[i,'ConceptCode']=="MDG" and  df.at[i,'ServiceTypeTag'] in ['Windshield Repair','Car Window Replacement','Auto Glass Care','G12 Road Hazard Guarantee'] ):
                df.at[i,'ServiceClass']='Auto Glass'
            if(df.at[i,'ConceptCode']=="MDG" and  df.at[i,'ServiceTypeTag'] in ['Advance Measurement','National Accounts','Industry Solutions','Business Emergency Service'] ):
                df.at[i,'ServiceClass']='Business Glass'

        else:
            df.at[i,'ServiceClass']='Service Type Not Available'
    print('Service Classes Identification Process Complete')


def MasterAffinityScoredata(masterdf1):
    masterdf1['ServiceId'] = np.nan
    i=0
    for i in range(len(masterdf1)):
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Refrigerator Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=101
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Refrigerator Not Cold Enough' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=102
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Freezer Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=103
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Dishwasher Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=104
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Ovens,Stove Tops & Ranges' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=105
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Ice Machine Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=106
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Garbage Disposal Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=107
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Microwave Oven Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=108
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Vent Hoods' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=109
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Wine Coolers' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=110
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Trash Compactors' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=111
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Outdoor Kitchens' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=112
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Washing Machine Repair' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=113
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Dryer Repair' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=114
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Dryer Vent Cleaning' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=115
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Residential Appliance Parts' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=116


    for k in range(len(masterdf1)):
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Refrigerators' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=201
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Freezers' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=202
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Ovens, Stove Tops & Ranges' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=203
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Ice Machines' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=204
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Mixers' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=205
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Pizza Tables' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=206
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Sandwich Prep Tables' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=207
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Steam Tables' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=208
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Walk-In Freezers' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=209
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Deep Fryers' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=210
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Bar Coolers' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=211
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Washing Machine Repair' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=212
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Dryer Repair' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=213
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Dryer Vent Cleaning' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=214
                if(masterdf1.at[i,'ConceptCode']=="MRA" and masterdf1.at[i,'ServiceTypeTag']=='Commercial Appliance Parts' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=215


    for j in range(len(masterdf1)):
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Drain Cleaning' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=301
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='HydroScrub - Jetting' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=302
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Sewer System Backups' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=303
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Sewer Line Repair & Replacements' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=304
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Trenchless Sewer Line Repair' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=305
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Video Camera Inspection' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=306
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Diagnosis & Inspection' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=307
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=308
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Replacement & Installations' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=309
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Water Line Repairs' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=310
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Water Line Replacement & Installations' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=311
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Frozen Pipes' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=312
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Leaking Pipes' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=313
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Water Heater Repair & Replacements' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=314
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing System Maintenance' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=315
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Emergency Service' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=316
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Well Pump Service' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=317
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Well Pump Installation & Replacement' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=318
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Sump Pumps' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=319
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Clogged Drains' and masterdf1.at[i,'SyncCustomerType']== "Residential" ):
                    masterdf1.at[i,'ServiceId']=320


    for m in range(len(masterdf1)):
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Drain Cleaning' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='HydroScrub - Jetting' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=402
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Sewer System Backups' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=403
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Sewer Line Repair & Replacements' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=404
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Trenchless Sewer Line Repair' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=405
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Video Camera Inspection' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=406
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Diagnosis & Inspection' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=407
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Repairs' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=408
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing Replacement & Installations' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=409
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Water Line Repairs' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=410
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Water Line Replacement & Installations' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=411
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Frozen Pipes' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=412
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Leaking Pipes' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=413
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Water Heater Repair & Replacements' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=414
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Plumbing System Maintenance' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=415
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Emergency Service' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=416
                if(masterdf1.at[i,'ConceptCode']=="MRR" and masterdf1.at[i,'ServiceTypeTag']=='Clogged Drains' and masterdf1.at[i,'SyncCustomerType']== "Commercial" ):
                    masterdf1.at[i,'ServiceId']=417

    for z in range(len(masterdf1)):
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Window Repair'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Shower Doors'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Double Pane Windows'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Custom Glass Solutions'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Emergency Service'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Windshield Repair'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Car Window Replacement'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Auto Glass Care'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Advance Measurement'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='National Accounts'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Industry Solutions'):
                    masterdf1.at[i,'ServiceId']=401
                if(masterdf1.at[i,'ConceptCode']=="MDG" and masterdf1.at[i,'ServiceTypeTag']=='Emergency Service'):
                    masterdf1.at[i,'ServiceId']=401
    #Get list of unique items
    itemList=list(set(masterdf1["ServiceId"].tolist()))

    #Get count of users
    userCount=len(set(masterdf1["ServiceId"].tolist()))

    #Create an empty data frame to store item affinity scores for items.
    itemAffinity= pd.DataFrame(columns=('item1', 'item2', 'score'))
    rowCount=0

    #For each item in the list, compare with other items.
    for ind1 in range(len(itemList)):

        #Get list of users who bought this item 1.
        item1Users = masterdf1[masterdf1.ServiceId==itemList[ind1]]["MasterCustomerId"].tolist()
        #print("Item 1 ", item1Users)

        #Get item 2 - items that are not item 1 or those that are not analyzed already.
        for ind2 in range(ind1, len(itemList)):

            if ( ind1 == ind2):
                continue

            #Get list of users who bought item 2
            item2Users=masterdf1[masterdf1.ServiceId==itemList[ind2]]["MasterCustomerId"].tolist()
            #print("Item 2",item2Users)

            #Find score. Find the common list of users and divide it by the total users.
            commonUsers= len(set(item1Users).intersection(set(item2Users)))
            score=commonUsers / userCount

            #Add a score for item 1, item 2
            itemAffinity.loc[rowCount] = [itemList[ind1],itemList[ind2],score]
            rowCount +=1
            #Add a score for item2, item 1. The same score would apply irrespective of the sequence.
            itemAffinity.loc[rowCount] = [itemList[ind2],itemList[ind1],score]
            rowCount +=1

    print("Affinity Scoring Completed")
    return itemAffinity

if __name__ == '__main__':
    PATH = 'C:\\Users\\Praveen\\Documents\\MSDS\\ISQS5381-capstoneproject\\'
    RAWFILE = 'Inputfile.txt'
    CONCEPTCODE = ['MRA','MRR','MDG']
    rawRec = []
    custID = []
    numRec = 2000
    getFileSize(PATH,RAWFILE)
    readFileInToList(PATH,RAWFILE,rawRec,numRec)

    # use below function to read specific lines
    # readNRowsFileInToList(PATH,RAWFILE,recList,5000,100000)

    # Filter out the empty lines from the records
    rawRec = list(filter(lambda i:i != '\n',rawRec))

    # drop the first record as it is header
    header = rawRec[0]
    rawRec = rawRec[1:]

    custID = getUniqueCustIDs(PATH,RAWFILE,custID,CONCEPTCODE)
    rawRec = fixNewLineIssue(rawRec,custID,CONCEPTCODE)
    processedRec = fixTabIssue(rawRec)

    # check all the correct number of columns in records
    ['Something is wrong' for i in processedRec if(i.count('\t')!=77)] or 'All is Well'

    masterDF = listToDataFrame(processedRec,header)

    masterTagger(masterDF)
    serviceclassTagger(masterDF)
    masterDF.to_csv(PATH+'taggeddataframe.csv')
    scoredf = pd.DataFrame(masterDF.loc[(masterDF["ServiceTypeTag"] != "Data Not Available") & (masterDF["ServiceTypeTag"] != "nan"), ["MasterCustomerId","ServiceTypeTag","SyncCustomerType","ConceptCode"]])
    scoredf.reset_index(drop = True, inplace = True)
    itemAffinity = MasterAffinityScoredata(scoredf)
    itemAffinity.to_csv(PATH+'score.csv')
