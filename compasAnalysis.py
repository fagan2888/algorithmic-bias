

#compasAnalysis.py is for exploring the data set used in the 2016 ProPublica article,
#"Machine Bias: There's software used across the country to predict future
#criminals. And it's biased against blacks."
#
#Story:
#https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing/
#Methodology:
#https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm/
#data set on gitub at
#https://github.com/propublica/compas-analysis
#
#Eric Saund
#February, March 2020
#

#Contents:
# -plot recidivism data, confusion matrices, and ROC curve
# -synthesize imaginary data sets
# -build independent models
#   -feature engineering
#   -plotting and evaluation functions
#   -some additional material for running models on the Titanic data set
# -ELI5 feature importances
# -export as .json files for viewing in the Confusion Matrix Dashboard



import csv
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')    #needs explanation
import matplotlib.ticker as mpl_ticker
import numpy as np
import random
import math
import json




#In this file, data items are represented as dicts with key: str feature-name,  value: str feature-value.
#A colletion of data is a list of dict, referred to here as ddict_list.


################################################################################
#
#Positive outcome tag
#


#This was originally written for Broward Compas data, where the positive value
#is about recidivism.  Even as this expands to include Titanic and possibly other
#data sets, for now the Positive outcome is called gl_recidivism_tag
#
#Use these global variables to switch between predicting and scoring two-year recidivism
#versus violent recidivism.  It seems that for violent recidivism, it is counted even if
#it occurs after two years.
#To change these values within an interpreter session, use setRecidType() below.
#
###gl_recidivism_tag = 'is_recid'          #Not sure why this is different from two_year_recid

try:
    gl_recidivism_tag
except:
    gl_recidivism_tag = 'two_year_recid'
    gl_decile_score_tag = 'decile_score'
    
#gl_recidivism_tag = 'is_violent_recid'
#gl_decile_score_tag = 'v_decile_score'


#
#
################################################################################


################################################################################
#
#Load and plot the ProPublica recidivism data to replicate and expand on
#the distributions and confusion matrices they report, and the ROC curve
#from the COMPAS scores, which they don't build or report on.
#

gl_data_dirname = 'data'
gl_data_filename = 'compas-scores-two-years.csv'
gl_v_data_filename = 'compas-scores-two-years-violent.csv'


def loadDataReturnDDictList(recid_type='recid'):
    row_list = loadData(recid_type)
    ddict_list = []    #list of dict: data-keys/ data-values
    field_name_index_dict = {}  #key: str field_name
                                #value: int index

    header_row = row_list[0]
    for i in range(len(header_row)):
        header = header_row[i]
        field_name_index_dict[header] = i

    for row in row_list[1:]:
        ddict = {}
        for field_name in field_name_index_dict.keys():
            field_index = field_name_index_dict.get(field_name)
            field_value = row[field_index]
            ddict[field_name] = field_value
        ddict_list.append(ddict)
    return ddict_list
                           

#recid_type can be 'recid' or one of ('v_recid', 'violent', 'violent_recid')
def loadData(recid_type='recid'):
    if recid_type in ('v_recid', 'violent', 'violent_recid'):
        data_filepath = os.path.join(gl_data_dirname, gl_v_data_filename)
    else:
        data_filepath = os.path.join(gl_data_dirname, gl_data_filename)        
    row_list = []
    print('loading data from: ' + data_filepath)
    with open(data_filepath, 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_list.append(row)
    return row_list


#Writes a list of dicts to to a json file
def writeTDictListToJSONFile(tdict_list, filepath):
    with open(filepath, 'w') as file:
        json.dump(tdict_list, file, indent=4)


def readDDictListFromJSONFile(filepath):
    with open(filepath, 'r', encoding='utf8') as file:
        ddict_list = json.load(file)
    return ddict_list
    



#field-name-value-list is a list of tuple, (str field_name, str comp, str or int field_value)
#that must be true for the field to be included in the result
#e.g. ('das_b_screening_arrest', '<=', 30)
#This applies boolean AND to the filter tuples.
def filterDdict(ddict_list, field_name_comp_value_tup_list):
    new_ddict_list = []
    for ddict in ddict_list:
        ok_p = True
        for item_tup in field_name_comp_value_tup_list:
            target_field_name = item_tup[0]
            comp = item_tup[1]
            target_field_value = item_tup[2]
            try:
                target_field_value = int(target_field_value)
            except:
                target_field_value = "'" + target_field_value + "'"
                
            field_val= ddict.get(target_field_name)
            if field_val == '':
                field_val = 100000000
            if field_val == None:
                print('problem: field_val is none for target_field_name: ' + str(target_field_name) + ' item_tup: ' + str(item_tup))
                return ddict
            try:
                field_val = int(field_val)
            except:
                field_val = "'" + field_val + "'"
            if type(target_field_value) != type(field_val):
                print('problem: target_field_value: ' + str(target_field_value) + ' is type: ' + str(type(target_field_value)) + ' but field_val: ' + str(field_val) + ' is type: ' + str(type(field_val)))
                print('item_tup: ' + str(item_tup))
                return ddict
            #print(str(field_val) + ' comp: ' + str(comp) + ' target_field_value: ' + str(target_field_value))
            eval_str = str(field_val) + ' ' + comp + ' ' + str(target_field_value)
            #print('eval_str: ' + str(eval_str))
            if not eval(eval_str):
                ok_p = False
                break
        if ok_p:
            new_ddict_list.append(ddict)
    return new_ddict_list


#Apply filters according to 
#https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
#recid_type can be 'recid' or one of ('v_recid', 'violent', 'violent_recid')
def applyFilters(ddict_list, recid_type='recid'):
    if recid_type in ('v_recid', 'violent', 'violent_recid'):
        score_text_field = 'v_score_text'
    else:
        score_text_field = 'score_text'
    ddict_list = filterDdict(ddict_list, [('days_b_screening_arrest', '<=', '30'),
                                          ('days_b_screening_arrest', '>=', '-30'),
                                          ('is_recid', '!=', '-1'),
                                          ('c_charge_degree', '!=', 'O'),
                                          (score_text_field, '!=', 'N/A')])
    return ddict_list



#This builds a histogram of counts of the decile_score_tag which should be an int
#in range 1-n_bins.
#Returns an array of int count indexed from 0 to n_bins-1.
def buildHistByDecile(ddict_list, decile_score_tag=gl_decile_score_tag, n_bins=10):
    hist = [0]*n_bins
    for ddict in ddict_list:
        decile_score = ddict.get(decile_score_tag)
        if decile_score == None:
            print('problem in buildHistByDecile: no decile_score_tag named ' + decile_score_tag)
            return
        decile = int(decile_score) - 1
        hist[decile] += 1
    return hist


def getFieldPossibleValues(ddict_list, field_name):
    val_set = set()
    for ddict in ddict_list:
        val = ddict.get(field_name)
        val_set.add(val)
    return val_set


def tallyFieldValues(ddict_list, field_name):
    val_count_dict = {}
    for ddict in ddict_list:
        val = ddict.get(field_name)
        if val_count_dict.get(val) == None:
            val_count_dict[val] = 0
        val_count_dict[val] += 1
    return val_count_dict



#recid_type can be 'recid' or one of ('v_recid', 'violent', 'violent_recid')
#Use this to set the global variables for recidivism type and also run the
#feature calibration
#ddict_list is needed to compute charge description map.
#Use ddict_list (7214 records), not ddict_list2 (6152 records)
def setRecidType(recid_type, ddict_list):
    global gl_recidivism_tag
    global gl_decile_score_tag
    if recid_type == 'recid':
        gl_recidivism_tag = 'two_year_recid'
        gl_decile_score_tag = 'decile_score'
    elif recid_type in ('v_recid', 'violent', 'violent_recid'):
        gl_recidivism_tag = 'is_violent_recid'
        gl_decile_score_tag = 'v_decile_score'
    else:
        print('unrecognized recid_type ' + recid_type)
        return
    computeChargeDescMap(ddict_list)
    computeAgeFeatureMap()    

        


#To get a handle on the data, tally recid/norecid counts for the field values for the
#field called field_name.
def tallyFieldValuesVsRecid(ddict_list, field_name):
    val_count_dict = {}     #key:   field_value
                            #value: list: [count, num_recid, num_norecid]
    for ddict in ddict_list:
        val = ddict.get(field_name)
        if val_count_dict.get(val) == None:
            val_count_dict[val] = [0, 0, 0]
        val_count_dict[val][0] += 1
        is_recid = ddict.get(gl_recidivism_tag)
        if is_recid == '1':
            val_count_dict[val][1] += 1
        elif is_recid == '0':
            val_count_dict[val][2] += 1
        else:
            print('not recognizing value for is_recid: ' + str(is_recid))
            return ddict

    val_count_list = []
    for val in val_count_dict.keys():
        val_count_tally = val_count_dict[val]
        val_count_tally.append(val)
        val_count_list.append(val_count_tally)
    val_count_list.sort(key = lambda x: x[0], reverse=True)
    return val_count_list
    


    
#Plot a  bar chart of no-recidivism (green) stacked on recidivism (red)
#Overplot the recidivisim rate in the color passed, per decile of the decile_scores.
#ddict_list is a list of dict of key/value pairs, including the keys, decile_score and is_recid.
#plot_what can be one of {'both', 'bars', 'ratio'}
def plotRecidByDecile(ddict_list, recid_rate_color='blue', plot_what='both', ymax=640):
    recid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '1')])
    norecid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '0')])
    
    recid_hist = buildHistByDecile(recid_list)
    norecid_hist = buildHistByDecile(norecid_list)
    ratio_hist = []
    N = 10

    ind = np.arange(N)
    width = 0.35

    plt.ylim(0, ymax)
    if plot_what in ('both', 'all', 'bars'):
        p1 = plt.bar(ind, recid_hist, width, color='red')
        p2 = plt.bar(ind, norecid_hist, width, bottom=recid_hist, color='green')

    if plot_what in ('both', 'all', 'ratio'):
        for i in range(N):
            denom = (recid_hist[i] + norecid_hist[i])
            ratio_plt = ymax * float(recid_hist[i]) / max(denom, .00001)
            ratio_hist.append(ratio_plt)
            #print(str(i) + ' ratio: ' + str(float(recid_hist[i]) / (recid_hist[i] + norecid_hist[i])))
        p3 = plt.plot(ind, ratio_hist, color=recid_rate_color)
    
    plt.show()


#Build an ROC (Reciver Operating Characteristic) curve of Recall vs False Positive Rate.
#Work backwards from the 10th decile of recidivism prediction.
#  FPR =  FP / (FP + TN)
#  Recall = TP / (TP + FN)
#Returns an ROC curve in the form of a tuple: (list-of-float: fpr_list, list-of-float: recall_list)
#Note that the spacing of False Postive Rate in fpr_list will not be even.
#This hardcodes 10 deciles, for the Broward data.
def buildROCCurve(ddict_list):
    recid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '1')])
    recid_hist = buildHistByDecile(recid_list)
    
    norecid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '0')])
    norecid_hist = buildHistByDecile(norecid_list)
    
    recid_total = sum(recid_hist)
    norecid_total = sum(norecid_hist)

    fpr_by_decile = [0]
    recall_by_decile = [0]

    cum_fp = 0
    cum_tn = norecid_total
    cum_tp = 0
    cum_fn = recid_total
    for i in range(9, -1, -1):
        cum_tp += recid_hist[i]        
        cum_fp += norecid_hist[i]
        cum_tn -= norecid_hist[i]
        cum_fn -= recid_hist[i]

        fpr_i = float(cum_fp) / (cum_fp + cum_tn)
        recall_i = float(cum_tp) / (cum_tp + cum_fn)
        fpr_by_decile.append(fpr_i)
        recall_by_decile.append(recall_i)
    print('AUC:' + str(computeROC_AUC((fpr_by_decile, recall_by_decile))))
    return (fpr_by_decile, recall_by_decile)
        

#Plot a Receiver Operating Characteristic curve
#roc_fpr_recall_tup is tuple of list:  (roc_fpr, roc_recall)
def plotROC(roc_fpr_recall_tup, color='blue'):
    auc = computeROC_AUC(roc_fpr_recall_tup)
    print('AUC: ' + str(auc))
    plt.figure(figsize=(5, 5))
    fpr_list = roc_fpr_recall_tup[0]
    recall_list = roc_fpr_recall_tup[1]
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.plot(fpr_list, recall_list, color=color)
    plt.show()


#Compute Area Under Curve for a Receiver Operating Characteristic curve.
#roc_fpr_recall_tup is tuple of list:  (roc_fpr, roc_recall)
#test:
#straight = []
#for i in range(11):
#  straight.append(i/10.0)
def computeROC_AUC(roc_fpr_recall_tup):
    auc_sum = 0.0
    roc_fpr = roc_fpr_recall_tup[0]
    roc_recall = roc_fpr_recall_tup[1]
    n = len(roc_fpr)
    if len(roc_recall) != n:
        print('roc_fpr ' + str(len(roc_fpr)) + ' and roc_recall ' + str(len_roc_recall) + ' need to be lists of the same length, each monotonically increasing from 0 to 1')
        return

    last_x = 0
    last_y = 0
    for i in range(n):
        x_i = roc_fpr[i]
        y_i = roc_recall[i]
        delta_x = x_i - last_x
        delta_y = y_i - last_y
        auc_i = (delta_x * last_y) + delta_x * delta_y / 2.0
        auc_sum += auc_i
        last_x = x_i
        last_y = y_i
    return auc_sum



#Returns a confusion matrix based on decile_score
#decile scores range from 1 to 10
#The ProPublica paper states that decile scores from 1-4 are given score_text = 'Low',
# scores 5, 6, 7 have score_text = 'Medium', scores 8, 9, 10 have score_text = 'High'
#Here, a decile_score <= decile_score_threshold is called low.
#The default decile_score_threshold = 4 corresponds to the score_text = 'Low'
#The confusion matrix is
#                       Prediction               
#                      Low  Med/High
#
#           norecid     TN     FP
# Outcome 
#             recid     FN     TP
#
def buildConfusionMatrix(ddict_list, decile_score_threshold = 4, print_p=True):
    confusion_matrix = []

    norecid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '0')])
    norecid_low = filterDdict(norecid_list, [(gl_decile_score_tag, '<=', decile_score_threshold)])
    norecid_high = filterDdict(norecid_list, [(gl_decile_score_tag, '>', decile_score_threshold)])


    confusion_matrix.append([len(norecid_low), len(norecid_high)])

    recid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '1')])
    recid_low = filterDdict(recid_list, [(gl_decile_score_tag, '<=', decile_score_threshold)])
    recid_high = filterDdict(recid_list, [(gl_decile_score_tag, '>', decile_score_threshold)]) 

    confusion_matrix.append([len(recid_low), len(recid_high)])
    count = sum(confusion_matrix[0]) + sum(confusion_matrix[1])

    if print_p:
        tn = confusion_matrix[0][0]
        tp = confusion_matrix[1][1]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        TPR = float(tp) / (tp + fn)     #recall
        PPV = float(tp) / (tp + fp)     #precision
        f1 = 2*TPR*PPV / (TPR + PPV)
        FPR = float(fp) / (fp + tn)
        FNR = float(fn) / (fn + tp)
        TNR = float(tn) / (tn + fp)
        recid_ratio = float(sum(confusion_matrix[1])) / ( sum(confusion_matrix[0]) + sum(confusion_matrix[1]))
        print('count: ' + str(count))
        print(str(confusion_matrix[0]))
        print(str(confusion_matrix[1]))
        print('FPR: ' + str(FPR))
        print('FNR: ' + str(FNR))
        print('TNR: ' + str(TNR))
        print('TPR = recall: ' + str(TPR))
        print('PPV = precision: ' + str(PPV))
        print('f1: ' + str(f1))
        print('recidivism ratio: ' + str(recid_ratio))
    return confusion_matrix


#pyplot examples
#https://matplotlib.org/gallery/index.html

#Plot two histograms of number of defendents versus decile_score,
#plus a line graph of the fraction of defendents in each decile
#that are Caucasian versus African-American
#One plot for Caucasian, the other for African-American.
def plotCauAAHistByDecileScore(ddict_list):

    ddict_list_cau = filterDdict(ddict_list, [('race', '==', 'Caucasian')])
    ddict_list_aa = filterDdict(ddict_list, [('race', '==', 'African-American')])
    cau_hist = buildHistByDecile(ddict_list_cau)
    aa_hist = buildHistByDecile(ddict_list_aa)
    max_count = max(max(cau_hist), max(aa_hist))

    ind = np.arange(10)
    width = 0.35
    
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10,8)
    axs[0].set_ylim(0, max_count)
    axs[1].set_ylim(0, max_count)
    
    axs[0].bar(ind, cau_hist, width, color='orange')
    axs[1].bar(ind, aa_hist, width, color='saddlebrown')

    cau_count = sum(cau_hist)
    aa_count = sum(aa_hist)
    cau_frac_list = []
    aa_frac_list = []

    for decile in range(0, 10):
        decile_count = cau_hist[decile] + aa_hist[decile]
        cau_frac_list.append(float(cau_hist[decile]) / decile_count)
        aa_frac_list.append(float(aa_hist[decile]) / decile_count)

    axs[2].set_ylim(0, 1.0)
    axs[2].plot(ind, cau_frac_list, width, color='orange')
    axs[2].plot(ind, aa_frac_list, width, color='saddlebrown')

    plt.show()
    



#data record examples

#ddict_list[0] = {'id': '1', 'name': 'miguel hernandez', 'first': 'miguel', 'last': 'hernandez', 'compas_screening_date': '2013-08-14', 'sex': 'Male', 'dob': '1947-04-18', 'age': '69', 'age_cat': 'Greater than 45', 'race': 'Other', 'juv_fel_count': '0', 'decile_score': '1', 'juv_misd_count': '0', 'juv_other_count': '0', 'priors_count': '0', 'days_b_screening_arrest': '-1', 'c_jail_in': '2013-08-13 06:03:42', 'c_jail_out': '2013-08-14 05:41:20', 'c_case_number': '13011352CF10A', 'c_offense_date': '2013-08-13', 'c_arrest_date': '', 'c_days_from_compas': '1', 'c_charge_degree': 'F', 'c_charge_desc': 'Aggravated Assault w/Firearm', 'is_recid': '0', 'r_case_number': '', 'r_charge_degree': '', 'r_days_from_arrest': '', 'r_offense_date': '', 'r_charge_desc': '', 'r_jail_in': '', 'r_jail_out': '', 'violent_recid': '', 'is_violent_recid': '0', 'vr_case_number': '', 'vr_charge_degree': '', 'vr_offense_date': '', 'vr_charge_desc': '', 'type_of_assessment': 'Risk of Recidivism', 'score_text': 'Low', 'screening_date': '2013-08-14', 'v_type_of_assessment': 'Risk of Violence', 'v_decile_score': '1', 'v_score_text': 'Low', 'v_screening_date': '2013-08-14', 'in_custody': '2014-07-07', 'out_custody': '2014-07-14', 'start': '0', 'end': '327', 'event': '0', 'two_year_recid': '0'}

#fdict_list[0] = {'age': 4, 'juv_fel_count': 0, 'juv_misd_count': 0, 'juv_other_count': 0, 'priors_count': 0, 'c_charge_degree': 1, 'c_charge_desc': [0.09090909090909091, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}}



#
#
################################################################################

################################################################################
#
#Synthesize data sets with controlled properties.
#This allows exploration of the characteristics of data distributions
#that lead to interesting properties of the resulting confusion matrices.
#This is all precursor and development work for data synthesis that ended
#up in the Confusion Matrix Dashboard.
#


def generateRandomDecileSamples(N, race_label, frac_recid):
    ddict_list = []
    for i in range(N):
        decile = random.randint(1, 10)
        recid = '1' if random.random() < frac_recid else '0'

        ddict = {'race': race_label,
                 'decile_score': str(decile),
                 gl_recidivism_tag: recid}
        ddict_list.append(ddict)
    return ddict_list


def generateUniformDecileSamples(N, race_label, frac_recid):
    ddict_list = []
    n_per_decile = int(N / 10)
    n_recid_per_decile = int(n_per_decile * frac_recid)
    n_norecid_per_decile = n_per_decile - n_recid_per_decile
    for decile in range(1, 11):
        for i in range(n_recid_per_decile):
            ddict = {'race': race_label,
                     'decile_score': str(decile),
                     gl_recidivism_tag: '1'}
            ddict_list.append(ddict)
        for i in range(n_norecid_per_decile):
            ddict = {'race': race_label,
                     'decile_score': str(decile),
                     gl_recidivism_tag: '0'}
            ddict_list.append(ddict)

    return ddict_list




#This generates a sample of synthetic individuals.
#They are evenly distributed across decile_score.
#The overall recidivism rate is frac_recid.
#The imagined prediction algorithm is linear.  It assigns the gl_recidivsm_tag (like 'is_recid')
#at a level of min_frac_recid for decile_score = 1.  Then, is_recid increases linearly
#with decile_score such that the total fraction of is_recid samples matches frac_recid.
def generateUniformDecileSamplesLinearPrediction(N, race_label, frac_recid, min_frac_recid):
    ddict_list = []
    n_per_decile = int(N / 10)

    for decile_i in range(10):
        str_decile = str(decile_i+1)
        n_recid_at_decile = int(round(n_per_decile * (min_frac_recid + decile_i * (frac_recid - min_frac_recid)/4.5)))
        n_norecid_at_decile = n_per_decile - n_recid_at_decile
        
        for i in range(n_recid_at_decile):
            ddict = {'race': race_label,
                     'decile_score': str_decile,
                     gl_recidivism_tag: '1'}
            ddict_list.append(ddict)
        for i in range(n_norecid_at_decile):
            ddict = {'race': race_label,
                     'decile_score': str_decile,
                     gl_recidivism_tag: '0'}
            ddict_list.append(ddict)

    return ddict_list



#This generates a sample of synthetic individuals.
#The samples decline in proportion to decile_score.
#decile_score = 1 (bin 0) has a fraction of the total population N as initial_frac_N.
#Then, the number of samples per decile declines linearly so the total popluation is N.
#To get the recidivism rate, you have to adjust frac_recid, decile_0_frac_recid, and decile_0_recid_offset.
#
#The overall recidivism rate is frac_recid.
#The imagined prediction algorithm is linear.  It assigns gl_recidivism_tag at a level
#of min_frac_recid for decile_score = 1.  Then, is_recid increases linearly
#with decile_score such that the total fraction of is_recid samples matches frac_recid.
def generateDecliningDecileSamplesLinearPrediction(N, race_label, frac_recid, decile_0_frac_recid,
                                                   decile_0_recid_offset, initial_frac_N):
    ddict_list = []
    frac_delta = (initial_frac_N - 1.0)
    n_slope = frac_delta / 4.5

    n_sum = 0
    n_recid_sum = 0
    for decile_i in range(10):
        frac = n_slope * decile_i
        n_at_decile = int(round((N/10.0) * (initial_frac_N - frac)))
        n_sum += n_at_decile

        str_decile = str(decile_i+1)

        frac_recid_at_decile = decile_0_frac_recid + decile_i * (frac_recid - decile_0_frac_recid)/4.5 \
            + decile_0_recid_offset
        print('frac_recid_at_decile ' + str(decile_i) + ' : ' + str(frac_recid_at_decile))
        
        n_recid_at_decile = int(round(n_at_decile * frac_recid_at_decile))
        n_recid_sum += n_recid_at_decile
        n_norecid_at_decile = n_at_decile - n_recid_at_decile        
        
        print('decile_i: ' + str(decile_i) + ' n_at_decile: ' + str(n_at_decile) + ' n_sum: ' + str(n_sum) + ' n_recid_at_decile: ' + str(n_recid_at_decile) + ' n_norecid_at_decile: ' + str(n_norecid_at_decile) + ' sum: ' + str(n_recid_sum))

        
        for i in range(n_recid_at_decile):
            ddict = {'race': race_label,
                     'decile_score': str_decile,
                     gl_recidivism_tag: '1'}
            ddict_list.append(ddict)
        for i in range(n_norecid_at_decile):
            ddict = {'race': race_label,
                     'decile_score': str_decile,
                     gl_recidivism_tag: '0'}
            ddict_list.append(ddict)
    frac_recid = float(n_recid_sum)/n_sum
    #print('n_recid_sum: ' + str(n_recid_sum) + ' str: frac_recid: ' + str(frac_recid))
    return ddict_list, frac_recid




#Use Newton's method to find the best decile_0_recid_offset to the function,
#generateDecliningDecileSamplesLinearPrediction() in order to achieve the target_frac_recid.
def iterateGenerateDecliningDecileSamplesLinearPredictionNewton(N, race_label, target_frac_recid, decile_0_frac_recid,
                                                                initial_frac_N):
    di_frac_map = {}    
    ddict_list_0, frac_0 = generateDecliningDecileSamplesLinearPrediction(N, race_label,
                                                                          target_frac_recid,
                                                                          decile_0_frac_recid,
                                                                          0,
                                                                          initial_frac_N)
    di_frac_map[0] = frac_0
    offset = 2.0
    ddict_list, frac_oh = generateDecliningDecileSamplesLinearPrediction(N, race_label,
                                                                         target_frac_recid,
                                                                         decile_0_frac_recid,
                                                                         offset,
                                                                         initial_frac_N)
    
    deriv = (frac_oh - frac_0) / float(offset)
    offset = 0
    frac_oh = frac_0
    best_offset = offset
    best_frac_oh = frac_0
    best_ddict_list = ddict_list_0
    
    iter = 0
    while iter < 10:
        iter += 1
        last_offset = offset
        last_frac_oh = frac_oh
        delta_to_target = last_frac_oh - target_frac_recid 
        offset = last_offset - delta_to_target / float(deriv)
        
        #print('last_offset: ' + str(last_offset) + ' last_frac_oh: ' + str(last_frac_oh) + ' delta_to_target: ' + str(delta_to_target) + ' new_offset: ' + str(offset))

        if di_frac_map.get(offset) != None:
            break
        ddict_list, frac_oh = generateDecliningDecileSamplesLinearPrediction(N, race_label,
                                                                             target_frac_recid,
                                                                             decile_0_frac_recid,
                                                                             offset,
                                                                             initial_frac_N)
        if abs(target_frac_recid - frac_oh) < abs(target_frac_recid - best_frac_oh):
            best_frac_oh = frac_oh
            best_offset = offset
            best_ddict_list = ddict_list
        
        di_frac_map[offset] = frac_oh
        deriv = (frac_oh - last_frac_oh) / float(offset - last_offset)
        if abs(deriv) < .00001:
            break

    print('\ndone at ' + str(iter) + ' best_offset: ' + str(offset) + ' best_frac_oh: ' + str(best_frac_oh))

    plotRecidByDecile(best_ddict_list, 'blue', 'both', 200)


    

def f1(precision, recall):
    return 2.0 * precision * recall / (precision + recall)





#Generate a synthetic fdict_list of dict key: feature-name value: feature-value
#with features, 'decile_score' and gl_recidivism_tag.
#Returns an fdict_list, with number of decile_score and recidivism values set by the lists passed
# recid_count_by_decile_list is a list of int of length 10, one per decile score
# norecid_count_by_decile_list is a list of int of length 10, one per decile score
# multiplier can be > 1 in order to generate descrete fdicts, in case the counts are less than 1.
def generateSamplesFromRecidNorecidCounts(recid_count_by_decile_list, norecid_count_by_decile_list, multiplier=1):
    fdict_list = []
    i_decile_recid = 1
    for count_recid in recid_count_by_decile_list:
        for i in range(int(count_recid * multiplier)):
            fdict = {'decile_score':i_decile_recid,
                     gl_recidivism_tag: '1'}
            fdict_list.append(fdict)
        i_decile_recid += 1            

    i_decile_norecid = 1    
    for count_norecid in norecid_count_by_decile_list:
        for i in range(int(count_norecid * multiplier)):
            fdict = {'decile_score':i_decile_norecid,
                     gl_recidivism_tag: '0'}
            fdict_list.append(fdict)
        i_decile_norecid += 1

    return fdict_list

    


#
#
################################################################################


################################################################################
#
#models
#


import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline


#gl_default_gbr_n_estimators = 1000
#gl_default_gbr_n_estimators = 100
#gl_default_gbr_n_estimators = 50
gl_default_gbr_n_estimators = 30
#gl_default_gbr_n_estimators = 20
#gl_default_gbr_n_estimators = 10
#gl_default_gbr_n_estimators = 10
#gl_default_gbr_n_estimators = 4  
gl_default_gbr_max_depth = 3        #scikit-learn default is 3
#gl_default_gbr_max_depth = 5        #scikit-learn default is 3
gl_default_gbr_max_features = None
#gl_default_gbr_subsample = .75      #scikit-learn default is 1.0
gl_default_gbr_subsample = 1.0      #scikit-learn default is 1.0
#gl_default_gbr_subsample = .9      #scikit-learn default is 1.0

#After experimenting with 5-fold cross validation, the AUC, precision, and recall
#scores, and the optimal mcc score and threshold for getting that, I found that
#GBR parameters that are most stable and resistant to overfitting are:
#n_estimators = 30
#max_depth = 3
#subsample = 1.0



########################################
#
#turning raw data into features
#


#in the "maximal" feature set, include 'sex' and 'race' as features
#omit age_cat because we construct our own age categories
gl_features_max = ['sex', 'race', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']

#in the "race" feature set, omit 'sex' but include 'race' as a feature
#omit age_cat because we construct our own age categories
gl_features_race = ['race', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']

#in the "minimal" feature set, omit 'sex' and 'race' as features
#omit age_cat because we construct our own age categories
#This is the default for most experiments.
gl_features_min = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']


####################
#
#Titanic features
#
#Added material to build and test models on Titanic data.
#This works with a ddict-list produced by featuresTitanic.py, which does a bit of feature engineering
#for the Titanic data set.
#2020/03/21
#


#All possible values of categorical feature are listed.
#gl_feature_value_listing = OrderedDict([('Pclass', ['1', '2', '3']),                      #0  
#                                        ('Title', ['Mr', 'Mrs', 'Master', 'Miss', 'Honorific']),  #1
#                                        ('Sex', ['male', 'female']),                              #2
#                                        ('AgeCategory', ['lt1', '1', 'lt15', 'lt30', 'lt50', 'ge50']), #3
#                                        ('FamilySize', ['0', '1', '2', '3-4', 'gt4']),    #4
#                                        ('FarePerPerson', ['le10', 'le40', 'le100', 'gt100', 'Unknown']),
#                                        ('Deck', ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']), #6
#                                        ('EmbarkedFF', ['S', 'C', 'Q', 'Unknown'])])  #7
#
#The complete feature set is
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
#'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeCategory', 'FamilySize',
#'Deck', 'FarePerPerson', 'EmbarkedFF'])
#
#Raw features are
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
# 'Ticket', 'Fare', 'Cabin', 'Embarked']
#
#These are features to use according to featuresTitanic.py:
gl_features_titanic = ['Pclass', 'Title', 'Sex', 'AgeCategory', 'FamilySize', 'FarePerPerson', 'Deck', 'EmbarkedFF']
gl_features_titanic_omit_sex = ['Pclass', 'Title', 'AgeCategory', 'FamilySize', 'FarePerPerson', 'Deck', 'EmbarkedFF']


#Legacy calling the positive outcome value, "recidivism"
gl_recidivism_tag_titanic = 'Survived'


gl_feature_type_val_listing_map_titanic = {'Pclass': 'scalar',
                                           'Title': 'scalar',
                                           'Sex': 'scalar',
                                           'AgeCategory': 'scalar',
                                           'FamilySize': 'scalar',
                                           'FarePerPerson': 'scalar',
                                           'Deck': 'scalar',
                                           'EmbarkedFF': 'scalar'}


#Call this to switch between setup for Broward data vs Titanic data.
#Make sure to use the feature_list gl_features_titanic in train and test programs
def setupForDataSet(data_set_name = 'titanic'):
    global gl_recidivism_tag
    if data_set_name == 'titanic':
        gl_recidivism_tag = gl_recidivism_tag_titanic
        gl_feature_type_val_listing_map.update(gl_feature_type_val_listing_map_titanic)
    elif data_set_name == 'broward':
        gl_recidivism_tag = 'two_year_recid'
        



def extractTitanicFeatureValueToFloat(key, value):
    if key == 'Pclass':
        return getTitanicPclassFeature(value)
    elif key == 'Title':
        return getTitanicTitleFeature(value)
    elif key == 'Sex':
        return getTitanicSexFeature(value)
    elif key == 'AgeCategory':
        return getTitanicAgeCategoryFeature(value)
    elif key == 'FamilySize':
        return getTitanicFamilySizeFeature(value)
    elif key == 'FarePerPerson':
        return getTitanicFarePerPersonFeature(value)
    elif key == 'Deck':
        return getTitanicDeckFeature(value)
    elif key == 'EmbarkedFF':
        return getTitanicEmbarkedFFFeature(value)
    else:
        print('extractTitanicFeatureValueToFloat() could not interpret key: ' + str(key) + ' value: ' + str(value))
        return


    
#Note: The titanic features are already categorical due to featuresTitanic.py

##############
#
#('Pclass', ['1', '2', '3'])
#    

gl_titanic_Pclass_feature_map = {'1':0,
                                 '2':1,
                                 '3':2}

gl_titanic_Pclass_inverse_feature_map = {0:'1',
                                         1:'2',
                                         2:'3'}

def getTitanicPclassFeature(pclass):
    pclass = str(pclass)
    pclass_val = gl_titanic_Pclass_feature_map.get(pclass)
    if pclass_val == None:
        print('Problem in getTitanicPclassFeature: pclass ' + str(pclass) + ' type ' + str(type(pclass)) + ' is not recognized')
        return
    return pclass_val
    
#
#    
##############

##############
#
#('Title', ['Mr', 'Mrs', 'Master', 'Miss', 'Honorific'])
#    

gl_titanic_Title_feature_map = {'Mr':0,
                                'Mrs':1,
                                'Master':2,
                                'Miss':3,
                                'Honorific':4}

gl_titanic_Title_inverse_feature_map = {0:'Mr',
                                        1:'Mrs',
                                        2:'Master',
                                        3:'Miss',
                                        4:'Honorific'}

def getTitanicTitleFeature(title):
    title_val = gl_titanic_Title_feature_map.get(title)
    if title_val == None:
        print('Problem in getTitanicTitleFeature: title ' + title + ' is not recognized')
        return
    return title_val
    
#
#    
##############

##############
#
#('Sex', ['male', 'female'])
#

gl_titanic_Sex_feature_map = {'male':0,
                              'female':1}

gl_titanic_Sex_inverse_feature_map = {0:'male',
                                      1:'female'}

def getTitanicSexFeature(sex):
    sex_val = gl_titanic_Sex_feature_map.get(sex)
    if sex_val == None:
        print('Problem in getTitanicSexFeature: sex ' + sex + ' is not recognized')
        return
    return sex_val
    
#
#    
##############

##############
#
#('AgeCategory', ['lt1', '1', 'lt15', 'lt30', 'lt50', 'ge50'])
#

gl_titanic_AgeCategory_feature_map = {'lt1':0,
                                      '1':1,
                                      'lt15':2,
                                      'lt30':3,
                                      'lt50':4,
                                      'ge50':5}

gl_titanic_AgeCategory_inverse_feature_map = {0:'lt1',
                                              1:'1',
                                              2:'lt15',
                                              3:'lt30',
                                              4:'lt50',
                                              5:'ge50'}

def getTitanicAgeCategoryFeature(age_category):
    age_category_val = gl_titanic_AgeCategory_feature_map.get(age_category)
    if age_category_val == None:
        print('Problem in getTitanicAgeCategoryFeature: age_category ' + age_category + ' is not recognized')
        return
    return age_category_val
    
#
#    
##############

##############
#
#('FamilySize', ['0', '1', '2', '3-4', 'gt4'])
#

gl_titanic_FamilySize_feature_map = {'0':0,
                                     '1':1,
                                     '2':2,
                                     '3-4':3,
                                     'gt4':4}

gl_titanic_FamilySize_inverse_feature_map = {0:'0',
                                              1:'1',
                                              2:'2',
                                              3:'3-4',
                                              4:'gt40'}

def getTitanicFamilySizeFeature(family_size):
    family_size_val = gl_titanic_FamilySize_feature_map.get(family_size)
    if family_size_val == None:
        print('Problem in getTitanicFamilySizeFeature: family_size ' + family_size + ' is not recognized')
        return
    return family_size_val
    
#
#    
##############

##############
#
#('FarePerPerson', ['le10', 'le40', 'le100', 'gt100', 'Unknown']
#    
 
gl_titanic_FarePerPerson_feature_map = {'le10':0,
                                        'le40':1,
                                        'le100':2,
                                        'gt100':3,
                                        'Unknown':4}

gl_titanic_FarePerPerson_inverse_feature_map = {0:'le10',
                                                1:'le40',
                                                2:'le100',
                                                3:'gt100',
                                                4:'Unknown'}

def getTitanicFarePerPersonFeature(fare_per_person):
    fare_per_person_val = gl_titanic_FarePerPerson_feature_map.get(fare_per_person)
    if fare_per_person_val == None:
        print('Problem in getTitanicFarePerPersonFeature: fare_per_person ' + fare_per_person + ' is not recognized')
        return
    return fare_per_person_val
    
#
#    
##############

##############
#
#('Deck', ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown'])
#

gl_titanic_Deck_feature_map = {'A':0,
                               'B':1,
                               'C':2,
                               'D':3,
                               'E':4,
                               'F':5,
                               'T':6,
                               'G':7,
                               'Unknown':8}

gl_titanic_Deck_inverse_feature_map = {0:'A',
                                       1:'B',
                                       2:'C',
                                       3:'D',
                                       4:'E',
                                       5:'F',
                                       6:'T',
                                       7:'G',
                                       8:'Unknown'}

def getTitanicDeckFeature(deck):
    deck_val = gl_titanic_Deck_feature_map.get(deck)
    if deck_val == None:
        print('Problem in getTitanicDeckFeature: deck ' + deck + ' is not recognized')
        return
    return deck_val
    
#
#    
##############

##############
#
#('EmbarkedFF', ['S', 'C', 'Q', 'Unknown'])])
#

gl_titanic_EmbarkedFF_feature_map = {'S':0,
                                      'C':1,
                                      'Q':2,
                                      'Unknown':3}

gl_titanic_EmbarkedFF_inverse_feature_map = {0:'S',
                                              1:'C',
                                              2:'Q',
                                              3:'Unknown'}


def getTitanicEmbarkedFFFeature(embarkedff):
    embarkedff_val = gl_titanic_EmbarkedFF_feature_map.get(embarkedff)
    if embarkedff_val == None:
        print('Problem in getTitanicEmbarkedFFFeature: embarkedff ' + embarkedff + ' is not recognized')
        return
    return embarkedff_val
    
#
#    
##############
#                                       
#################### Titanic features


####################
#
#Broward and general features
#



#don't make this global, too much risk of error which switching feature sets
#key: int index in X_array
#value: str: '[feature_name]:[feature_value]'
#try:
#    gl_feature_name_value_reverse_map
#except:
#    gl_feature_name_value_reverse_map = None
    

#ddict_list is a list of key: raw_feature_name, value: raw_feature_value
#of a data set as loaded by loadDataReturnDDictList().
#Run this after you have run computeChargeDescMap()
#This function returns three values: (X, y, feature_name_value_reverse_map)
# X is list of np.array
# y is a list of float groundtruth recidivisim values  0 = norecid, 1 = recid
#These are suitable for training and testing scikit-learn models.
#feature_name_value_reverse_map is a dict:  key: int index in X_array
#                                           value: str: '[feature_name]:[feature_value]'
#The feature_name_value_reverse_map is not needed for training and testing, but
#it is used for eli5 explanation of feature weights and importances.
#This is returned along with the X_array and y_array in order to make sure that
#callers keep the feature sets and feature vectors straight.
def convertDdictsToFeatureVectorsAndGT(ddict_list, feature_list=gl_features_min):
    X_array = []    #list of np.float32
    y_array = []    #list of float 0 = norecid, 1 = recid

    fdict_list = extractDDictsAsFeatures(ddict_list, feature_list)
    fdict0 = fdict_list[0]

    for i_dict in range(len(ddict_list)):
        fdict = fdict_list[i_dict]
        ddict = ddict_list[i_dict]
        x_list = []
        for feature_name in feature_list:
            val = fdict.get(feature_name)
            if val == None:
                print('problem: val is None for feature_name: ' + feature_name)
                global gl_ddict_prob
                global gl_fdict_prob
                gl_ddict_prob = ddict
                gl_fdict_prob = fdict
                return ddict, fdict
            if type(val) is list:
                x_list.extend(val)
            else:
                x_list.append(val)
        #print('x_list: ' + str(x_list))
        x_np = np.array(x_list, dtype = np.float32)
        X_array.append(x_np)
        recid = ddict.get(gl_recidivism_tag)    # '0' or '1'
        if recid == None:
            print('Problem in convertDdictsToFeatureVectorsAndGT(), recid is None for gl_recidivism_tag: ' + gl_recidivism_tag)
            return
        y = float(recid)                 #0=norecid, 1=recid 
        y_array.append(y)

    feature_name_value_reverse_map = constructFeatureNameValueReverseMap(feature_list)
    return X_array, y_array, feature_name_value_reverse_map


#Construct a feature_name_value_reverse_map that allows lookup of
#feature names and values. This is for explanation of feature weights and importances using eli5.
#Returns a dict:  key: int index in X_array
#                 value: str: '[feature_name]:[feature_value]'
def constructFeatureNameValueReverseMap(feature_list):
    feature_name_value_reverse_map = {}

    x_index = 0
    for feature_name in feature_list:
        val_type = gl_feature_type_val_listing_map.get(feature_name)
        
        if val_type == 'scalar':
            feature_name_value_reverse_map[x_index] = feature_name
            x_index += 1
            continue
        
        if val_type == 'vector':
            for cd_i in range(len(gl_charge_index_desc_map)):
                desc_str = 'c_charge_desc:' + gl_charge_index_desc_map[cd_i]
                feature_name_value_reverse_map[x_index] = desc_str
                x_index += 1
            continue

        #drop to here if a problem
        print('Problem in constructFeatureNameValueReverseMap(): unrecognized val_type ' + str(val_type) + ' for feature_name: ' + feature_name)
        return
    return feature_name_value_reverse_map
        


#This lists the type of a feature as it is used in the feature vector, x.
#Most features are of type 'scalar', meaning that the feature takes a single float value
#which might be a categorical index.  For example, the feature juv_fel_count might take one of the
#values, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, where 5.0 means five or more juvenile felony counts.
#The c_charge_desc feature is type 'vector' because it uses many indices in the feature vector encoding.
#The 0th index of the vector portion is the average recid value for the charge description.
#The remaining indices are a one-hot array indicating which charge description it is.
gl_feature_type_val_listing_map = {'age': 'scalar',
                                   'juv_fel_count': 'scalar', 
                                   'juv_misd_count': 'scalar',
                                   'juv_other_count': 'scalar',
                                   'priors_count': 'scalar',
                                   'sex': 'scalar',
                                   'race': 'scalar',
                                   'c_charge_degree': 'scalar',
                                   'c_charge_desc': 'vector'}


#ddict_list is a list of dict: key: raw_feature_name, value: raw_feature_value
#This function selects features to use, and for each feature used, it converts
#the raw features to either float or array of float.
#The fdict_list returned is a list of dict: key: feature_name  value: float or array of float
#An array of float might be a one-hot encoding of categorical values, or even a mixture of a
#one-hot encoding and another float for that feature such as a mean value.
def extractDDictsAsFeatures(ddict_list, feature_list=gl_features_min):
    fdict_list = []
    for ddict in ddict_list:
        fdict = {}
        for key in feature_list:
            if key == 'age':
                fdict['age'] = getAgeFeature(ddict.get('age'))
            elif key == 'juv_fel_count':
                fdict['juv_fel_count'] = getJuvFelCountFeature(ddict.get('juv_fel_count'))
            elif key == 'juv_misd_count':
                fdict['juv_misd_count'] = getJuvMisdCountFeature(ddict.get('juv_misd_count'))
            elif key == 'juv_other_count':
                fdict['juv_other_count'] = getJuvOtherCountFeature(ddict.get('juv_other_count'))
            elif key == 'priors_count':
                fdict['priors_count'] = getPriorsCountFeature(ddict.get('priors_count'))
            elif key == 'c_charge_desc':
                fdict['c_charge_desc'] = getCChargeDescFeatureVector(ddict.get('c_charge_desc'))
            elif key == 'race':
                fdict['race'] = getRaceDescFeature(ddict.get('race'))
            elif key == 'sex':
                if ddict['sex'] in ('Male', 'male'):
                    fdict['sex'] = 0                 #male
                else:
                    fdict['sex'] = 1                 #female
            elif key == 'c_charge_degree':
                if ddict['c_charge_degree'] == 'M': 
                    fdict['c_charge_degree'] = 0     #misdemeanor
                else:
                    fdict['c_charge_degree'] = 1     #felony
            #not the best way to do this, but this was tacked on later when we took in Titanic data.
            elif feature_list in (gl_features_titanic, gl_features_titanic_omit_sex):
                float_val = extractTitanicFeatureValueToFloat(key, ddict.get(key))
                fdict[key] = float_val
        fdict_list.append(fdict)
    return fdict_list



#Use the following procedure to inform these feature mappings:
#import compas as co
#ddict_list = co.loadDataReturnDict()
#tt = co.tallyFieldValuesVsRecid(ddict_list, 'age')
#for item in tt:
#  item.append(float(item[1])/(item[1]+item[2]))
#tt.sort(key=lambda x: x[3])


##############
#
#age
#    

#Map int age to a categorical int that corresponds to an age range with sufficient instances that
#the age feature can interact with other features.
def getAgeFeature(age):
    age = int(age)
    if gl_age_feature_map == None:
        print('need to compute gl_age_feature_map')
        return
    age_feature_val = gl_age_feature_map.get(age)
    if age_feature_val == None:
        print('no age feature for age ' + str(age))
        return
    return age_feature_val

#These are age ranges that tend to show similar recidivism scores
# 18-22 is in the range .6 - 1.0 
# 23-33 is in the range .5 - .6
# 34-50 is in the range .4 -  .5
# 51-59 is in the range .3 -  .4
# 60+ has a lower average

#age_range            0   1   2   3   4 
gl_age_segments = [17, 22, 33, 50, 59, 100000]

def computeAgeFeatureMap():
    global gl_age_feature_map
    gl_age_feature_map = {}
    for age in range(18, 110):
        for i in range(len(gl_age_segments)):
            if age >= gl_age_segments[i]:
                gl_age_feature_map[age] = i
                continue
try:
    gl_age_feature_map   #key: int age
                         #value int age_range
except:
    gl_age_feature_map = computeAgeFeatureMap()

#
#    
##############


##############
#
#race
#Note: race is used only if the expanded feature set gl_features_max or gl_features_race are used.
#

gl_race_feature_map = {'African-American': 0,
                       'Asian': 1,
                       'Caucasian': 2,
                       'Hispanic': 3,
                       'Native American': 4,
                       'Other': 5}

gl_race_inverse_map = {0 : 'African-American',
                       1 : 'Asian',
                       2 : 'Caucasian',
                       3 : 'Hispanic',
                       4 : 'Native American',
                       5 : 'Other'}


def getRaceDescFeature(race):
    race_val = gl_race_feature_map.get(race)
    if race_val == None:
        print('Problem in getRaceDescFeature: race ' + race + ' is not recognized')
        return
    return race_val
    
#
#    
##############



##############
#
#juv_fel_count
#

#use tallyFieldValuesVsRecid(ddict_list, field_name):

#Map juv_fel_count to an int
def getJuvFelCountFeature(juv_fel_count):
    juv_fel_count = int(juv_fel_count)
    if juv_fel_count >= 5:
        return 5
    return juv_fel_count

    
#
#    
##############


##############
#
#juv_misd_count
#    

def getJuvMisdCountFeature(juv_misd_count):
    juv_misd_count = int(juv_misd_count)
    if juv_misd_count >= 4:
        return 4
    return juv_misd_count
    
#
#    
##############


##############
#
#juv_misd_count
#    

def getJuvOtherCountFeature(juv_other_count):
    juv_other_count = int(juv_other_count)
    if juv_other_count >= 4:
        return 4
    return juv_other_count
    
#
#    
##############



##############
#
#priors_count
#    

#map priors_count to an int
#  0   1   2    3    4    5
#  0   1   2  3,4,5  6,7  8+
def getPriorsCountFeature(priors_count):
    priors_count = int(priors_count)
    if priors_count <= 2:
        return priors_count
    if priors_count <= 5:
        return 3
    if priors_count <= 7:
        return 4
    return 5
    
#
#    
##############



##############
#
#c_charge_desc
#

#The feature, c_charge_desc, is based on analysis of 7214 records of ProPublica/compas data.
#There are 438 different charge descriptions, like 'Agg Batter Grt/Bod/Harm' and
#'Possession of Hydromorphone'.  57 of these are drug related, as indicated by keywords.
#Many of these are rare, and they are consolidated into a single charge description
#feature value, 'drug-related'.
#Many of the remaining charge descriptions are rare, or occur fewer than some minimum
#number of times. These are consolidated into a single charge description feature value, 'other'.
#The result is 73 charge description categories that have at least 10 distinct observations.

#Use tallyFieldValuesVsRecid() to list the counts of recid and norecid for each value.
#index 0 is a float mean value for the charge


#Returns an array of float
#This looks up the val_ar returned from gl_charge_desc_map
def getCChargeDescFeatureVector(c_charge_desc):
    val_ar = gl_charge_desc_map.get(c_charge_desc)
    if val_ar == None:
        val_ar = gl_charge_desc_map.get('other')
    return val_ar



try:
    gl_charge_desc_map          #key: str charge_desc
                                #value: list: [redid_count, norecid_count, recid_rate]
    gl_charge_index_desc_map    #key: int charge_index    index in charge_desc list, not the final X vector
                                #value str charge_descr
except:
    gl_charge_desc_map = None
    gl_charge_index_desc_map = None


#charge_desc_feature_vector is an array of float
#0 is an average recidivism rate for the selected charge description.
#The remainder of the vector is one-hot.
#This function finds which of the indices is 1, and looks up the name of the
#corresponding charge description.
#Returns a str
def getCChargeDescNameFromCChargeDescFeatureVector(charge_desc_feature_vector):
    index = None
    for i in range(1, len(charge_desc_feature_vector)):
        if charge_desc_feature_vector[i] == 1:
            index = i
            break
    if index == None:
        print('could not find a 1 in the one-hot vector')
        return
    charge_desc_name = gl_charge_index_desc_map.get(i)
    return charge_desc_name

#charge_desc_feature_vector is an array of float
#0 is an average recidivism rate for the selected charge description.
#The remainder of the vector is one-hot.
#This function finds which of the indices is 1, and returns that.
def getCChargeDescIndexFromCChargeDescFeatureVector(charge_desc_feature_vector):
    index = None
    for i in range(1, len(charge_desc_feature_vector)):
        if charge_desc_feature_vector[i] == 1:
            index = i
            return index




#Minimum number of charge incidents for the charge description to be accepted
#as a feature spot in a one-hot array of charge description values.
gl_min_charge_incidents = 10

#ddict_list is a list of dict: key: feature-name, value: feature_value
#Call computeConsolidatedChargeDescTally() to get a tally of charge descriptions
#that have suffient instances to call out separately, and consolidate others.
#This function then takes the tally and constructs a mapping of each charge description
#to an array of float.  The first value is the average recid rate for that charge desc.
#The remaining indices are for a one-hot vector of which charge desc it is.
#Returns a dict: key: charge_desc
#                value: array of float
def computeChargeDescMap(ddict_list):
    cd_tally = computeConsolidatedChargeDescTally(ddict_list)
    cd_names = list(cd_tally.keys())
    cd_names.sort()
    n_cd_names = len(cd_names)
    
    charge_desc_map = {}          #key: str charge_desc
                                  #value: list: [redid_count, norecid_count, recid_rate]
    charge_index_desc_map = {}    #key: int charge_index    index in charge_desc list, not the final X vector
                                  #value str charge_descr
    charge_index_desc_map[0] = 'charge desc average'  #average recid for the applicable charge descr 
                                  
   #value str charge_descr
    one_hot_i = 1   #reserve 0 for the average
    for cd_name in cd_names:
        tally = cd_tally.get(cd_name)
        oh_arr = [0] * (n_cd_names + 1)
        oh_arr[0] = tally[2]          #average for this charge desc, cd_name
        oh_arr[one_hot_i] = 1.0
        charge_desc_map[cd_name] = oh_arr
        charge_index_desc_map[one_hot_i] = cd_name
        one_hot_i += 1            

    global gl_charge_desc_map
    gl_charge_desc_map = charge_desc_map
    global gl_charge_index_desc_map
    gl_charge_index_desc_map = charge_index_desc_map
    print('setting gl_charge_index_desc_map')
    return 

    

#ddict_list is a list of dict: key: feature-name, value: feature_value
#Run through the values of the 'c_charge_desc' feature of the ddicts passed.
#Compile a list of the counts of recid and norecid instances, and the average.
#Returns a dict: key: charge_desc
#                value: [redid_count, norecid_count, recid_rate]
def computeConsolidatedChargeDescTally(ddict_list):
    cd_map = {}  #key: charge_desc
                 #value: [norecid_count, recid_count]
    for ddict in ddict_list:
        cd_val = ddict.get('c_charge_desc')
        if getNameInvolvesDrugs(cd_val):
            cd_val = 'drug-related'
        if cd_map.get(cd_val) == None:
            cd_map[cd_val] = [0, 0]
        is_recid = ddict.get(gl_recidivism_tag)
        if is_recid == '1':
            cd_map[cd_val][0] += 1   #recid
        else:
            cd_map[cd_val][1] += 1   #norecid

    cd_map['other'] = [1, 1]
    keys_to_delete = []
    for cd_val in cd_map.keys():
        tup = cd_map.get(cd_val)
        if tup[0] + tup[1] < gl_min_charge_incidents:
            cd_map['other'][0] += tup[0]
            cd_map['other'][1] += tup[1]
            keys_to_delete.append(cd_val)
    for key in keys_to_delete:
        del cd_map[key]
    for cd_val in cd_map.keys():
        cd_val_rtally = cd_map.get(cd_val)
        cd_val_rtally.append(float(cd_val_rtally[0])/(cd_val_rtally[0] + cd_val_rtally[1]))  #average recid rate

    return cd_map



gl_drug_name_set = set(['cocaine', 'cannabis', 'methylenedioxymethcath', 'alprazolam', 'pyrrolidinovalerophenone',
                        'methamphetamine', 'heroin', 'oxycodone', 'contr', 'contraband', 'schedule', 'narcotic',
                        'narcotics', 'drug', 'methylenediox', 'hydromorphone', 'hydrocodone', 'mdma', 'ecstasy',
                        'controlled', 'paraphernalia', 'marijuana', 'amphetamine', 'tetrahydrocannabinols',
                        'thc', 'morphine', 'ethylone', 'methylethcathinone', 'codeine', 'carisoprodol', 
                        'benzylpiperazine', 'drugs', 'meth', 'diox', 'lsd', 'fluoro', 'cont', 'contr', 'control',
                        'methox', 'pentyl', 'lorazepam', 'buprenorphine', 'trifluoromethylphenylpipe',
                        'trifluoromethylphenylpiperazine', 'butylone', 'diazepam', 'tobacco',
                        'phentermine', 'methylenediox', 'pyrrolidinovalerophenone', 'methylenedioxymethcath',
                        'butanediol', 'clonazepam', 'amobarbital', 'fentanyl', 'pyrrolidinobutiophenone',
                        'methadone', 'anabolic', 'steroid', 'xlr11'])


            
def getNameInvolvesDrugs(name):
    name = name.lower()
    name = name.replace('-', ' ')
    name = name.replace('/', ' ')    
    name = name.replace('(', ' ')
    name = name.replace(')', ' ')
    name = name.replace('3', ' ')
    name = name.replace('2', ' ')
    name = name.replace('4', ' ')
    name = name.replace('5', ' ')
    name = name.replace('6', ' ')
    name_tokens = name.split()
    for token in name_tokens:
        if token in gl_drug_name_set:
            return True
    return False




#For each charge description feature, tally the breakdown by race.
#fdict_list is a list of dict: key: feature_name, value: feature_value,
#which has been processed from raw features ddict_list by
#convertDdictsToFeatureVectorsAndGT()
#fdict_list should include a 'race' feature
#the value of the 'c_charge_desc' feature 
# 
def assessChargeDescFeatureDistributionByRace(fdict_list, tol = .2):
    charge_desc_race_tally_map = {} #key: str charge-description
                                     #value: list of int, count, indexed by race index in
                                     #gl_race_feature_map
    for fdict in fdict_list:
        charge_desc_feature_vector = fdict.get('c_charge_desc')
        charge_desc_index = getCChargeDescIndexFromCChargeDescFeatureVector(charge_desc_feature_vector)
        if charge_desc_index == None:
            print('problem: charge_desc_index is None for fdict')
            return fdict
        race_index = fdict.get('race')
        if race_index == None:
            print('problem: race_index is None for fdict')
            return fdict

        if charge_desc_race_tally_map.get(charge_desc_index) == None:
            charge_desc_race_tally_map[charge_desc_index] = [0] * len(gl_race_feature_map)
        charge_desc_race_tally_map[charge_desc_index][race_index] += 1

    displayChargeDescrDistributionByRace(charge_desc_race_tally_map)


    outlier_charge_stats_tally_map = identifyOutlierChargeDescItem(fdict_list, charge_desc_race_tally_map, tol)
    print('\n\noutliers: ' + str(len(outlier_charge_stats_tally_map)))
    displayChargeDescrDistributionByRace(outlier_charge_stats_tally_map)
    return outlier_charge_stats_tally_map


#Run through the dictionary of charge descriptions and count breakdowns by race
#Pull out the ones that have a fraction of +/- frac_tolerance for either
#'African-American' or 'Caucasian', with respect to fraction of that race
#in the population
#fdict_list is a dict key: feature-name value: feature-value
#charge_desc_race_tally_map is a dict:  key:   int charge_desc_index
#                                       value: list of int count, indexed by race_index
#returns a subset of the charge_desc_race_tally_map passed
def identifyOutlierChargeDescItem(fdict_list, charge_desc_race_tally_map, tol = .2):
    cau_count = 0
    aa_count = 0
    for fdict in fdict_list:
        if fdict.get('race') == gl_race_feature_map.get('African-American'):
            aa_count += 1
        elif fdict.get('race') == gl_race_feature_map.get('Caucasian'):
            cau_count += 1
    overall_aa_frac = float(aa_count)/len(fdict_list)
    overall_cau_frac = float(cau_count)/len(fdict_list)
    print('overall_aa_frac: ' + str(overall_aa_frac))
    print('overall_cau_frac: ' + str(overall_cau_frac))

    outlier_charge_desc_race_tally_map = {}

    for charge_index in charge_desc_race_tally_map:
        race_distr = charge_desc_race_tally_map.get(charge_index)
        count_sum = float(sum(race_distr))
        aa_frac = race_distr[gl_race_feature_map.get('African-American')] / count_sum
        cau_frac = race_distr[gl_race_feature_map.get('Caucasian')] / count_sum
        if abs(aa_frac - overall_aa_frac) > tol or\
           abs(cau_frac - overall_cau_frac) > tol:
            #if abs(aa_frac - overall_aa_frac) > tol:
            #    print('outlier aa_frac: ' + str(aa_frac) + ' overall: ' + str(overall_aa_frac) + ' absdiff: ' + str(abs(aa_frac-overall_aa_frac)))
            #if abs(cau_frac - overall_cau_frac) > tol:
            #    print('outlier cau_frac: ' + str(cau_frac) + ' overall: ' + str(overall_cau_frac) + ' absdiff: ' + str(abs(cau_frac-overall_cau_frac)))
            outlier_charge_desc_race_tally_map[charge_index] = race_distr
    return outlier_charge_desc_race_tally_map
        


def displayChargeDescrDistributionByRace(charge_desc_race_tally_map):
    charge_index_list = list(charge_desc_race_tally_map.keys())
    charge_index_list.sort()
    aa_cau_race_index_list = []
    aa_cau_race_index_list.append(gl_race_feature_map.get('African-American'))
    aa_cau_race_index_list.append(gl_race_feature_map.get('Caucasian'))
    for charge_index in charge_index_list:
        charge_name = gl_charge_index_desc_map.get(charge_index)

        race_distr = charge_desc_race_tally_map.get(charge_index)
        if race_distr == None:
            print('blank: ' + str(charge_index))
            continue
        count_sum = sum(race_distr)
        ratio_distr = []
        race_count_str = ''
        ratio_distr_str = ''
        #for i_race in range(len(race_distr)):
        for i_race in aa_cau_race_index_list:
            ratio = float(race_distr[i_race]) / count_sum
            race_count_str += '{:4}'.format(race_distr[i_race])
            ratio_distr_str += '  {:.2f}'.format(ratio)
            
        print(str(charge_index) + '  {:50}'.format(charge_name) + ' ' + race_count_str + '   ' + ratio_distr_str)
    

#
#
#################### Broward
#
############################



############################
#
#Generate readable descriptions of feature categories and values, for ELI5 explanations.
#

#dict key: feature_name:
#     value: dict: key: feature_value
#                  value: str description-string

gl_feature_name_value_str_map = {'age': {0:'18-22',
                                         1:'23-33',
                                         2:'34-50',
                                         3:'51-59',
                                         4:'60+'},
                                 'juv_fel_count': {0:'0',
                                                   1:'1',
                                                   2:'2',
                                                   3:'3',
                                                   4:'4',
                                                   5:'5+'},
                                 'juv_misd_count': {0:'0',
                                                     1:'1',
                                                     2:'2',
                                                     3:'3',
                                                     4:'4+'},
                                 'juv_other_count': {0:'0',
                                                     1:'1',
                                                     2:'2',
                                                     3:'3',
                                                     4:'4+'},                                                     
                                 'priors_count': {0:'0',
                                                  1:'1',
                                                  2:'2',
                                                  3:'3-5',
                                                  4:'6 or 7',
                                                  5:'8+'},
                                 'race': gl_race_inverse_map,
                                 'sex': {0:'Male',
                                         1:'Female'},
                                 'c_charge_degree':{0:'Misdemeanor',
                                                    1:'Felony'}}
    

#This is used to generate a readable description of a feature value for ELI5 explanations.
#This is essentially the inverse of the various feature mappers above.
#
#feature_name_value and value can be like one of the following
# categorical feature and value:                                      'age'  35
# the average value of a c_charge_desc:                               'c_charge_desc:charge desc average'  .455
# the boolean truth value a particular one-hot charge description:    'c_charge_desc:'Agg Batter Grt/Bod/Harm'  1.0
#Returns two str values: feature_name_str, str feature_value_str
def generateFeatureValueReadableStringElements(feature_name, feature_value):
    colon_index = feature_name.find(':')
    if colon_index < 0:
        int_fv = int(feature_value)
        feature_value_str = gl_feature_name_value_str_map.get(feature_name).get(int_fv)
        return feature_name, feature_value_str

    #feature_name will be like 'c_charge_desc:charge desc average' or else 'c_charge_desc:'Agg Batter Grt/Bod/Harm'
    str_feature_name = feature_name[:colon_index]
    str_fv = feature_name[colon_index+1:]
    if str_fv == 'charge desc average':
        str_feature_name += ' charge desc average'
        str_fv = '{:.2f}'.format(feature_value)
        return str_feature_name, str_fv

    #feature_name will be like 'c_charge_desc:'Agg Batter Grt/Bod/Harm'  and feature_value will be 0.0 or 1.0
    if feature_value == 0.0:
        str_truth_value = 'False'
    else:
        str_truth_value = 'True'
        
    str_feature_name += ' : ' + str_fv
    return str_feature_name, str_truth_value
    
#
#    
##############
    
##############
#
#
#

#Constuct a deep copy of the data ddict_list, then replace feature values as indicated.
#ddict_list is a list of dict: key: feature-name, value: feature_value
#feature_replacement_list is a list of tuple: (str feature-name, str feature-value),
def copyDDictsReplaceFeatureValues(ddict_list, feature_replacement_list):
        
    ddict_list_repl = []
    for ddict in ddict_list:
        ddict_repl = ddict.copy()
        for fv_tup in feature_replacement_list:
            feature_name = fv_tup[0]
            feature_value = fv_tup[1]
            ddict_repl[feature_name] = feature_value
        ddict_list_repl.append(ddict_repl)
    return ddict_list_repl


#
#    
##############
#
############################






############################
#
#GradientBoostingRegressor model
#


def trainGBR(X_train, y_train, n_estimators=None, max_depth=None, max_features=None, subsample=None):

    if n_estimators == None:
        gbr_n_estimators = gl_default_gbr_n_estimators
    else:
        gbr_n_estimators = n_estimators
    if max_depth == None:
        gbr_max_depth = gl_default_gbr_max_depth
    else:
        gbr_max_depth = max_depth
    if max_features == None:
        gbr_max_features = gl_default_gbr_max_features
    else:
        gbr_max_features = max_features
    if subsample == None:
        gbr_subsample = gl_default_gbr_subsample
    else:
        gbr_subsample = subsample

    gbr = GradientBoostingRegressor(n_estimators=gbr_n_estimators, max_depth=gbr_max_depth,
                                    max_features=gbr_max_features, subsample=gbr_subsample)

    gbr.fit(X_train, y_train)

    return gbr


    
#
#
############################


############################
#
#Linear Regression
#


def trainLR(X_train, y_train, scale=False):

    #linr = LinearRegression(normalize = True)

    linr = LinearRegression()    
    if scale:
        scaler = preprocessing.StandardScaler()  #with_std=False)
        global gl_last_scaler
        gl_last_scaler = scaler
        linr_pipeline = make_pipeline(scaler, linr)
        linr_pipeline.fit(X_train, y_train)
        return linr_pipeline                
    else:
        linr.fit(X_train, y_train)
        return linr

    
#
#
############################


############################
#
#Logistic Regression
#


    
def trainLogR(X_train, y_train):
    global gl_last_scaler

    #print('in trainLogR')
    
    scaler = preprocessing.StandardScaler()
    logr = LogisticRegression()
    logr_pipeline = make_pipeline(scaler, logr)

    logr_pipeline.fit(X_train, y_train)

    return logr_pipeline

    
    
#
#
############################



############################
#
#
#
    
def testModel(X_test, model):
    if str(type(model)) == "<class 'sklearn.pipeline.Pipeline'>":
        if len(model.steps) > 1 and model.steps[1][0] == 'logisticregression':
            probs_ar = model.predict_proba(X_test)
            prob_1_ar = [item[1] for item in probs_ar]
            #print('min_prob: ' + str(min(prob_1_ar)))
            return prob_1_ar
    return model.predict(X_test)

    
#
#
############################


############################
#
#Plotting for a GBR or Logistic or Linear model
#


#For plotting regressor prediction scores.
#If the training data ranges from 0.0 to 1.0, then these bounds give room for
#predictions below and above these values.
#Often though we reset these to 0.0 and 1.0.
try:
    gl_score_plot_range_min
except:
    gl_score_plot_range_min = -.25
    gl_score_plot_range_max = 1.25



#y is a list of float groundtruth recidivism,  0.0 = norecid, 1.0 = recid
#pred_scores is a list of float prediction scores in synch with y.
#Generally, pred_scores will be in the range [-.25, 1.25] for Linear, [0, 1] for Logistic and GBR.
#This plots two curves, the prediction score density y=0.0 and y=1.0, in different colors.
def plotPredictionScores(y, pred_scores, ratio_color=None, ymax = 700, show_plots_p=True):
    if len(y) != len(pred_scores):
        print('y count ' + str(len(y)) + ' != pred_scores count ' + str(len(pred_scores)))
        return

    range_min = gl_score_plot_range_min
    range_max = gl_score_plot_range_max
    the_range = range_max - range_min
    n_bins = 20    
    #n_bins = 30    
    #n_bins = 40
    #n_bins = 50
    #n_bins = 60
    #n_bins = 100
    hist_arrays = []
    for i in range(3):
        hist_arrays.append([0] * n_bins)

    for i in range(len(y)):
        y_i = y[i]
        y_index = int(y_i)       #0 = norecid, 1 = recid
        hist_array = hist_arrays[y_index]  #which hist_array to add this count to, recid or norecid
        pred = pred_scores[i]
        hist_index = int(max(0.0, min(1.0, (pred - range_min)/the_range)) * n_bins)
        hist_index = max(0, min(n_bins-1, hist_index))
        hist_array[hist_index] += 1
        hist_arrays[2][hist_index] += 1  #sum of recid and norecid

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, ymax)

    #ratio array
    ratio_array = []    
    for i in range(n_bins):
        denom = (hist_arrays[0][i] + hist_arrays[1][i])
        if denom == 0:
            ratio = 0
        else:
            ratio = float(hist_arrays[1][i]) / denom * ymax
        ratio_array.append(ratio)

    if show_plots_p:
        ax1.plot(hist_arrays[0], color='green')
        ax1.plot(hist_arrays[1], color='red')
        ax1.plot(hist_arrays[2], color='lightsteelblue')        


    print(str(ratio_array))
    if ratio_color != None:
        ind = np.arange(n_bins)
        p3 = plt.plot(ind, ratio_array, color=ratio_color)


    #stackoverflow...
    ticks_x = mpl_ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/n_bins * the_range + range_min))
    ax1.xaxis.set_major_formatter(ticks_x)
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=1.25))    #10 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=2.5))   #20 bins, -5. - 1.5

    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=3.333333333333))   #20 bins   -.25 - 1.25
    locs = [-.25, 0, .25, .5, .75, 1.0, 1.25]
    #locs = [-.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2]
    locs = [ loc * n_bins / the_range + 5/the_range for loc in locs]
    ax1.xaxis.set_major_locator(mpl_ticker.FixedLocator(locs))
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=3.75))  #30 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=5.0))   #40 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=6.25))  #50 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=7.5))   #60 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=12.5))   #100 bins    
    #ax1.xaxis.set_minor_formatter(ticks_x)

    #ax1.grid(b=True)
    plt.show()
    


#returns a list of list of int [pos_counts, neg_counts]
def predictionScoresToHistArrays(y, pred_scores, n_bins=20):
    if len(y) != len(pred_scores):
        print('y count ' + str(len(y)) + ' != pred_scores count ' + str(len(pred_scores)))
        return

    range_min = gl_score_plot_range_min
    range_max = gl_score_plot_range_max
    the_range = range_max - range_min
    #n_bins = 20    
    #n_bins = 30    
    #n_bins = 40
    #n_bins = 50
    #n_bins = 60
    #n_bins = 100
    hist_arrays = []
    for i in range(2):
        hist_arrays.append([0] * n_bins)

    for i in range(len(y)):
        y_i = y[i]
        y_index = int(y_i)       #0 = norecid, 1 = recid
        hist_array = hist_arrays[y_index]  #which hist_array to add this count to, recid or norecid
        pred = pred_scores[i]
        hist_index = int(max(0.0, min(1.0, (pred - range_min)/the_range)) * n_bins)
        hist_index = max(0, min(n_bins-1, hist_index))
        hist_array[hist_index] += 1

    hist_arrays.reverse()  #because we stuffed them [neg_counts, pos_counts] 
    return hist_arrays

        
    


#y is a list of float groundtruth recidivism,  0.0 = norecid, 1.0 = recid
#pred_scores is a list of float prediction scores in synch with y.
#Generally, pred_scores will be in the range [-.25, 1.25].
#This tallies TP, FP, TN, FN bars into deciles by prediction score, and
#displays stacked bars in the manner of displaying COMPAS predictions.
def plotPredictionScoresAsStackedDecileBars(y, pred_scores, ratio_color=None, ymax = 700):
                                            
    if len(y) != len(pred_scores):
        print('y count ' + str(len(y)) + ' != pred_scores count ' + str(len(pred_scores)))
        return

    pred_min = min(pred_scores)
    pred_max = max(pred_scores)
    range_min = pred_min
    range_max = pred_max
    the_range = range_max - range_min
    n_bins = 10
    #n_bins = 20    
    #n_bins = 30    
    #n_bins = 40
    #n_bins = 50
    #n_bins = 60
    #n_bins = 100
    hist_arrays = []
    for i in range(3):
        hist_arrays.append([0] * n_bins)

    for i in range(len(y)):
        y_i = y[i]
        y_index = int(y_i)       #0 = norecid, 1 = recid
        hist_array = hist_arrays[y_index]  #which hist_array to add this count to, recid or norecid
        pred = pred_scores[i]
        hist_index = int(max(0.0, min(1.0, (pred - range_min)/the_range)) * n_bins)
        hist_index = max(0, min(n_bins-1, hist_index))
        hist_array[hist_index] += 1
        hist_arrays[2][hist_index] += 1  #sum of recid and norecid

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, ymax)

    #ratio array
    ratio_array = []    
    for i in range(n_bins):
        denom = (hist_arrays[0][i] + hist_arrays[1][i])
        if denom == 0:
            ratio = 0
        else:
            ratio = float(hist_arrays[1][i]) / denom * ymax
        ratio_array.append(ratio)

    ind = np.arange(10)
    width = 0.35    
    recid_hist = hist_arrays[1]
    norecid_hist = hist_arrays[0]
    p1 = ax1.bar(ind, recid_hist, width, color='red')
    p2 = ax1.bar(ind, norecid_hist, width, bottom=recid_hist, color='green')        
    #ax1.plot(hist_arrays[0], color='green')
    #print(str(hist_arrays[1]))
    #ax1.plot(hist_arrays[1], color='red')
    #print(str(hist_arrays[2]))
    ax1.plot(hist_arrays[2], color='lightsteelblue')

    if ratio_color != None:
        ind = np.arange(n_bins)
        p3 = plt.plot(ind, ratio_array, color=ratio_color)

    #Big trial and error pain to get the number and ticks to look good.
    #stackoverflow...
    #ticks_x = mpl_ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/n_bins * the_range + range_min))
    #ax1.xaxis.set_major_formatter(ticks_x)
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=1.25))    #10 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=2.5))   #20 bins, -5. - 1.5

    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=3.333333333333))   #20 bins   -.25 - 1.25
    #locs = [-.25, 0, .25, .5, .75, 1.0, 1.25]
    #locs = [-.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2]
    #locs = [ loc * n_bins / the_range + 5/the_range for loc in locs]
    #ax1.xaxis.set_major_locator(mpl_ticker.FixedLocator(locs))
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=3.75))  #30 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=5.0))   #40 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=6.25))  #50 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=7.5))   #60 bins
    #ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(base=12.5))   #100 bins    
    #ax1.xaxis.set_minor_formatter(ticks_x)

    #ax1.grid(b=True)
    plt.show()
    
    

    

#y is a list of float groundtruth recidivism,  0.0 = norecid, 1.0 = recid
#pred_scores is a list of float prediction scores in synch with y.
#Generally, pred_scores will be in the range [-.5, 1.5] or [-.25, 1.25], or [0, 1].
#This builds a table of prob recid vs prediction score, in the same manner as
#the function, plotPredictionScores() above tallies counts of recid and norecid across
#a histogram of prediction scores.
#Returns three values:   array of float, score_offset, score_scale
# where the estimated prob of recidivism is  array[pred_score_scale * (pred_score - offset)]
def buildProbRecidVsScoreTable(y, pred_scores):
    if len(y) != len(pred_scores):
        print('y count ' + str(len(y)) + ' != pred_scores count ' + str(len(pred_scores)))
        return

    range_min = gl_score_plot_range_min
    range_max = gl_score_plot_range_max
    the_range = range_max - range_min
    n_bins = 30
    hist_arrays = []
    for i in range(2):
        hist_arrays.append([0] * n_bins)

    for i in range(len(y)):
        y_i = y[i]
        y_index = int(y_i)    #0 = norecid, 1 = recid
        #print('label: ' + str(label) + ' label_index: ' + str(label_index))
        hist_array = hist_arrays[y_index]  #which hist_array to add this count to, recid or norecid
        pred = pred_scores[i]
        hist_index = int(max(0.0, min(1.0, (pred - range_min)/the_range)) * n_bins)
        hist_index = max(0, min(n_bins-1, hist_index))
        hist_array[hist_index] += 1

    pred_prob_array = np.zeros(n_bins)
    for bin_i in range(n_bins):
        n_recid = hist_arrays[1][bin_i]
        n_total = hist_arrays[0][bin_i] + hist_arrays[1][bin_i]
        if n_total == 0:
            if bin_i > n_bins/2.0:
                prob_recid = 1.0
            else:
                prob_recid = 0.0
        else:
            prob_recid = float(n_recid)/n_total
        pred_prob_array[bin_i] = prob_recid

    pred_score_offset = range_min
    pred_score_scale = float(n_bins)/the_range
    return pred_prob_array, pred_score_offset, pred_score_scale


def lookupTableValue(table_ar, offset, scale, pred_score):
    bin_i = int((pred_score - offset) * scale)
    bin_i = max(0, min(bin_i, len(table_ar)-1))
    return table_ar[bin_i]

def plotTable(table_ar, offset, scale):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    n_bins = len(table_ar)
    range_min = offset
    the_range = n_bins/scale
    
    ax1.plot(table_ar)

    #stackoverflow...
    ticks_x = mpl_ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/n_bins * the_range + range_min))
    ax1.xaxis.set_major_formatter(ticks_x)
                                       
    plt.show()




#Build an ROC (Reciver Operating Characteristic) curve of Recall vs False Positive Rate.
#y_list is a list of groundtruth values, 0 = norecid, 1 = recid
#pred_list is a list of float prediction scores.
#The ROC curve essentially sweeps across the prediction scores as possible thresholds,
#from highest prediction score to lowest, 
#and accumulates False Positive rate and Recall rate along the way.
#  FPR =  FP / (FP + TN)
#  Recall = TP / (TP + FN)
#Returns an ROC curve in the form of a tuple: (list-of-float: fpr_list, list-of-float: recall_list)
#Note that the spacing of False Postive Rate in fpr_list will not be even.
#Note that in the event that many prediction scores are identical, then the samples in y_list will
#come in random bunches of 1 or 0. This will cause the ROC curve to be wiggly.
def buildROCCurveFromPredictionScores(y_list, pred_list):
    if len(y_list) != len(pred_list):
        print('y_list ' + str(len(y_list)) + ' needs to be the same length as pred_list ' + str(len_pred_list))
        return
    y_pred_tup_list = []
    for i in range(len(y_list)):
        y = y_list[i]
        pred = pred_list[i]
        y_pred_tup_list.append((y, pred))
        
    #sort by prediction score
    y_pred_tup_list.sort(key=lambda x: x[1], reverse=True)  #sweep from highest prediction score to lowest,
                                                            #catching more True Postives and False Positives as we go.
                                                            
    recid_total = sum(y_list)
    norecid_total = len(y_list) - recid_total

    fpr_by_increment = [0]
    recall_by_increment= [0]

    cum_fp = 0
    cum_tn = norecid_total
    cum_tp = 0
    cum_fn = recid_total
    for y_pred_tup in y_pred_tup_list:
        y = y_pred_tup[0]
        if y == 0:
            cum_tn -= 1
            cum_fp += 1
        else:
            cum_fn -= 1
            cum_tp += 1

        fpr_i = float(cum_fp) / (cum_fp + cum_tn)
        recall_i = float(cum_tp) / (cum_tp + cum_fn)
        fpr_by_increment.append(fpr_i)
        recall_by_increment.append(recall_i)
        
    print('AUC: ' + str(computeROC_AUC((fpr_by_increment, recall_by_increment))))
    return (fpr_by_increment, recall_by_increment)



#Returns three values:
# confusion_matrix, f1_score, mcc_score
# https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
def buildConfusionMatrixFromPredictionScores(y_list, pred_list, score_threshold, print_p=True):
    if len(y_list) != len(pred_list):
        print('y_list ' + str(len(y_list)) + ' needs to be the same length as pred_list ' + str(len(pred_list)))
        return

    n_norecid_low = 0
    n_norecid_high = 0
    n_recid_low = 0
    n_recid_high = 0

    for i in range(len(y_list)):
        y = y_list[i]
        pred = pred_list[i]

        if pred <= score_threshold and y == 0:
            n_norecid_low += 1
        elif pred <= score_threshold and y == 1:
            n_recid_low += 1            
        elif pred > score_threshold and y == 0:
            n_norecid_high += 1            
        elif pred > score_threshold and y == 1:
            n_recid_high += 1            

    confusion_matrix = []
    confusion_matrix.append([n_norecid_low, n_norecid_high])
    confusion_matrix.append([n_recid_low, n_recid_high])
    count = sum(confusion_matrix[0]) + sum(confusion_matrix[1])
    if count != len(y_list):
        print('problem in count')

    tn = confusion_matrix[0][0]
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    TPR = float(tp) / (tp + fn)     #recall
    if tp+fp > 0.0:
        PPV = float(tp) / (tp + fp)     #precision
    else:
        PPV = 0.0
    if TPR + PPV > 0:
        f1 = 2*TPR*PPV / (TPR + PPV)
    else:
        f1 = 0.0
    FPR = float(fp) / (fp + tn)
    FNR = float(fn) / (fn + tp)
    TNR = float(tn) / (tn + fp)
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denom == 0:
        mcc = 0.0
    else:
        mcc = (tp*tn - fp*fn) / denom
    recid_ratio = float(sum(confusion_matrix[1])) / ( sum(confusion_matrix[0]) + sum(confusion_matrix[1]))
    if print_p:
        print('count: ' + str(count))        
        print(str(confusion_matrix[0]))
        print(str(confusion_matrix[1]))
        print('FPR: ' + str(FPR))
        print('FNR: ' + str(FNR))
        print('TNR: ' + str(TNR))
        print('TPR = recall: ' + str(TPR))
        print('PPV = precision: ' + str(PPV))
        print('f1: ' + str(f1))
        print('mcc: ' + str(mcc))
        print('recidivism ratio: ' + str(recid_ratio))
    return confusion_matrix, f1, mcc



def chooseOptimalF1Score(y_list, pred_list):
    f1_max = 0.0
    thr_max = 0.0
    thr_last = 0.0
    for divisor in (10.0, 100.0, 1000.0):
        for i in range(10):
            thr = thr_last + float(i)/divisor
            cm, f1, mcc = buildConfusionMatrixFromPredictionScores(y_list, pred_list, thr, False)
            if f1 > f1_max:
                thr_max = thr
                f1_max = f1
        thr_last = thr_max - .5/divisor
        cm, f1_max, mcc_max = buildConfusionMatrixFromPredictionScores(y_list, pred_list, thr_last, False)        
        #print('thr_max: ' + str(thr_max) + ' thr_last: ' + str(thr_last))
    return thr_max, f1_max


#MCC score is said to do better than f1 score for imbalanced data.
#From this data set, I agree with that.
# https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
def chooseOptimalMCCScore(y_list, pred_list):
    mcc_max = -100000
    thr_max = 0.0
    thr_last = 0.0
    for divisor in (10.0, 100.0, 1000.0):
        for i in range(10):
            thr = thr_last + float(i)/divisor
            cm, f1, mcc = buildConfusionMatrixFromPredictionScores(y_list, pred_list, thr, False)
            if mcc > mcc_max:
                thr_max = thr
                mcc_max = mcc
        thr_last = thr_max - .5/divisor
        cm, f1_max, mcc_max = buildConfusionMatrixFromPredictionScores(y_list, pred_list, thr_last, False)        
        #print('thr_max: ' + str(thr_max) + ' thr_last: ' + str(thr_last))        
    return thr_max, mcc_max




#
#
############################

############################
#
#ELI5 feature importances (for a GBR model only)
#

import eli5
#This import triggers a warning:
#C:\Python36x64\lib\site-packages\sklearn\utils\deprecation.py:144: FutureWarning: The sklearn.feature_selection.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.feature_selection. Anything that cannot be imported from sklearn.feature_selection is now part of the private API.



def explainGBRGlobalFeatureWeights_ELI5(gbr, feature_name_value_reverse_map):
    global gl_weights_explain_eli5
    gl_weights_explain_eli5 = eli5.explain_weights(gbr, top=1000)   #, top=1000)
    weights_explain_dict = eli5.formatters.format_as_dict(gl_weights_explain_eli5)
    importances_list = weights_explain_dict.get('feature_importances').get('importances')
    feature_name_value_importance_list = []  #(feature_name, feature_value, importance)
    for item in importances_list:  #item is a dict: {feature: 'Column_92', 'weight': .33, ...}
        col_name = item.get('feature')
        col_index = int(col_name[len('x'):])
        feature_name_value = feature_name_value_reverse_map.get(col_index)
        if feature_name_value == None:
            print('problem: no feature_name_value found for col_index: ' + str(col_index))
            return
        weight = item.get('weight')
        feature_name_value_importance_list.append((feature_name_value, weight))

    feature_name_value_importance_list.sort(key = lambda x: x[1], reverse=True)
    for item in feature_name_value_importance_list:
        feature_name_value = item[0]
        weight = item[1]
        print('{:50}'.format(feature_name_value) + '   {:.3f}'.format(weight))


#for Logistic Regresssion
def explainLogRGlobalFeatureWeights_ELI5(model, feature_name_value_reverse_map):
    global gl_weights_explain_eli5
    gl_weights_explain_eli5 = eli5.explain_weights(model, top=1000)   #, top=1000)
    weights_explain_dict = eli5.formatters.format_as_dict(gl_weights_explain_eli5)
    
    #if str(type(model)) == "<class 'sklearn.pipeline.Pipeline'>":
    #   model.steps[1][0] == 'logisticregression':
    #    feature_weights_dict = weights_explain_dict.get('targets')[0].get('feature_weights')
    #elif str(type(model)) == "<class 'sklearn.pipeline.Pipeline'>" and \
    #   model.steps[1][0] == 'linearregression':
    #    feature_weights_dict = weights_explain_dict.get('targets')[0].get('feature_weights')
    #else:
    #    print('did not see this as a logisticregression pipeline')
    #    return

    feature_weights_dict = weights_explain_dict.get('targets')[0].get('feature_weights')
    
    #print('\nPOS')
    importances_list = feature_weights_dict.get('pos')
    feature_name_value_importance_list = []  #(feature_name, feature_value, importance)
    for item in importances_list:  #item is a dict: {feature: 'Column_92', 'weight': .33, ...}
        col_name = item.get('feature')
        if col_name == '<BIAS>':
            feature_name_value = 'BIAS'
        else:
            col_index = int(col_name[len('x'):])
            feature_name_value = feature_name_value_reverse_map.get(col_index)
            if feature_name_value == None:
                print('problem: no feature_name_value found for col_index: ' + str(col_index))
                return
        weight = item.get('weight')
        feature_name_value_importance_list.append((feature_name_value, weight))

    #feature_name_value_importance_list.sort(key = lambda x: x[1], reverse=True)
    #for item in feature_name_value_importance_list:
    #    feature_name_value = item[0]
    #    weight = item[1]
    #    print('{:50}'.format(feature_name_value) + '   {:.3f}'.format(weight))

    #print('\nNEG')
    importances_list = feature_weights_dict.get('neg')
    global gl_importances_list
    gl_importances_list = importances_list
    #feature_name_value_importance_list = []  #(feature_name, feature_value, importance)
    for item in importances_list:  #item is a dict: {feature: 'Column_92', 'weight': .33, ...}
        col_name = item.get('feature')
        if col_name == '<BIAS>':
            feature_name_value = 'BIAS'
        else:
            col_index = int(col_name[len('x'):])
            feature_name_value = feature_name_value_reverse_map.get(col_index)
            if feature_name_value == None:
                print('problem: no feature_name_value found for col_index: ' + str(col_index))
                return
        weight = item.get('weight')
        feature_name_value_importance_list.append((feature_name_value, weight))

    #feature_name_value_importance_list.sort(key = lambda x: x[1], reverse=True)
    #for item in feature_name_value_importance_list:
    #    feature_name_value = item[0]
    #    weight = item[1]
    #    print('{:50}'.format(feature_name_value) + '   {:.3f}'.format(weight))


    feature_name_value_importance_list.sort(key = lambda x: abs(x[1]), reverse=True)
    for item in feature_name_value_importance_list:
        feature_name_value = item[0]
        weight = item[1]
        print('{:50}'.format(feature_name_value) + '   {:.3f}'.format(weight))        
        



        
        
    



    
#Call ELI5 explain_prediction to tell the feature importances for the prediction on the ddict.
#ddict describes one case: dict: key: feature-name, value: feature-value
#gbr is a Gradient Boosting Regressor model.
#Returns two lists:
# significant_feature_value_listing_pos, significant_feature_value_listing_neg
#Each list is a list of tuple:  (float feature_contribution, string feature_name, string feature_value)
def testFeatureContributionsToPrediction_SelectMostImportantFeatures_ELI5(gbr, ddict,
                                                                          feature_list,
                                                                          abs_contribution_threshold = .05,
                                                                          score_prob_table = None,
                                                                          print_p = True):
    X, y, feature_name_value_reverse_map = convertDdictsToFeatureVectorsAndGT([ddict], feature_list)
    x = X[0]
    pred_explain_eli5 = eli5.explain_prediction(gbr, x)

    pred_explain_eli5_dict = eli5.formatters.as_dict.format_as_dict(pred_explain_eli5)
    #return pred_explain_eli5_dict

    feature_weights_dict = pred_explain_eli5_dict.get('targets')[0].get('feature_weights')
    significant_feature_value_listing_pos = []  #list of tuple: (float: contribution,
                                                #   string: feature_name, string: feature_value)
    significant_feature_value_listing_neg = []  #list of tuple: (float: contribution,
                                                #   string: feature_name, string: feature_value)
    features_l_pos = feature_weights_dict.get('neg')  #dict {'weight':..., 'feature': int index,
                                                      #      'value': int index}
    features_l_neg = feature_weights_dict.get('pos')
    features_l = features_l_pos[:]
    features_l.extend(features_l_neg)
    score = pred_explain_eli5_dict.get('targets')[0].get('score')
    bias = 0
    if print_p:
        print('-----------------')
        case_summary_str = 'Case summary: '
        for feature_name in feature_list:
            feature_value = ddict.get(feature_name)            
            print(feature_name + '    ' + feature_value)
        #print(str(fdict))
        print('prediction_score:  {:.2f}'.format(score))
        if score_prob_table != None:
            (table_ar, offset, scale) = score_prob_table
            est_prob_recid = lookupTableValue(table_ar, offset, scale, score)
            print('estimated prob. recid: {:.2f}'.format(est_prob_recid))
        
    count = 1
    for_sum = 0.0
    against_sum = 0.0
    for feature_l in features_l:
        #print('\n' + str(feature_l))
        feature_contribution = feature_l.get('weight')
        feature_value = feature_l.get('value')   #1.0 = True or 0.0 = False for one-hot, or
                                                 #could be any float.
        if abs(feature_contribution) < abs_contribution_threshold:
            continue
        feature_l_index_str = feature_l.get('feature')  #of the form, 'x342'
        if feature_l_index_str == '<BIAS>':
            bias = feature_contribution
            continue

        vector_index = int(feature_l_index_str[1:])   #vector index
        feature_name_value = feature_name_value_reverse_map.get(vector_index)
        colon_index = feature_name_value.find(':')

        #if colon_index < 0:
        #    feature_name = feature_name_value
        #else:
        #    feature_name = feature_name_value[0:colon_index]
        
        #turn feature_name_value and feature_value into items for a readable description
        #feature_name_value and value can be like one of the following
        # categorical feature and value                       'age'  35
        # the average value of a c_charge_desc                'c_charge_desc:charge desc average'  .455
        # the boolean truth value a particular one-hot        'c_charge_desc:'Agg Batter Grt/Bod/Harm'  1.0
        #      charge description
        feature_name_str, feature_value_str = generateFeatureValueReadableStringElements(feature_name_value, feature_value)
        vv = (feature_contribution, feature_name_str, feature_value_str)
        if feature_contribution > 0:
            significant_feature_value_listing_pos.append(vv)
            #print(str(vv))
            for_sum += vv[0]
        else:
            significant_feature_value_listing_neg.append(vv)
            #print(str(vv))
            against_sum += vv[0]
        
    significant_feature_value_listing_pos.sort(key = lambda x: x[0], reverse=True)
    significant_feature_value_listing_neg.sort(key = lambda x: x[0])

    if print_p:
        print('\n    Indicators FOR predict recidivsim:  sum: {:.2f}'.format(for_sum))
        for item in significant_feature_value_listing_pos:
            feature_name_str = item[1]
            feature_value_str = item[2]
            print('{:50}'.format(feature_name_str) + '  ' + '{:20}'.format(feature_value_str) + '  {0:1.3f}'.format(item[0]))
        print('\n    Indicators AGAINST predict recidivism:   sum: {:.2f}'.format(against_sum))
        for item in significant_feature_value_listing_neg:
            feature_name_str = item[1]
            feature_value_str = item[2]
            print('{:50}'.format(feature_name_str) + '  ' + '{:20}'.format(feature_value_str) + '  {0:1.3f}'.format(item[0]))            
        print('\nbias: ' + str(bias))
        total_sum = bias + for_sum + against_sum
        print('for_sum + against_sum: {:.2f}'.format(for_sum + against_sum) + ' total_sum: {:.2f}'.format(total_sum) + '\n')

    return significant_feature_value_listing_pos, significant_feature_value_listing_neg, score

        


#
#
############################

#
#
################################################################################


################################################################################
#
#Exporting predictions and outcomes as .json files to display and explore using
#the PCDM Dashboard (Prediction Distribution Confusion Matrix).
#

#File format:
#{ 'data-set-name': '<data set name>',
#  'notes': '<notes',
#  'data-slices':
#     {'<slice-name>':
#         {'pos-outcomes': [34, 33, 22, ...],
#          'neg-outcomes': [4, 3, 2, ...]
#     }
#}
#
#The arrays of ints are counts per bin (e.g. decile bin) for number of condition-positive
#and condition-negative outcomes.  All count arrays in the file must be the same length
#under the assumption that a common model is used to generate all distributions.
#The data slices should include one called 'all-data'.
#
#For example, the Broward Recidivism data might include several .json files corresponding
#to different models.  One model will be the observed COMPAS decile predictions, then
#other models might be Logistic Regression models trained on various feature sets, such
#as gl_features_min (omitting sex and race) and gl_features_max (including sex and race).
#
#Each data file will contain a data slice called 'all-data' corresponding to the full
#population of data samples.  Then, it can contain breakdowns of the full population
#such as 'African-American' and 'Caucasian'.



gl_broward_compas_notes = ''





#This applies to predictions from COMPAS because it looks specifically for decile_score
#slice_spec_list is a list or tuple of tuple string specifying sectors for the data set.
#Each tuple is a field-name-value-list which is a list of tuple, (str field_name, str comp, str or int field_value, slice_name)
#that must be true for the field to be included in the result
#e.g. ('das_b_screening_arrest', '<=', 30)
#This function prepends 'all-data' to slice_spec_list.
#Example:   ('race', '==', 'Asian')
def writeCOMPASPredictionsToJSONFile(ddict_list, json_filepath, nickname, display_name, notes, 
                                     slice_spec_list = None, overwrite_existing_file_p=False):


    if os.path.exists(json_filepath):
        if not overwrite_existing_file_p:
            print('filepath ' + json_filepath + ' already exists, not overwriting')            
            return;
        print('filepath ' + json_filepath + ' already exists, overwriting with a new file')

    json_dict = {'data-set-nickname': nickname,
                 'data-set-display-name': display_name, 
                 'notes': notes,
                 'data-slices': {}
                 }

    recid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '1')])
    recid_hist = buildHistByDecile(recid_list)
    norecid_list = filterDdict(ddict_list, [(gl_recidivism_tag, '==', '0')])
    norecid_hist = buildHistByDecile(norecid_list)
    json_dict['data-slices']['all-data'] = {'pos-outcomes': recid_hist,
                                            'neg-outcomes': norecid_hist}

    if slice_spec_list == None:
        slice_spec_list = []
    for slice_spec in slice_spec_list:
        if len(slice_spec) > 3:
            slice_name = slice_spec[3]
        else:
            slice_name = slice_spec[2]
        ddict_list_slice = filterDdict(ddict_list, [slice_spec])
        recid_list = filterDdict(ddict_list_slice, [(gl_recidivism_tag, '==', '1')])
        recid_hist = buildHistByDecile(recid_list)
        norecid_list = filterDdict(ddict_list_slice, [(gl_recidivism_tag, '==', '0')])
        norecid_hist = buildHistByDecile(norecid_list)
        json_dict['data-slices'][slice_name] = {'pos-outcomes': recid_hist,
                                                'neg-outcomes': norecid_hist}
        

    with open(json_filepath, 'w') as file:
        json.dump(json_dict, file, indent=4)




#Take a set of groundtruth observations y and predictions y_preds and write to a JSON file.
#This is used for example if KFold cross validation is used to produce a set of GBR models
#that were used to produce KFold predictions.
#
#After running runTrainTestKFold(ddict_list2, ...
#call as 
#>>> ca.writeModelPredictionsToJSONFile(gl_last_y, gl_last_preds, ddict_list2, json_filepath, data_set_name, display_name,
#                                       notes, 10, [('race', '==', 'African American')('race', '==', 'Caucasian')], True)
def writeModelPredictionsToJSONFile(y, preds, ddict_list, json_filepath, data_set_name, display_name, notes, num_bins=20,
                                    slice_spec_list = None, overwrite_existing_file_p=False):
    if os.path.exists(json_filepath):
        if not overwrite_existing_file_p:
            print('filepath ' + json_filepath + ' already exists, not overwriting')            
            return;
        print('filepath ' + json_filepath + ' already exists, overwriting with a new file')

    json_dict = {'data-set-nickname': data_set_name,
                 'data-set-display-name': display_name, 
                 'notes': notes,
                 'data-slices': {}
                 }

    ddict_index_map = {}   #key: ddict  value: int index in ddict_list
    for index in range(len(ddict_list)):
        ddict = ddict_list[index]
        ddict['index'] = index;
    
    hist_arrays = predictionScoresToHistArrays(y, preds, num_bins)
    pos_hist = hist_arrays[0]
    neg_hist = hist_arrays[1]
    json_dict['data-slices']['all-data'] = {'pos-outcomes': pos_hist,
                                            'neg-outcomes': neg_hist}
    if slice_spec_list == None:
        slice_spec_list = []
    for slice_spec in slice_spec_list:
        if len(slice_spec) > 3:
            slice_name = slice_spec[3]
        else:
            slice_name = slice_spec[2]
        ddict_list_slice = filterDdict(ddict_list, [slice_spec])
        y_slice = []
        preds_slice = []
        for ddict in ddict_list_slice:
            index = ddict.get('index')
            y_slice.append(y[index])
            preds_slice.append(preds[index])
        
        hist_arrays = predictionScoresToHistArrays(y_slice, preds_slice, num_bins)
        pos_hist = hist_arrays[0]
        neg_hist = hist_arrays[1]
        json_dict['data-slices'][slice_name] = {'pos-outcomes': pos_hist,
                                                'neg-outcomes': neg_hist}
    with open(json_filepath, 'w') as file:
        json.dump(json_dict, file, indent=4)
        

#
#
################################################################################

################################################################################
#
#Top level user functions.
#


#gl_model = 'gbr'   Gradient Boosting Regressor
#gl_model = 'linr'  Linear Regression
#gl_model = 'logr'  Logistic Regression


#model_type can be one of {'gbr', 'linr', 'logr'}
def setModelType(model_type = 'gbr'):
    global gl_model_type
    global gl_model_train_fun

    if model_type not in ('gbr', 'linr', 'logr'):
        print('model type ' + model_type + ' not recognized')
        return
    gl_model_type = model_type
    print('setting model type to: ' + model_type)
    if model_type == 'gbr':
        gl_model_train_fun = trainGBR
    elif model_type == 'linr':
        gl_model_train_fun = trainLR
    elif model_type == 'logr':
        gl_model_train_fun = trainLogR        


        
try:
    gl_model_type
except:
    setModelType('logr')

                

#train a model on all of the data and test on all of the data
#Use this for Linear Regression and Logistic Regression models.
#The results are put into a bunch of global variables which you can then feed into
#plotting or write to file functions.
def runTrainTestFull(ddict_list, feature_list = gl_features_min, score_threshold=None):
    print('feature_list: ' + str(feature_list))
    X, y, feature_name_value_reverse_map = convertDdictsToFeatureVectorsAndGT(ddict_list, feature_list)
    model = gl_model_train_fun(X, y)
    preds = testModel(X, model)
    
    roc = buildROCCurveFromPredictionScores(y, preds)
    if score_threshold == None:
        #score_threshold, f1 = chooseOptimalF1Score(y, preds)
        #print('choosing score_threshold: ' + str(score_threshold) + ' for optimal f1 score: ' + str(f1))
        score_threshold, mcc = chooseOptimalMCCScore(y, preds)
        print('choosing score_threshold: ' + str(score_threshold) + ' for optimal mcc score: ' + str(mcc))
    else:
        print('using score_threshold passed: ' + str(score_threshold))
    cm, f1, mcc = buildConfusionMatrixFromPredictionScores(y, preds, score_threshold, True)
    score_prob_table = buildProbRecidVsScoreTable(y, preds)

    global gl_last_cm
    global gl_last_model
    global gl_last_X
    global gl_last_y
    global gl_last_preds
    global gl_last_roc
    global gl_last_score_prob_table
    global gl_last_feature_name_value_reverse_map
    gl_last_cm = cm
    gl_last_model = model
    gl_last_X = X
    gl_last_y = y
    gl_last_preds = preds
    gl_last_roc = roc
    gl_last_score_prob_table = score_prob_table
    gl_last_feature_name_value_reverse_map = feature_name_value_reverse_map

    #builtins is misleading.  It sets the variable in the interpreter the first time,
    #but it doesn't overwrite an existing value.
    #builtins.last_cm = cm
    #builtins.last_gbr = gbr
    #builtins.last_X = X
    #builtins.last_y = y
    #builtins.last_preds = preds
    #builtins.last_roc = roc
    #builtins.last_score_prob_table = score_prob_table
    #builtins.last_feature_name_value_reverse_map = feature_name_value_reverse_map


gl_kfold_n_splits = 5

    
#train and test models with KFold cross validation
#For each fold, build a model using 4/5 of the data, then use this
#model to predict the remaining 1/5 of the data.
#Compute summary statistics as usual using the complete set of predictions.
#But every prediction is based on a model that doesn't include that prediction's data
#in its training set.
#This is strictly needed for modeling with many-parameter models like GBR.
#By experimentation, I selected default GBR parameters that appear to not overfit on
#the ProPublica/COMPAS data set, so it is probably fine to use a GBR model trained on all of the data.  
#A linear model has few enough parameters that k-fold slicing is probably not necessary.
#The results are put into a bunch of global variables which you can then feed into
#plotting or write to file functions.
def runTrainTestKFold(ddict_list, feature_list = gl_features_min, score_threshold=None):
    print('feature_list: ' + str(feature_list))
    X, y, feature_name_value_reverse_map = convertDdictsToFeatureVectorsAndGT(ddict_list, feature_list)

    kfold = KFold(n_splits=gl_kfold_n_splits, shuffle=True)
    i_fold = 0
    #initialize preds with an empty list of predictions to be filled in from each fold slice
    preds = [None] * len(ddict_list)
    
    for train_i_ar, test_i_ar in kfold.split(ddict_list):  #type is numpy array
        print('fold: ' + str(i_fold))
        i_fold += 1
        train_i_list = list(train_i_ar)
        test_i_list = list(test_i_ar)

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for i_train in train_i_list:
            X_train.append(X[i_train])
            y_train.append(y[i_train])
        model_fold = gl_model_train_fun(X_train, y_train)
        
        for i_test in test_i_list:            
            X_test.append(X[i_test])

        preds_test = testModel(X_test, model_fold)

        #put the pred_test into preds in alignment with X_test and y_test.
        pred_index = 0
        for i_test in test_i_list:
            preds[i_test] = preds_test[pred_index]
            pred_index += 1

        #development and debugging
        #temp: print the confusion matrix for just this slice of predictions
        #y_temp = []
        #index = 0
        #for i_test in test_i_list:
        #    y_temp.append(y[index])
        #    index += 1
        #buildConfusionMatrixFromPredictionScores(y_temp, preds_test, .5, True)

    #print('preds: ' + str(preds))

    roc = buildROCCurveFromPredictionScores(y, preds)
    if score_threshold == None:
        #score_threshold, f1 = chooseOptimalF1Score(y, preds)
        #print('choosing score_threshold: ' + str(score_threshold) + ' for optimal f1 score: ' + str(f1))
        score_threshold, mcc = chooseOptimalMCCScore(y, preds)
        print('choosing score_threshold: ' + str(score_threshold) + ' for optimal mcc score: ' + str(mcc))
    else:
        print('using score_threshold passed: ' + str(score_threshold))
    cm, f1, mcc = buildConfusionMatrixFromPredictionScores(y, preds, score_threshold, True)
    score_prob_table = buildProbRecidVsScoreTable(y, preds)


    global gl_last_cm
    global gl_last_X
    global gl_last_y
    global gl_last_preds
    global gl_last_roc
    global gl_last_score_prob_table
    global gl_last_feature_name_value_reverse_map
    gl_last_cm = cm
    gl_last_X = X
    gl_last_y = y
    gl_last_preds = preds
    gl_last_roc = roc
    gl_last_score_prob_table = score_prob_table
    gl_last_feature_name_value_reverse_map = feature_name_value_reverse_map
    

#Take a trained model and test it separately on Caucasian and African-American data sets.
#Make sure the feature_list matches the one used to train the model.
def runRaceComparisonForModel(ddict_list, model, score_threshold, feature_list = gl_features_min):
    print('feature_list: ' + str(feature_list))
    
    print('\nCaucasian')
    ddict_list_cau = filterDdict(ddict_list, [('race', '==', 'Caucasian')])
    X_cau, y_cau, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_cau, feature_list)
    preds_cau = testModel(X_cau, model)
    roc_cau = buildROCCurveFromPredictionScores(y_cau, preds_cau)
    cm_cau, f1_cau, mcc_cau = buildConfusionMatrixFromPredictionScores(y_cau, preds_cau, score_threshold)
    score_prob_table_cau = buildProbRecidVsScoreTable(y_cau, preds_cau)
    global gl_last_cm_cau
    global gl_last_X_cau
    global gl_last_y_cau
    global gl_last_preds_cau
    global gl_last_roc_cau
    global gl_last_score_prob_table_cau
    gl_last_cm_cau = cm_cau
    gl_last_X_cau = X_cau
    gl_last_y_cau = y_cau
    gl_last_preds_cau = preds_cau
    gl_last_roc_cau = roc_cau
    gl_last_score_prob_table_cau = score_prob_table_cau

    print('\nAfrican-American')
    ddict_list_aa = filterDdict(ddict_list, [('race', '==', 'African-American')])    
    X_aa, y_aa, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_aa, feature_list)
    preds_aa = testModel(X_aa, model)    

    roc_aa = buildROCCurveFromPredictionScores(y_aa, preds_aa)
    cm_aa, f1_aa, mcc_aa = buildConfusionMatrixFromPredictionScores(y_aa, preds_aa, score_threshold)
    score_prob_table_aa = buildProbRecidVsScoreTable(y_aa, preds_aa)    
    global gl_last_cm_aa
    global gl_last_X_aa
    global gl_last_y_aa
    global gl_last_preds_aa
    global gl_last_roc_aa
    global gl_last_score_prob_table_aa
    gl_last_cm_aa = cm_aa
    gl_last_X_aa = X_aa
    gl_last_y_aa = y_aa
    gl_last_preds_aa = preds_aa
    gl_last_roc_aa = roc_aa
    gl_last_score_prob_table_aa = score_prob_table_aa




#Take a trained model and test it separately on Caucasian and African-American data sets,
#but reverse the 'race' feature for Caucasian and African-Americans.
#This can make a difference only for a model that uses the race feature.
#Therefore, this function by default uses the feature_list, gl_features_max.
def runRaceComparisonForModel_InvertRace(ddict_list, model, score_threshold, feature_list = gl_features_max):
    print('feature_list: ' + str(feature_list))
    
    print('\nCaucasian')
    ddict_list_cau = filterDdict(ddict_list, [('race', '==', 'Caucasian')])
    ddict_list_cau_race_inverted = copyDDictsReplaceFeatureValues(ddict_list_cau, [('race', 'African-American')])
    X_cau_rinv, y_cau_rinv, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_cau_race_inverted, feature_list)
    preds_cau_rinv = testModel(X_cau_rinv, model)
    roc_cau_rinv = buildROCCurveFromPredictionScores(y_cau_rinv, preds_cau_rinv)
    cm_cau_rinv, f1_cau_rinv, mcc_cau_rinv = buildConfusionMatrixFromPredictionScores(y_cau_rinv, preds_cau_rinv,
                                                                                      score_threshold)
    score_prob_table_cau_rinv = buildProbRecidVsScoreTable(y_cau_rinv, preds_cau_rinv)
    global gl_last_cm_cau_rinv
    global gl_last_X_cau_rinv
    global gl_last_y_cau_rinv
    global gl_last_preds_cau_rinv
    global gl_last_roc_cau_rinv
    global gl_last_score_prob_table_cau_rinv
    gl_last_cm_cau_rinv = cm_cau_rinv
    gl_last_X_cau_rinv = X_cau_rinv
    gl_last_y_cau_rinv = y_cau_rinv
    gl_last_preds_cau_rinv = preds_cau_rinv
    gl_last_roc_cau_rinv = roc_cau_rinv
    gl_last_score_prob_table_cau_rinv = score_prob_table_cau_rinv

    print('\nAfrican-American')
    ddict_list_aa = filterDdict(ddict_list, [('race', '==', 'African-American')])
    ddict_list_aa_race_inverted = copyDDictsReplaceFeatureValues(ddict_list_aa, [('race', 'Caucasian')])

    X_aa_rinv, y_aa_rinv, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_aa_race_inverted, feature_list)
    preds_aa_rinv = testModel(X_aa_rinv, model)
    roc_aa_rinv = buildROCCurveFromPredictionScores(y_aa_rinv, preds_aa_rinv)
    cm_aa_rinv, f1_aa_rinv, mcc_aa_rinv = buildConfusionMatrixFromPredictionScores(y_aa_rinv, preds_aa_rinv,
                                                                                      score_threshold)
    score_prob_table_aa_rinv = buildProbRecidVsScoreTable(y_aa_rinv, preds_aa_rinv)
    global gl_last_cm_aa_rinv
    global gl_last_X_aa_rinv
    global gl_last_y_aa_rinv
    global gl_last_preds_aa_rinv
    global gl_last_roc_aa_rinv
    global gl_last_score_prob_table_aa_rinv
    gl_last_cm_aa_rinv = cm_aa_rinv
    gl_last_X_aa_rinv = X_aa_rinv
    gl_last_y_aa_rinv = y_aa_rinv
    gl_last_preds_aa_rinv = preds_aa_rinv
    gl_last_roc_aa_rinv = roc_aa_rinv
    gl_last_score_prob_table_aa_rinv = score_prob_table_aa_rinv




#Take a trained model and test it separately on Male and Female data sets.
#Make sure the feature_list matches the one used to train the model.
def runSexComparisonForModel(ddict_list, model, score_threshold, feature_list = gl_features_min):
    print('feature_list: ' + str(feature_list))
    
    print('\nMale')
    ddict_list_m = filterDdict(ddict_list, [('sex', '==', 'Male')])
    X_m, y_m, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_m, feature_list)
    preds_m = testModel(X_m, model)
    roc_m = buildROCCurveFromPredictionScores(y_m, preds_m)
    cm_m, f1_m, mcc_m = buildConfusionMatrixFromPredictionScores(y_m, preds_m, score_threshold)
    score_prob_table_m = buildProbRecidVsScoreTable(y_m, preds_m)
    global gl_last_cm_m
    global gl_last_X_m
    global gl_last_y_m
    global gl_last_preds_m
    global gl_last_roc_m
    global gl_last_score_prob_table_m
    gl_last_cm_m = cm_m
    gl_last_X_m = X_m
    gl_last_y_m = y_m
    gl_last_preds_m = preds_m
    gl_last_roc_m = roc_m
    gl_last_score_prob_table_m = score_prob_table_m

    print('\nFemale')
    ddict_list_f = filterDdict(ddict_list, [('sex', '==', 'Female')])    
    X_f, y_f, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_f, feature_list)
    preds_f = testModel(X_f, model)    

    roc_f = buildROCCurveFromPredictionScores(y_f, preds_f)
    cm_f, f1_f, mcc_f = buildConfusionMatrixFromPredictionScores(y_f, preds_f, score_threshold)
    score_prob_table_f = buildProbRecidVsScoreTable(y_f, preds_f)    
    global gl_last_cm_f
    global gl_last_X_f
    global gl_last_y_f
    global gl_last_preds_f
    global gl_last_roc_f
    global gl_last_score_prob_table_f
    gl_last_cm_f = cm_f
    gl_last_X_f = X_f
    gl_last_y_f = y_f
    gl_last_preds_f = preds_f
    gl_last_roc_f = roc_f
    gl_last_score_prob_table_f = score_prob_table_f



    

#train a model on all of the data for a selected race.
#target_race can be 'Caucasian', 'African-American', or another race.
#Note however that the data sizes for other races are much smaller and you'll end up
#overfitting when training on a full race-selected data set that is very small.
def runTrainTestFull_SelectRace(ddict_list, target_race, feature_list = gl_features_min,
                                score_threshold = None):
    ddict_list_race = filterDdict(ddict_list, [('race', '==', target_race)])
    print('data set size for race ' + target_race + ' is ' + str(len(ddict_list_race)))
    X_race, y_race, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_race, feature_list)

    model_race = gl_model_train_fun(X_race, y_race)
    preds_race = testModel(X_race, model_race)    
    roc_race = buildROCCurveFromPredictionScores(y_race, preds_race)
    if score_threshold == None:
        #score_threshold, f1 = chooseOptimalF1Score(y, preds)
        #print('choosing score_threshold: ' + str(score_threshold) + ' for optimal f1 score: ' + str(f1))
        score_threshold, mcc = chooseOptimalMCCScore(y_race, preds_race)
        print('choosing score_threshold: ' + str(score_threshold) + ' for optimal mcc score: ' + str(mcc))
    else:
        print('using score_threshold passed: ' + str(score_threshold))
    cm_race, f1_race, mcc_race = buildConfusionMatrixFromPredictionScores(y_race, preds_race, score_threshold)
    score_prob_table_race = buildProbRecidVsScoreTable(y_race, preds_race)

    global gl_last_target_race
    global gl_last_cm_target_race
    global gl_last_model_target_race
    global gl_last_X_target_race
    global gl_last_y_target_race
    global gl_last_preds_target_race
    global gl_last_roc_target_race
    global gl_last_score_prob_table_target_race
    global gl_last_feature_name_value_reverse_map_target_race
    gl_last_target_race = target_race
    gl_last_cm_target_race = cm_race
    gl_last_model_target_race = model_race
    gl_last_X_target_race = X_race
    gl_last_y_target_race = y_race
    gl_last_preds_target_race = preds_race
    gl_last_roc_target_race = roc_race
    gl_last_score_prob_table_target_race = score_prob_table_race
    gl_last_feature_name_value_reverse_map_target_race = rev_map    
    





#run the model on ddict_list including 'all-data' and the slices specified in slice_spec_list.
#write as a json file that can be imported into the PDCM Dashboard (Prediction Distribution Confusion Matrix)
#This applies to a Logistic Regression model type where we can apply a single fit model to
#the entire data set.
def runModelWriteHistArraysToJSONFile(ddict_list, model, json_filepath, data_set_name, display_name, notes,
                                      feature_list = gl_features_min,
                                      num_bins=20,
                                      slice_spec_list = None,
                                      overwrite_existing_file_p=False):
    if os.path.exists(json_filepath):
        if not overwrite_existing_file_p:
            print('filepath ' + json_filepath + ' already exists, not overwriting')            
            return;
        print('filepath ' + json_filepath + ' already exists, overwriting with a new file')

    json_dict = {'data-set-nickname': data_set_name,
                 'data-set-display-name': display_name, 
                 'notes': notes,
                 'data-slices': {}
                 }

    X, y, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list, feature_list)
    preds = testModel(X, model)
    hist_arrays = predictionScoresToHistArrays(y, preds, num_bins)
    recid_hist = hist_arrays[0]
    norecid_hist = hist_arrays[1]
    json_dict['data-slices']['all-data'] = {'pos-outcomes': recid_hist,
                                            'neg-outcomes': norecid_hist}
    if slice_spec_list == None:
        slice_spec_list = []
    for slice_spec in slice_spec_list:
        if len(slice_spec) > 3:
            slice_name = slice_spec[3]
        else:
            slice_name = slice_spec[2]
        ddict_list_slice = filterDdict(ddict_list, [slice_spec])
        X, y, rev_map = convertDdictsToFeatureVectorsAndGT(ddict_list_slice, feature_list)
        preds = testModel(X, model)
        hist_arrays = predictionScoresToHistArrays(y, preds, num_bins)
        recid_hist = hist_arrays[0]
        norecid_hist = hist_arrays[1]
        json_dict['data-slices'][slice_name] = {'pos-outcomes': recid_hist,
                                                'neg-outcomes': norecid_hist}
    with open(json_filepath, 'w') as file:
        json.dump(json_dict, file, indent=4)
        
        

#
#
################################################################################


