

#appleSnacks.py is for generating synthetic example data sets to explore what
#algorithmic bias looks like in a Confusion Matrix.

import random
import json
import compasAnalysis as ca
import os
import random
from matplotlib import pyplot as plt
import math
import scipy.special as sp



#map from feature values to ways of assigning a score
#If 'float', then sample values from a gaussian distribution using the means and std dev in
#gl_feature_dist_map.
#If a list, then sample values according to the fractions in gl_feature_dist_map.
gl_feature_val_to_score_maps = {'age': 'float',
                                'teacher':(('Ms. Popcorn', 0.0), ('Miss Fruitdale', 1.0), ('Mr. Applebaum', 2.0)),
                                'height': 'float',
                                'pet':(('turtle', 0.0), ('fish', 1.0), ('cat', 2.0), ('dog', 3.0))}

gl_feature_weight_map = {'age': 1.0, 'teacher': 1.0, 'height': 1.0, 'pet':1.0}


#For each feature, we can have either a normal distribution ['normal', mean, std-dev, min, max]
#or else a distribution with probalities indicated:  ['dist', p1, p2, ...]

#linear spec is (min_val, max_val, min_frac)
#min_frac can range from 0.0 to 2.0.  1.0 means uniform


#normal spec arguments:   [mean, stddev, min, max]
#uniform spec arguments:   [min, max]
#linear spec arguments:   [min, max, min_val_frac]   min_val_frac ranges from 0 to 2.0
#triangular spec arguments:   [min, mode, max]
#categorical spec arguments:   [ frac1, frac2, ...]   where the fracs sum to 1.0.


#Equal distribution of all feature values for girls and boys.
gl_feature_dist_map_uniform = \
    {'distribution-name': 'feature_dist_map_uniform',
     'notes':'Synthesized data with four factors (age, height, teacher, pet). <br> Feature distributions are uniform and the same for girls and boys.<br> This produces a balanced distribution of preference for apple snack versus other snack.',
     'girl':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.333, .333, .333),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.25, .25, .25, .25),
                    'offset':0,
                    'scale':.3333}                  #scale score range to 1.0
    },
     
     'boy':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.333, .333, .333),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.25, .25, .25, .25),
                   'offset':0,
                   'scale':.3333}                   #scale score range to 1.0
     }
    }


#This distribution skews girls' and boys' features equally toward values that prefer apple snacks.
gl_feature_dist_map_skewed_same = \
    {'distribution-name': 'feature_dist_map_skewed_same',
     'notes': 'Synthesized data with four factors (age, height, teacher, pet).<br> Feature distributions are skewed toward higher values.<br> This produces a distribution of preference that is skewed toward apple snack versus other.<br> Same distribution for girls and boys.',
    'girl':{'age': {'linear': (6.0, 10.0, .6),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, .6), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.266, .333, .4),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.15, .2, .3, .35),
                    'offset':0,
                    'scale':.3333}
    },
     
     'boy':{'age': {'linear': (6.0, 10.0, .6),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, .6), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.266, .333, .4),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.15, .2, .3, .35),
                   'offset':0,
                   'scale':.3333}\
     }
 }



#Girls' feature distributions are skewed towards values that prefer apple,
#boys' feature distributions that are skewed toward values that prefer no apple.
#There is no direct sex feature in the computation of preference score.
gl_feature_dist_map_skewed_opposite = \
    {'distribution-name': 'feature_dist_map_skewed_opposite',
     'notes': 'Synthesized data with four factors (age, height, teacher, pet).<br> Feature distributions for girls are skewed toward higher values.<br> Feature distributions for boys are  skewed toward lower values.<br> These feature distributions lead to distributions of predicted preference that skew toward apple snack for girls (rightward), and toward other snack for boys (leftward).',
     'girl':{'age': {'linear': (6.0, 10.0, .6),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, .6), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.266, .333, .4),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.15, .20, .30, .35),
                    'offset':0,
                    'scale':.3333}
    },
     
     'boy':{'age': {'linear': (6.0, 10.0, 1.4),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, 1.4), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.4, .333, .266),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.35, .3, .2, .15),
                   'offset':0,
                   'scale':.3333}
     }
    }






#Girls' feature distributions are skewed towards values that prefer apple,
#boys' feature distributions that are skewed toward values that prefer no apple.
#There is no direct sex feature in the computation of preference score.
#Girls probs and bin assignments are skewed toward other (left)
gl_feature_dist_map_skewed_opposite_bias_girls = \
    {'distribution-name': 'feature-dist-map-skewed-opposite-bias-girls',
     'notes':'Synthesized data with four factors (age, height, teacher, pet).<br> Feature distributions are skewed so that girls prefer apple snack and boys prefer other snack.<br> Then, the girls prediction scores were shifted left (toward other) by .1 (2 bins). This creates a bias for girls to be assigned a lower prediction score than they would otherwise get based on their preferences which are computed from their factors (age, etc.).<br> Even though the girls and boys distributions look similar, and their Confusion Matrix values can be made congruent by adjusting the decision threshold, the positive prediction ratios are offset.  This is reflected in a high PPRS (Positive Prediction Ratio Score).',
     'girl':{'age': {'linear': (6.0, 10.0, .6),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, .6), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.266, .333, .4),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.15, .2, .3, .35),
                    'offset':0,
                    'scale':.3333},
             'prob-distortion':(1.0, 0.0, -.1)
             },
     
     'boy':{'age': {'linear': (6.0, 10.0, 1.4),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, 1.4), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.4, .333, .266),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.35, .3, .2, .15),
                   'offset':0,
                   'scale':.3333}
            },
     
     'preference-filter': {'girl': (1.0, .0),  #skew samples toward lower scores...
                           #skew samples toward higher scores...
                           #  ...regardless of outcome                           
                           'boy': (.0, 1.0)}
     }




gl_feature_dist_map_uniform_randomize_girls_uniform = \
    {'distribution-name': 'feature_dist_map_uniform_randomize_girls_uniform',
     'notes':'Synthesized data with four factors (age, height, teacher, pet).<br> Feature distributions are uniform and the same for girls and boys.<br> This produces a balanced distribution of preference for apple snack versus other snack.<br> Then, 1/3 of the girls prediction scores were replaced by a uniform random value. This creates a bias for girls to be assigned a different prediction score than they would otherwise get based on their preferences which are computed from their factors (age, etc.).<br> This squashes the girls preference distributions.  The Positive Prediction Ratio Curves bend toward .5 at lower and higher preference scores that previously had fewer samples.<br> The bias is reflected in a high PPRS (Positive Prediction Ratio Score).',
     'girl':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.333, .333, .333),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.25, .25, .25, .25),
                    'offset':0,
                    'scale':.3333},                 #scale score range to 1.0
             'randomize-distortion': (.3333, 'uniform')
             },

     'boy':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.333, .333, .333),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.25, .25, .25, .25),
                   'offset':0,
                   'scale':.3333},                  #scale score range to 1.0
            }
     }




#Start with equal distribution of all usual feature values for girls and boys.
#Then, 50% of girls prediction scores are re-assigned to a normal distribution with mean .65, stddev .15
gl_feature_dist_map_uniform_randomize_girls_normal_p65_p15 = \
    {'distribution-name': 'feature_dist_map_uniform_randomize_girls_normal_p65_p15',
     'notes':'Synthesized data with four factors (age, height, teacher, pet).<br> Feature distributions are uniform and the same for girls and boys.<br> This produces a balanced distribution of preference for apple snack versus other snack.<br> Then, 50% of the girls prediction scores were replaced by a normal random value,<br> mean = .65, stddev = .15.<br> This creates a bias for girls to be assigned a different prediction score than they would otherwise get based on their preferences which are computed from their factors (age, etc.).<br> The bias is reflected in a high PPRS (Positive Prediction Ratio Score).',
     'girl':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.333, .333, .333),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.25, .25, .25, .25),
                    'offset':0,
                    'scale':.3333},                 #scale score range to 1.0
             'randomize-distortion': (.5, ('normal', .65, .15))  #mean, stdev
             },

     'boy':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.333, .333, .333),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.25, .25, .25, .25),
                   'offset':0,
                   'scale':.3333},                  #scale score range to 1.0
            }
     }






#Surplus
#This turns out not to be informative.  This causes rebalancing of the distributions for
#girls and boys toward equal CN and CP.  This effectively makes them about the same.
gl_feature_dist_map_skewed_opposite_balanced = \
    {'distribution-name': 'feature_dist_map_skewed_opposite_balanced',    
     'girl':{'age': {'linear': (6.0, 10.0, .6),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, .6), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.266, .333, .4),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.15, .2, .3, .35),
                    'offset':0,
                    'scale':.3333}
             },
     
     'boy':{'age': {'linear': (6.0, 10.0, 1.4),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, 1.4), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.4, .333, .266),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.35, .3, .2, .15),
                   'offset':0,
                   'scale':.3333}
            },
     'preference-filter': {'girl': (1.0, .0),  #skew samples toward lower scores...
                           'boy': (.0, 1.0)}
     }



#Surplus
#Equal distribution of all usual feature values for girls and boys.
#This biases girls prediction scores toward apple (right) while leaving boys predictions alone.
gl_feature_dist_map_uniform_bias_sex = \
    {'distribution-name': 'feature_dist_map_uniform_bias_sex_feature',
     'notes':'Distributions of features: age, height, teacher, pet, is uniform. This produces a balanced distribution of preference for apple snack versus other.  Same distribution for girls and boys.  But an additional sex feature is used that is skewed toard apple preference for girls, other preference for boys',
     'girl':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                     'offset': 6.0,
                     'scale': .25},                 #scale score range to 1.0
             'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                        'offset': 40.0,
                        'scale': .05},              #scale score range to 1.0
             'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                         'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                         'dist': (.333, .333, .333),
                         'offset': 0,
                         'scale': .5},              #scale score range to 1.0
             'pet':{'categorical':(0, 1, 2, 3),     #range=0-3 size 3
                    'names':('turtle', 'fish', 'cat', 'dog'),
                    'dist':(.25, .25, .25, .25),
                    'offset':0,
                    'scale':.3333},                 #scale score range to 1.0
             'prob-distortion':(1.0, 0.0, .1)       #(a, b, c)  prob' = a * (prob - b) + c
             },

     'boy':{'age': {'linear': (6.0, 10.0, 1.0),  #range 6-10 size 4.0
                    'offset': 6.0,
                    'scale': .25},                 #scale score range to 1.0
            'height': {'linear': (40.0, 60.0, 1.0), #range = 40-60 size 20
                       'offset': 40.0,
                       'scale': .05},              #scale score range to 1.0
            'teacher': {'categorical':(0, 1, 2),   #range=0-2 size 2
                        'names':('Ms. Popcorn', 'Miss Fruitdale', 'Mr. Applebaum'),
                        'dist': (.333, .333, .333),
                        'offset': 0,
                        'scale': .5},               #scale score range to 1.0
            'pet':{'categorical':(0, 1, 2, 3),      #range=0-3 size 3
                   'names':('turtle', 'fish', 'cat', 'dog'),
                   'dist':(.25, .25, .25, .25),
                   'offset':0,
                   #scale score range to 1.0
                   #'prob-distortion':(1.0, 0.0, 0)     #(a, b, c)  prob' = a * (prob - b) + c                   
                   'scale':.3333}
            }
    }






gl_feature_dist_map = gl_feature_dist_map_uniform
#gl_feature_dist_map = 'gl_feature_dist_map_skewed_same'
#gl_feature_dist_map = 'gl_feature_dist_map_skewed_opposite'
#gl_feature_dist_map = 'gl_feature_dist_map_skewed_oppostite-balanced'



gl_preference_name_map = {True: 'apple', False:'other'}


gl_sex_index_map = {'girl':0, 'boy':1}

#proportions of girls and boys among kids generated.
#(frac_girl, frac_boy)
gl_sex_proportions = (.5, .5)

gl_prediction_score_tag = 'apple-prediction-score'
gl_prediction_bin_tag = 'apple-prediction-bin'
gl_outcome_tag = 'apple-preference'

gl_num_bins = 20



def determineScoreRange(feature_dist_map = None):
    if feature_dist_map == None:
        feature_dist_map = gl_feature_dist_map
    
    overall_min_val = 1000000
    overall_max_val = -1000000
    overall_min_score= 1000000
    overall_max_score = -1000000    
    for sex in ('girl', 'boy'):
        print('')
        sex_min_val = 0.0
        sex_max_val = 0.0
        sex_min_score = 0.0
        sex_max_score = 0.0
        for feature_name in feature_dist_map[sex].keys():
            if feature_name in ('prob-distortion', 'randomize-distortion'):
                continue
            feature_scale = feature_dist_map[sex][feature_name].get('scale')
            feature_offset = feature_dist_map[sex][feature_name].get('offset')
            normal_spec = feature_dist_map[sex][feature_name].get('normal')
            uniform_spec = feature_dist_map[sex][feature_name].get('uniform')
            triangular_spec = feature_dist_map[sex][feature_name].get('triangular')
            linear_spec = feature_dist_map[sex][feature_name].get('linear')            
            categorical_spec = feature_dist_map[sex][feature_name].get('categorical')
            if normal_spec != None:
                f_min_val = normal_spec[2] 
                f_max_val = normal_spec[3]
                f_min_score = 0.0
                f_max_score = (normal_spec[3] - normal_spec[2]) * feature_scale
            elif uniform_spec != None:
                f_min_val = uniform_spec[0] 
                f_max_val = uniform_spec[1] 
                f_min_score = 0.0
                f_max_score = (uniform_spec[1] - uniform_spec[0]) * feature_scale
            elif triangular_spec != None:
                f_min_val = triangular_spec[0] 
                f_max_val = triangular_spec[1] 
                f_min_score = 0.0
                f_max_score = (triangular_spec[1] - triangular_spec[0]) * feature_scale
                
            elif linear_spec != None:
                f_min_val = linear_spec[0] 
                f_max_val = linear_spec[1] 
                f_min_score = 0.0
                f_max_score = (linear_spec[1] - linear_spec[0]) * feature_scale                
                
            elif categorical_spec != None:
                print('categorical_spec: ' + str(categorical_spec))
                f_min_val = categorical_spec[0]
                f_max_val = categorical_spec[len(categorical_spec)-1]
                f_min_score = (categorical_spec[0] - feature_offset) * feature_scale
                f_max_score = (categorical_spec[len(categorical_spec)-1]  - feature_offset) * feature_scale
            print(feature_name + ' min: ' + str(f_min_val) + ' max: ' + str(f_max_val) + ' min_score: ' + str(f_min_score) + ' max_score: ' + str(f_max_score))
            sex_min_val += f_min_val
            sex_max_val += f_max_val
            sex_min_score += f_min_score
            sex_max_score += f_max_score
        overall_min_val = min(overall_min_val, sex_min_val)
        overall_max_val = max(overall_max_val, sex_max_val)
        overall_min_score = min(overall_min_score, sex_min_score)
        overall_max_score = max(overall_max_score, sex_max_score)
    print('overall_score_range: value: (' + str(overall_min_val) + ', ' + str(overall_max_val) + ')  score: (' + str(overall_min_score) + ', ' + str(overall_max_score))
    return overall_min_val, overall_max_val, overall_min_score, overall_max_score





#normal spec arguments:   [mean, stddev, min, max]
#uniform spec arguments:   [min, max]
#linear spec arguments:   [min, max, min_val_frac]   min_val_frac ranges from 0 to 2.0
#triangular spec arguments:   [min, mode, max]
#categorical spec arguments:   [ frac1, frac2, ...]   where the fracs sum to 1.0.



#This data generator model generates kids and their preferences for having an apple
#snack or some other kind of snack.
#Each kid gets a groundtruth expected-preference score based on some features.
#The expected preference will be a probability from 0 to 1 that they want an apple snack.
#Then, the kid gets an observed preference outcome which is based on a biased coin flip,
#with the bias being their expected preference.
#This process generates a distribution of kids with observed preferences that
#more-or-less align with expectations.
#
#Preferences are generated by sampling from the different features according
#to distributions set up in the feature_dist_map given by feature_dist_map
#
#The balance of scores is filtered by a linear probability distribution that skews
#the samples of girls or boys toward lower or higher prediction scores.
#
def generateAppleSnackSamples(num_kids, num_bins = gl_num_bins, feature_dist_map = None):
    if feature_dist_map == None:
        feature_dist_map = gl_feature_dist_map
    print('using feature distribution: ' + feature_dist_map.get('distribution-name'))
        
    kid_list = []

    min_val, max_val, range_min_score, range_max_score = determineScoreRange(feature_dist_map)
    
    for sex in ('girl', 'boy'):
        sex_index = gl_sex_index_map[sex]
        num_of_sex = int(num_kids * gl_sex_proportions[sex_index])

        kid_i = 0
        emergency_count = 0
        while kid_i < num_of_sex:
            emergency_count += 1
            if emergency_count > 1000000:
                print('exceeded emergency count so quitting')
                break
            #print('')

            feature_score_sum = 0.0
            kid_dict = {'sex':sex}
            for feature_name in feature_dist_map[sex].keys():
                if feature_name in ('prob-distortion', 'randomize-distortion'):
                    continue
                feature_scale = feature_dist_map[sex][feature_name].get('scale')
                feature_offset = feature_dist_map[sex][feature_name].get('offset')
                normal_spec = feature_dist_map[sex][feature_name].get('normal')
                uniform_spec = feature_dist_map[sex][feature_name].get('uniform')
                triangular_spec = feature_dist_map[sex][feature_name].get('triangular')
                linear_spec = feature_dist_map[sex][feature_name].get('linear')                
                categorical_spec = feature_dist_map[sex][feature_name].get('categorical')
                
                if normal_spec != None:
                    feature_value = random.gauss(normal_spec[0], normal_spec[1])
                    count = 0;
                    while feature_value < normal_spec[2] or feature_value > normal_spec[3]:
                        feature_value = random.gauss(normal_spec[0], normal_spec[1])                            
                        count += 1
                        if count > 100:
                            print('count exceeded 100 while sampling a normal distribution with mean ' + str(normal_spec[0]) + ' stdev: ' + str(normal_spec[1]) + ' min limit: ' + str(normal_spec[2]) + ' max_limit: ' + str(normal_spec[2]))

                    rr = normal_spec[3] - normal_spec[2];
                    rrb = 50.0/rr
                    
                    feature_value = max(normal_spec[2], min(normal_spec[3], feature_value))
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value

                elif uniform_spec != None:
                    feature_value = random.uniform(uniform_spec[0], uniform_spec[1])
                    rr = uniform_spec[1] - uniform_spec[0];
                    rrb = 50.0/rr  
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value

                elif triangular_spec != None:
                    feature_value = random.triangular(triangular_spec[0], triangular_spec[1], triangular_spec[2])
                    rr = triangular_spec[1] - triangular_spec[0];
                    rrb = 50.0/rr  
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value

                #linear spec is (min_val, max_val, slope_dir, frac_uniform, frac_triangular)                             
                elif linear_spec != None:
                    feature_value = sampleLinear(linear_spec[0], linear_spec[1], linear_spec[2])
                    rr = linear_spec[1] - linear_spec[0];
                    rrb = 50.0/rr  
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value
                    
                elif categorical_spec != None:
                    dist_tup = feature_dist_map[sex][feature_name].get('dist')
                    feature_cum_prob_ar = []
                    feature_cum_prob = 0.0
                    for feature_index in range(len(dist_tup)):
                        #print('feature_name: ' + feature_name + ' dist_tup: ' + str(dist_tup) + ' feature_index: ' + str(feature_index))
                        feature_cum_prob += dist_tup[feature_index]
                        feature_cum_prob_ar.append(feature_cum_prob)
                    feature_cum_prob_ar[len(feature_cum_prob_ar)-1] = 1.0
                    pp = random.random()
                    for feature_index in range(len(feature_cum_prob_ar)):
                        if pp < feature_cum_prob_ar[feature_index]:
                            break
                    feature_value = categorical_spec[feature_index]
                    feature_score = (feature_value - feature_offset) * feature_scale
                    feature_value_name = feature_dist_map[sex][feature_name].get('names')[feature_index]
                    #if feature_name == 'teacher':
                    #    print('cum: ' + str(feature_cum_prob_ar) + ' pp: ' + str(pp) + ' feature_index: ' + str(feature_index) + ' feature_value: ' + str(feature_value))
                    
                    kid_dict[feature_name] = feature_value_name

                #print('    feature_name: ' + feature_name + ' feature_score: ' + str(feature_score))
                feature_score_sum += feature_score
                
            #normalize the sum of feature scores to 1
            prob = (feature_score_sum - range_min_score) / (range_max_score - range_min_score)

            #apply distortion which alters the bin assignment from what it would ideally be
            #according to the kid's actual preferences
            prob_distortion = feature_dist_map[sex].get('prob-distortion')
            if prob_distortion == None:
                prob_pr = prob
            else:
                prob_pr = prob_distortion[0] * (prob - prob_distortion[1]) + prob_distortion[2]
                prob_pr = max(0.0, min(1.0, prob_pr))

            #print('prob_distortion: ' + str(prob_distortion))
            

            #bin_i = int(prob_pr * (num_bins-1)) + 1   #bins are indexed from 1
            bin_i = int(prob_pr * num_bins) + 1   #bins are indexed from 1
            #print(sex + ' prob: ' + str(prob) + ' bin_i: ' + str(bin_i) + ' prob_pr: ' + str(prob_pr));            

            #Under a randomize distortion, some fraction of kid probs are randomly reset to a different
            #bin on a uniform scale.
            randomize_distortion = feature_dist_map[sex].get('randomize-distortion')
            if randomize_distortion != None:
                randomize_prob = randomize_distortion[0]
                if random.random() < randomize_prob:
                    randomize_spec = randomize_distortion[1]
                    #'randomize-distortion': (.4, 'uniform')
                    #'randomize-distortion': (.4, ('normal', .5, .1))                    
                    if randomize_spec == 'uniform':
                        prob_r = random.random()
                        bin_i = round(prob_r * (num_bins-1)) + 1
                    elif type(randomize_spec) is tuple and randomize_spec[0] == 'normal':
                        prob_r = random.gauss(randomize_spec[1], randomize_spec[2])
                        prob_r = max(0.0, min(1.0, prob_r))
                        bin_i = round(prob_r * (num_bins-1)) + 1                        
            
            if bin_i > num_bins:
                print('bin_i wants to be ' + str(num_bins) + ' because of prob_pr ' + str(prob_pr))
                bin_i = min(bin_i, num_bins)
            kid_dict[gl_prediction_bin_tag] = bin_i
            #preference is the actual outcome of preferring apple snack (True) versus other snack (False)
            #Base preference on the kid's actual prob, not the distorted prob_pr.
            preference = random.random() < prob  
            preference_name = gl_preference_name_map[preference]
            kid_dict[gl_prediction_score_tag] = prob
            kid_dict[gl_outcome_tag] = preference_name
            kid_i += 1
            kid_list.append(kid_dict)
        
    return kid_list



def tallyFeatureBreakdowns(ddict_list, feature_name):
    value_count_map = {}
    for kid in ddict_list:
        val = kid.get(feature_name);
        if value_count_map.get(val) == None:
            value_count_map[val] = 0
        value_count_map[val] += 1

    value_count_list = []
    for key in value_count_map.keys():
        value_count_list.append((key, value_count_map.get(key)))
    value_count_list.sort(key=lambda x: x[0])
    return value_count_list



#https://stats.stackexchange.com/questions/171592/generate-random-numbers-with-linear-distribution
#return a sample from a distribution which is linear over the range min_val, max_val, with
#the probability at min_val being min_val_frac.
#min_val_frac can range from 0 to 2.0.
#A value of min_val_frac = 1.0 means use a uniform distribution.
def sampleLinear(min_val, max_val, min_val_frac):
    unif = random.random()
    alpha = 1.0 - min_val_frac
    if alpha == 0.0:
        x = unif
    else:
        x = (math.sqrt( alpha*alpha - 2*alpha + 4*alpha*unif + 1) - 1.0) / alpha
        x = (x + 1)/2.0

    val = min_val + x*(max_val - min_val)
    #print('min_val: ' + str(min_val) + ' max_val: ' + str(max_val) + ' min_val_frac: ' + str(min_val_frac) + ' x: ' + str(x) + ' val: ' + str(val))    
    return val

        

        

    
#Plot a  bar chart of no-recidivism (green) stacked on recidivism (red)
#Overplot the recidivisim rate in the color passed, per decile of the decile_scores.
#ddict_list is a list of dict of key/value pairs, including the keys, decile_score and is_recid.
#plot_what can be one of {'both', 'bars', 'ratio'}
def plotFeatureValue(ddict_list, feature_name, filter=None):
    if filter != None:
        ddict_list_test = ca.filterDdict(ddict_list, filter)
    else:
        ddict_list_test = ddict_list
    
    value_count_list = tallyFeatureBreakdowns(ddict_list_test, feature_name)
    
    y_max = 0
    for item in value_count_list:
        y_max = max(y_max, item[1])
    print('ymax: ' + str(y_max))
    plt.ylim(0, y_max)

    ind = [item[0] for item in value_count_list]
    vals = [item[1] for item in value_count_list]
    print('ind: ' + str(ind))
    print('vals: ' + str(vals))
           
    plt.plot(ind, vals)
    
    plt.show()




#slice_spec_list is a list or tuple of tuple string specifying sectors for the data set.
#Each tuple is a field-name-value-list which is a list of tuple, (str field_name, str comp, str or int field_value, slice_name)
#that must be true for the field to be included in the result
#e.g. ('das_b_screening_arrest', '<=', 30)
#This function prepends 'all-data' to slice_spec_list.
#Example:   ('sex', '==', 'girl')
def writePredictionsToJSONFileA(ddict_list, json_filepath, nickname, display_name, notes,
                                outcome_tag_tup = ('apple', 'other'),
                                slice_spec_list = None,
                                overwrite_existing_file_p=False, num_bins = None):

    if num_bins == None:
        num_bins = gl_num_bins

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

    pos_list = ca.filterDdict(ddict_list, [(gl_outcome_tag, '==', outcome_tag_tup[0])])
    pos_hist = ca.buildHistByDecile(pos_list, gl_prediction_bin_tag, num_bins)
    neg_list = ca.filterDdict(ddict_list, [(gl_outcome_tag, '==', outcome_tag_tup[1])])
    neg_hist = ca.buildHistByDecile(neg_list, gl_prediction_bin_tag, num_bins)
    json_dict['data-slices']['all-data'] = {'pos-outcomes': pos_hist,
                                            'neg-outcomes': neg_hist}

    if slice_spec_list == None:
        slice_spec_list = []
    for slice_spec in slice_spec_list:
        if len(slice_spec) > 3:
            slice_name = slice_spec[3]
        else:
            slice_name = slice_spec[2]

        ddict_list_slice = ca.filterDdict(ddict_list, [slice_spec])
        pos_list = ca.filterDdict(ddict_list_slice, [(gl_outcome_tag, '==', outcome_tag_tup[0])])
        pos_hist = ca.buildHistByDecile(pos_list, gl_prediction_bin_tag, num_bins)
        neg_list = ca.filterDdict(ddict_list_slice, [(gl_outcome_tag, '==', outcome_tag_tup[1])])
        neg_hist = ca.buildHistByDecile(neg_list, gl_prediction_bin_tag, num_bins)
        json_dict['data-slices'][slice_name] = {'pos-outcomes': pos_hist,
                                                'neg-outcomes': neg_hist}
        
    with open(json_filepath, 'w') as file:
        json.dump(json_dict, file, indent=4)






def generateAllDistributions(n_kids = 100000, n_bins=20):
    kdict_list_uniform = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_uniform)
    notes = gl_feature_dist_map_uniform.get('notes')
    writePredictionsToJSONFileA(kdict_list_uniform,
                                'C:/projects/ai-fairness/data/apple-snacks-uniform.json',
                                'apple-snacks-uniform', 'Apple Snacks Uniform',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)

    kdict_list_skewed_same = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_skewed_same)
    notes = gl_feature_dist_map_skewed_same.get('notes')
    writePredictionsToJSONFileA(kdict_list_skewed_same,
                                'C:/projects/ai-fairness/data/apple-snacks-skewed-same.json',
                                'apple-snacks-skewed-same', 'Apple Snacks Skewed Same',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)

    kdict_list_skewed_opposite = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_skewed_opposite)
    notes = gl_feature_dist_map_skewed_opposite.get('notes')
    writePredictionsToJSONFileA(kdict_list_skewed_opposite,
                                'C:/projects/ai-fairness/data/apple-snacks-skewed-opposite.json',
                                'apple-snacks-skewed-opposite', 'Apple Snacks Skewed Opposite',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)

    kdict_list_skewed_opposite_bias_girls = \
        generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_skewed_opposite_bias_girls)
    notes = gl_feature_dist_map_skewed_opposite_bias_girls.get('notes')
    writePredictionsToJSONFileA(kdict_list_skewed_opposite_bias_girls,
                                'C:/projects/ai-fairness/data/apple-snacks-skewed-opposite-bias-girls.json',
                                'apple-snacks-skewed-opposite-bias-girls', 'Apple Snacks Skewed Opposite Bias Girls',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)

    kdict_list_uniform_randomize_girls_uniform = \
        generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_uniform_randomize_girls_uniform)
    notes = gl_feature_dist_map_uniform_randomize_girls_uniform.get('notes')
    writePredictionsToJSONFileA(kdict_list_uniform_randomize_girls_uniform,
                                'C:/projects/ai-fairness/data/apple-snacks-uniform-randomize-girls-uniform.json',
                                'apple-snacks-uniform-randomize-girls-uniform', 'Apple Snacks Uniform Randomize Girls Uniform',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)
 
    kdict_list_uniform_randomize_girls_normal_p65_p15 = \
        generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_uniform_randomize_girls_normal_p65_p15)
    notes = gl_feature_dist_map_uniform_randomize_girls_normal_p65_p15.get('notes')
    writePredictionsToJSONFileA(kdict_list_uniform_randomize_girls_normal_p65_p15,
                                'C:/projects/ai-fairness/data/apple-snacks-uniform-randomize-girls_normal-p65-p15.json',
                                'apple-snacks-uniform-randomize-girls-normal-p65-p15', 'Apple Snacks Uniform Randomize Girls Normal .65/.15',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)
    
    

def generateAllDistributionsTemp(n_kids = 100000, n_bins=20):
    kdict_list_uniform_randomize_girls_uniform = \
        generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_uniform_randomize_girls_uniform)
    notes = gl_feature_dist_map_uniform_randomize_girls_uniform.get('notes')
    writePredictionsToJSONFileA(kdict_list_uniform_randomize_girls_uniform,
                                'C:/projects/ai-fairness/data/apple-snacks-uniform-randomize-girls-uniform.json',
                                'apple-snacks-uniform-randomize-girls-uniform', 'Apple Snacks Uniform Randomize Girls Uniform',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)
 
    kdict_list_uniform_randomize_girls_normal_p65_p15 = \
        generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_uniform_randomize_girls_normal_p65_p15)
    notes = gl_feature_dist_map_uniform_randomize_girls_normal_p65_p15.get('notes')
    writePredictionsToJSONFileA(kdict_list_uniform_randomize_girls_normal_p65_p15,
                                'C:/projects/ai-fairness/data/apple-snacks-uniform-randomize-girls-normal-p65-p15.json',
                                'apple-snacks-uniform-randomize-girls-normal-p65-p15', 'Apple Snacks Uniform Randomize Girls Normal .65/.15',
                                notes, ('apple', 'other'), [('sex', '==', 'girl'), ('sex', '==', 'boy')], True)
    
    


    


################################################################################
#
#Measure for algorithmic bias based on differences in the pos-prediction ratio per bin.
#2020/03/17
#

#The slice specs are of the form ('<feature-name>', '<comparator>', '<feature-value-name>')
#The two slice specs pick out two subsets of the data in dist_list to compare.
def computePosPredictionRatioScore(ddict_list, slice_spec_1, slice_spec_2, outcome_tag_tup = ('apple', 'other'),
                                   num_bins = None, plot_p = False):

    if num_bins == None:
        num_bins = gl_num_bins
    ddict_list_1 = ca.filterDdict(ddict_list, [slice_spec_1])
    pos_list_1 = ca.filterDdict(ddict_list_1, [(gl_outcome_tag, '==', outcome_tag_tup[0])])
    pos_hist_1 = ca.buildHistByDecile(pos_list_1, gl_prediction_bin_tag, num_bins)    
    neg_list_1 = ca.filterDdict(ddict_list_1, [(gl_outcome_tag, '==', outcome_tag_tup[1])])
    neg_hist_1 = ca.buildHistByDecile(neg_list_1, gl_prediction_bin_tag, num_bins)    

    ddict_list_2 = ca.filterDdict(ddict_list, [slice_spec_2])
    pos_list_2 = ca.filterDdict(ddict_list_2, [(gl_outcome_tag, '==', outcome_tag_tup[0])])
    pos_hist_2 = ca.buildHistByDecile(pos_list_2, gl_prediction_bin_tag, num_bins)        
    neg_list_2 = ca.filterDdict(ddict_list_2, [(gl_outcome_tag, '==', outcome_tag_tup[1])])
    neg_hist_2 = ca.buildHistByDecile(neg_list_2, gl_prediction_bin_tag, num_bins)    

    summ = 0
    denom_sum = 0
    w_binom_max_sum = 0

    ratio_1_ar = [-1] * num_bins   #-1 indicates invalid  value
    ratio_2_ar = [-1] * num_bins
    ratio_1_ar_sm = [-1] * num_bins
    ratio_2_ar_sm = [-1] * num_bins

    #stuff the ratio arrays and their smoothed versions with per-bin ratios
    for bin_i in range(num_bins):
        pos_1 = pos_hist_1[bin_i]
        neg_1 = neg_hist_1[bin_i]
        if pos_1 != 0 and neg_1 != 0:
            ratio_1 = pos_1 / (pos_1 + neg_1)
            ratio_1_ar[bin_i] = ratio_1
            ratio_1_ar_sm[bin_i] = ratio_1

        pos_2 = pos_hist_2[bin_i]
        neg_2 = neg_hist_2[bin_i]            
        if pos_2 != 0 and neg_2 != 0:
            ratio_2 = pos_2 / (pos_2 + neg_2)            
            ratio_2_ar[bin_i] = ratio_2
            ratio_2_ar_sm[bin_i] = ratio_2

    #smooth if possible
    for bin_i in range(1, num_bins-1):
        if ratio_1_ar[bin_i] != -1 and ratio_1_ar[bin_i-1] != -1 and ratio_1_ar[bin_i+1] != -1:
            ratio_1_ar_sm[bin_i] = .5 * ratio_1_ar[bin_i] + .25 * ratio_1_ar[bin_i-1] + .25 * ratio_1_ar[bin_i+1];
        if ratio_2_ar[bin_i] != -1 and ratio_2_ar[bin_i-1] != -1 and ratio_2_ar[bin_i+1] != -1:
            ratio_2_ar_sm[bin_i] = .5 * ratio_2_ar[bin_i] + .25 * ratio_2_ar[bin_i-1] + .25 * ratio_2_ar[bin_i+1];            

    for bin_i in range(num_bins):
        pos_1 = pos_hist_1[bin_i]
        neg_1 = neg_hist_1[bin_i]
        pos_2 = pos_hist_2[bin_i]
        neg_2 = neg_hist_2[bin_i]
        ratio_1 = ratio_1_ar_sm[bin_i]
        ratio_2 = ratio_2_ar_sm[bin_i]
        
        if ratio_1 == -1 or ratio_2 == -1:
            continue
        
        ratio_diff2 = (ratio_1 - ratio_2)*(ratio_1 - ratio_2)
        #ratio_diff2 = abs(ratio_1 - ratio_2) //prefer squared to abs for smoothed ratios
        denom = min((pos_1 + neg_1), (pos_2 + neg_2))
        weighted_ratio_diff = ratio_diff2 * denom
        denom_sum += denom

        cross_binom_1, cross_binom_2 = binomCrossProbs(pos_1, neg_1, pos_2, neg_2)
        cross_binom_1 = max(.0000000001, cross_binom_1)
        cross_binom_2 = max(.0000000001, cross_binom_2)
        nlog_cross_binom_1 = -math.log(cross_binom_1)
        nlog_cross_binom_2 = -math.log(cross_binom_2)
        cbinom_max = min(nlog_cross_binom_1, nlog_cross_binom_2)
        
        #cbinom_max = max(cross_binom_1, cross_binom_2)
        #cbinom_max = -math.log(cbinom_max)        
        w_binom_max = cbinom_max * denom
        w_binom_max_sum += w_binom_max 
        summ += weighted_ratio_diff
        
        #print('bin_i: ' + str(bin_i) + ' ratio_1: ' + str(ratio_1) + ' ratio_2: ' + str(ratio_2) + ' ratio_diff2: ' + str(ratio_diff2) + ' weighted_ratio_diff: ' + str(weighted_ratio_diff))
        #print('bin_i: ' + str(bin_i) + ' 1: {0:5} / {1:5} ratio1: {2:.3f}   2: {3:5} / {4:5}  ratio_2: {5:.3f}  ratio_diff^2:  {6:.3f}  denom: {7:5}  weighted: {8:.3f}  weighted_bm: {9:.3f}'.format(pos_1, neg_1, ratio_1, pos_2, neg_2, ratio_2, ratio_diff2, denom, weighted_ratio_diff, w_binom_max))
        
        print('bin_i: ' + str(bin_i) + ' 1: {0:5} / {1:5} ratio_1: {2:.3f}   2: {3:5} / {4:5}  ratio_2: {5:.3f}  max:  {6:.3f}  denom: {7:5}  weighted: {8:.3f}'.format(pos_1, neg_1, nlog_cross_binom_1, pos_2, neg_2, nlog_cross_binom_2, cbinom_max, denom, weighted_ratio_diff))

    pprs = summ / denom_sum
    ave_binom_max = w_binom_max_sum / denom_sum

    print('final_denom: ' + str(denom_sum * float(num_bins)))
    print('denom_sum: ' + str(denom_sum) + '  summ: ' + str(summ) + '   w_binom_max_sum: ' + str(w_binom_max_sum))
    print('pprs: ' + str(pprs) + '  ave_binom_max: ' + str(ave_binom_max))
    
    if plot_p:
        plt.ylim(0, 1.0)
        ind = range(num_bins)
        
        plt.plot(ind, ratio_1_ar, color='red')
        plt.plot(ind, ratio_2_ar, color='blue')
    
    plt.show()
        
    
    return pprs * 100.0, ave_binom_max


def computePPRSOnKids(n_kids, n_bins):
    
    kdict_list_uniform = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_uniform)
    pprs_u, abm_u = computePosPredictionRatioScore(kdict_list_uniform, ('sex', '==', 'girl'), ('sex', '==', 'boy'), ('apple', 'other'), n_bins)
    print('uniform: ' + str(pprs_u) + ' ave_binom_max: ' + str(abm_u))
    
    kdict_list_skewed_same = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_skewed_same)
    pprs_ss, abm_ss = computePosPredictionRatioScore(kdict_list_skewed_same, ('sex', '==', 'girl'), ('sex', '==', 'boy'), ('apple', 'other'), n_bins)
    print('skewed_same: ' + str(pprs_ss) + ' ave_binom_max: ' + str(abm_ss))
          
    kdict_list_skewed_opposite = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_skewed_opposite)
    pprs_o, abm_o = computePosPredictionRatioScore(kdict_list_skewed_opposite, ('sex', '==', 'girl'), ('sex', '==', 'boy'), ('apple', 'other'), n_bins)
    print('skewed_opposite ' + str(pprs_o) + ' ave_binom_max: ' + str(abm_o))
          
    kdict_list_uniform_bias_sex = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_uniform_bias_sex)
    pprs_bs, abm_bs = computePosPredictionRatioScore(kdict_list_uniform_bias_sex, ('sex', '==', 'girl'), ('sex', '==', 'boy'), ('apple', 'other'), n_bins)
    print('bias_sex: ' + str(pprs_bs) + ' ave_binom_max: ' + str(abm_bs))
    
    kdict_list_skewed_opposite_bias_girls = generateAppleSnackSamples(n_kids, n_bins, gl_feature_dist_map_skewed_opposite_bias_girls)
    pprs_obg, abm_obg = computePosPredictionRatioScore(kdict_list_skewed_opposite_bias_girls, ('sex', '==', 'girl'), ('sex', '==', 'boy'), ('apple', 'other'), n_bins)
    print('opposite_bias_girls: ' + str(pprs_obg) + ' ave_binom_max: ' + str(abm_obg))

    print('')
    print('uniform: ' + str(pprs_u) + ' ave_binom_max: ' + str(abm_u))
    print('skewed_same: ' + str(pprs_ss) + ' ave_binom_max: ' + str(abm_ss))
    print('skewed_opposite ' + str(pprs_o) + ' ave_binom_max: ' + str(abm_o))
    print('bias_sex: ' + str(pprs_bs) + ' ave_binom_max: ' + str(abm_bs))
    print('opposite_bias_girls: ' + str(pprs_obg) + ' ave_binom_max: ' + str(abm_obg))
        
        
    



def computePPRSOnKids1(n_kids, n_bins, feature_dist_map = None, plot_p=True):
    if feature_dist_map == None:
        feature_dist_map = gl_feature_dist_map_uniform
    
    kdict_list_uniform = generateAppleSnackSamples(n_kids, n_bins, feature_dist_map)
    pprs_u = computePosPredictionRatioScore(kdict_list_uniform, ('sex', '==', 'girl'), ('sex', '==', 'boy'), ('apple', 'other'), n_bins, plot_p)
    print('pprs: ' + str(pprs_u))


gl_feature_dist_map_list = [gl_feature_dist_map_uniform, gl_feature_dist_map_skewed_same, gl_feature_dist_map_skewed_opposite, gl_feature_dist_map_uniform_bias_sex, gl_feature_dist_map_skewed_opposite_bias_girls]

    
def computePPRSOnKidsN(n_kids, num_bins, feature_dist_map = None, plot_p=True):
    if type(feature_dist_map) is int:
        feature_dist_map = gl_feature_dist_map_list[feature_dist_map]
    elif feature_dist_map == None:
        feature_dist_map = gl_feature_dist_map_uniform
    
    kdict_list_uniform = generateAppleSnackSamples(n_kids, num_bins, feature_dist_map)
    pprs_u = computePosPredictionRatioScore(kdict_list_uniform, ('sex', '==', 'girl'), ('sex', '==', 'boy'), ('apple', 'other'), num_bins, plot_p)
    print('pprs: ' + str(pprs_u))

    
    

#https://en.wikipedia.org/wiki/Binomial_distribution
#
def binomCrossProbs(pos1, neg1, pos2, neg2):
    k1 = pos1
    n1 = pos1 + neg1
    p1 = pos1 / n1
    k2 = pos2
    n2 = pos2 + neg2
    p2 = pos2 / n2

    count = 0
    cross_binom_1 = binomMassP(n1, k1, p2)    
    while math.isnan(cross_binom_1) or math.isinf(cross_binom_1):
        count += 1
        if count > 10:
            print('!!!!!emergency exit')
            break
        n1 = n1/2
        k1 = k1/2
        cross_binom_1 = binomMassP(n1, k1, p2)

    count = 0        
    cross_binom_2 = binomMassP(n2, k2, p1)
    while math.isnan(cross_binom_2) or math.isinf(cross_binom_2):
        count += 1
        if count > 10:
            print('!!!!!emergency exit')
            break
        
        n2 = n2/2
        k2 = k2/2
        cross_binom_2 = binomMassP(n2, k2, p1)
        
    #print('cross_binom_1: ' + str(cross_binom_1) + '   cross_binom_2: ' + str(cross_binom_2))
    return cross_binom_1, cross_binom_2


def binomMassP(n, k, p):
    try:
        binom_mass = sp.binom(n, k) * math.pow(p, k) * math.pow(1.0-p, n-k)
        return binom_mass
    except:
        return binomMassP(n/2.0, k/2.0, p)
        

    
    
def bProbMass(n, k):
    p = float(k/n)
    #print('p: ' + str(p))
    #return sp.binom(n, k) * math.pow(p, k) * math.pow(1.0-p, (n-k))
    try:
        res = sp.binom(n, k) * math.pow(p, k) * math.pow(1.0-p, (n-k))
    except:
        print('bProbMass problem with n: ' + str(n) + ' k: ' + str(k))
        #res = sp.comb(n, k) * math.pow(p, k) * math.pow(1.0-p, (n-k))
    if math.isinf(res):
        return bProbMass(n/2.0, k/2.0)
    if math.isnan(res):
        return bProbMass(n/2.0, k/2.0)
    return res







#
#
################################################################################

################################################################################
#
#Archives
#



gl_value_index_to_value_maps_old = {'age':{0:6, 1:7, 2:8, 3:9, 4:10},
                                'sex':{0:'boy', 1:'girl'},
                                'teacher':{0:'Ms. Popcorn', 1:'Mr. Applebaum'},
                                'snack-preference':{False:'other', 1:'apple'}}

gl_value_to_value_index_maps_old = {'age':{6: 0, 7:1, 8:2, 9:3, 10:4},
                                'sex':{'boy':0, 'girl':1},
                                'teacher':{'Ms. Popcorn':0, 'Mr. Applebaum':1},
                                'snack-preference':{'other':False, 'apple':1}}

gl_prediction_score_tag_old = 'apple-prediction-score'
gl_prediction_bin_tag_old = 'apple-prediction-bin'

gl_outcome_tag_old = 'apple-preference'




#This data generator model generates kids and their preferences for having an apple
#snack or some other kind of snack.
#Each kid gets a groundtruth expected-preference score based on some features.
#The expected preference will be a probability from 0 to 1 that they want an apple snack.
#Then, the kid gets an observed preference outcome which is based on a biased coin flip,
#with the bias being their expected preference.
#This process generates a distribution of kids with observed preferences that
#more-or-less align with expectations.
#coeff_tup is a tuple of coefficients for the different features
#The expected-preference score formula is something like,
#  .5 +
#   (age - mean) * age_coeff +       age is in {6, 7, 8, 9, 10}  mean 8    older kids like apples
#   sex * sex_coeff +                sex is {-1 (boy), 1 (girl)            girls like apples
#   teacher * teacher_coeff          teacher is is {-1 (Ms. Popcorn), 1 (Mr. Applebaum)  Applebaum students like apples.
#
#
def generateAppleSnackSamples_old(num_kids, num_bins, coeff_tup,
                              age_dist = [.2, .2, .2, .2, .2], sex_dist = [.5, .5], teacher_dist = [.5, .5]):
    
    pos_outcome_hist = [0] * num_bins
    neg_outcome_hist = [0] * num_bins

    age_coeff = coeff_tup[0]
    sex_coeff = coeff_tup[1]
    teacher_coeff = coeff_tup[2]
    kid_list = []

    all_kids_feature_list = []      #list of tup of kid feature values
    for age_index in range(len(age_dist)):
        age_frac = age_dist[age_index]
        n_kids_at_age = age_frac * num_kids
        #print('n_kids_at_age: ' + str(n_kids_at_age))
        for sex_index in range(2):
            sex_frac = sex_dist[sex_index]
            n_kids_at_age_sex = n_kids_at_age * sex_frac
            #print('n_kids_at_age_sex: ' + str(n_kids_at_age_sex))            
            for teacher_index in range(2):
                teacher_frac = teacher_dist[teacher_index]
                n_kids_at_age_sex_teacher = int(n_kids_at_age_sex * teacher_frac)
                prob = age_coeff * (age_index-2) / 2.0 + \
                    sex_coeff * (sex_index - .5) * 2.0 + \
                    teacher_coeff * (teacher_index - .5) * 2.0
                prob = (prob + 1.0)/2.0
                bin_i = int(prob * num_bins) + 1   #bins are indexed from 1
                #print('n_kids_at_age_sex_teacher: ' + str(n_kids_at_age_sex_teacher))                            
                #print('age: ' + str(age_index) + ' sex: ' + str(sex_index) + ' teacher: ' + str(teacher_index) + ' prob: ' + str(prob))
                for i_kid in range(n_kids_at_age_sex_teacher):
                    preference = random.random() < prob

                    kid = {'age': gl_value_index_to_value_maps['age'][age_index],
                           'sex': gl_value_index_to_value_maps['sex'][sex_index],
                           'teacher': gl_value_index_to_value_maps['teacher'][teacher_index],
                           gl_prediction_score_tag: prob,
                           gl_prediction_bin_tag: bin_i,                           
                           gl_outcome_tag: gl_value_index_to_value_maps['snack-preference'][preference]}
                    kid_list.append(kid)
                    #print(str(i_kid) + ' ' + str(kid))
    return kid_list

        

def generateAppleSnackSamples_old(num_kids, num_bins = gl_num_bins, feature_dist_map_name = None):
    if feature_dist_map_name == None:
        feature_dist_map_name = gl_feature_dist_map_name
    feature_dist_map = gl_feature_dist_map_control[feature_dist_map_name]
    print('using feature distribution: ' + feature_dist_map_name)
        
    kid_list = []

    min_val, max_val, range_min_score, range_max_score = determineScoreRange(feature_dist_map_name)

    for sex in ('girl', 'boy'):
        sex_index = gl_sex_index_map[sex]
        num_of_sex = int(num_kids * gl_sex_proportions[sex_index])
        for kid_i in range(num_of_sex):
            #print('')

            feature_score_sum = 0.0
            kid_dict = {'sex':sex}
            for feature_name in feature_dist_map[sex].keys():
                feature_scale = feature_dist_map[sex][feature_name].get('scale')
                feature_offset = feature_dist_map[sex][feature_name].get('offset')
                normal_spec = feature_dist_map[sex][feature_name].get('normal')
                uniform_spec = feature_dist_map[sex][feature_name].get('uniform')
                triangular_spec = feature_dist_map[sex][feature_name].get('triangular')
                linear_spec = feature_dist_map[sex][feature_name].get('linear')                
                categorical_spec = feature_dist_map[sex][feature_name].get('categorical')
                
                if normal_spec != None:
                    feature_value = random.gauss(normal_spec[0], normal_spec[1])
                    count = 0;
                    while feature_value < normal_spec[2] or feature_value > normal_spec[3]:
                        feature_value = random.gauss(normal_spec[0], normal_spec[1])                            
                        count += 1
                        if count > 100:
                            print('count exceeded 100 while sampling a normal distribution with mean ' + str(normal_spec[0]) + ' stdev: ' + str(normal_spec[1]) + ' min limit: ' + str(normal_spec[2]) + ' max_limit: ' + str(normal_spec[2]))

                    rr = normal_spec[3] - normal_spec[2];
                    rrb = 50.0/rr
                    
                    feature_value = max(normal_spec[2], min(normal_spec[3], feature_value))
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value

                elif uniform_spec != None:
                    feature_value = random.uniform(uniform_spec[0], uniform_spec[1])
                    rr = uniform_spec[1] - uniform_spec[0];
                    rrb = 50.0/rr  
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value

                elif triangular_spec != None:
                    feature_value = random.triangular(triangular_spec[0], triangular_spec[1], triangular_spec[2])
                    rr = triangular_spec[1] - triangular_spec[0];
                    rrb = 50.0/rr  
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value

                #linear spec is (min_val, max_val, slope_dir, frac_uniform, frac_triangular)                             
                elif linear_spec != None:
                    feature_value = sampleLinear(linear_spec[0], linear_spec[1], linear_spec[2])
                    rr = linear_spec[1] - linear_spec[0];
                    rrb = 50.0/rr  
                    feature_value = round(feature_value * rrb) / rrb
                    feature_score = (feature_value - feature_offset) * feature_scale
                    kid_dict[feature_name] = feature_value
                    
                elif categorical_spec != None:
                    dist_tup = feature_dist_map[sex][feature_name].get('dist')
                    feature_cum_prob_ar = []
                    feature_cum_prob = 0.0
                    for feature_index in range(len(dist_tup)):
                        #print('feature_name: ' + feature_name + ' dist_tup: ' + str(dist_tup) + ' feature_index: ' + str(feature_index))
                        feature_cum_prob += dist_tup[feature_index]
                        feature_cum_prob_ar.append(feature_cum_prob)
                    feature_cum_prob_ar[len(feature_cum_prob_ar)-1] = 1.0
                    pp = random.random()
                    for feature_index in range(len(feature_cum_prob_ar)):
                        if pp < feature_cum_prob_ar[feature_index]:
                            break
                    feature_value = categorical_spec[feature_index]
                    feature_score = (feature_value - feature_offset) * feature_scale
                    feature_value_name = feature_dist_map[sex][feature_name].get('names')[feature_index]
                    #if feature_name == 'teacher':
                    #    print('cum: ' + str(feature_cum_prob_ar) + ' pp: ' + str(pp) + ' feature_index: ' + str(feature_index) + ' feature_value: ' + str(feature_value))
                    
                    kid_dict[feature_name] = feature_value_name

                #print('    feature_name: ' + feature_name + ' feature_score: ' + str(feature_score))
                feature_score_sum += feature_score
                

            prob = (feature_score_sum - range_min_score) / (range_max_score - range_min_score)
            #print('feature_name: ' + feature_name + ' feature_score_sum: ' + str(feature_score_sum) + ' prob: ' + str(prob))
            bin_i = int(prob * (num_bins-1)) + 1   #bins are indexed from 1
            if bin_i == num_bins:
                print('bin_i wants to be ' + str(num_bins) + ' because of prob ' + str(prob))
                bin_i = min(bin_i, num_bins-1)
            kid_dict[gl_prediction_bin_tag] = bin_i
            preference = random.random() < prob     
            preference_name = gl_preference_name_map[preference]
            kid_dict[gl_prediction_score_tag] = prob
            kid_dict[gl_outcome_tag] = preference_name
            kid_list.append(kid_dict)
        
    return kid_list



#This older version outputs histograms.
#Instead use the version above that output simulated kids and all of their features.
def generateAppleSnackSamplePredScores_obosolete(num_kids, num_bins, coeff_tup,
                                       age_dist = [.2, .2, .2, .2, .2], sex_dist = [.5, .5], teacher_dist = [.5, .5]):
    
    pos_outcome_hist = [0] * num_bins
    neg_outcome_hist = [0] * num_bins

    age_coeff = coeff_tup[0]
    sex_coeff = coeff_tup[1]
    teacher_coeff = coeff_tup[2]

    all_kids_feature_list = []    #list of tup of kid feature values
    for age_index in range(len(age_dist)):
        age_frac = age_dist[age_index]
        n_kids_at_age = age_frac * num_kids
        #print('n_kids_at_age: ' + str(n_kids_at_age))
        for sex_index in range(2):
            sex_frac = sex_dist[sex_index]
            n_kids_at_age_sex = n_kids_at_age * sex_frac
            #print('n_kids_at_age_sex: ' + str(n_kids_at_age_sex))            
            for teacher_index in range(2):
                teacher_frac = teacher_dist[teacher_index]
                n_kids_at_age_sex_teacher = int(n_kids_at_age_sex * teacher_frac)
                prob = age_coeff * (age_index-2) / 2.0 + \
                    sex_coeff * (sex_index - .5) * 2.0 + \
                    teacher_coeff * (teacher_index - .5) * 2.0
                prob = (prob + 1.0)/2.0
                
                #print('n_kids_at_age_sex_teacher: ' + str(n_kids_at_age_sex_teacher))                            
                #print('age: ' + str(age_index) + ' sex: ' + str(sex_index) + ' teacher: ' + str(teacher_index) + ' prob: ' + str(prob))
                for i_kid in range(n_kids_at_age_sex_teacher):
                    outcome = random.random() < prob
                    #if outcome:
                    #outcome = 1
                    #else:
                    #outcome = 0
                    #print('age: ' + str(age_index) + ' sex: ' + str(sex_index) + ' teacher: ' + str(teacher_index) + ' prob: ' + str(prob) + ' outcome: ' + str(outcome))

                    bin_i = int(prob * num_bins)
                    if outcome:
                        pos_outcome_hist[bin_i] += 1
                    else:
                        neg_outcome_hist[bin_i] += 1

    return pos_outcome_hist, neg_outcome_hist

                        
def writePosNegHistsToJSONFile(pos_outcome_hist, neg_outcome_hist, nickname, display_name, json_filepath, notes = ''):
    json_dict = {'data-set-nickname': nickname,
                 'data-set-display-name': display_name, 
                 'notes': notes,
                 'data-slices': {}
                 }

    json_dict['data-slices']['all-data'] = {'pos-outcomes': pos_outcome_hist,
                                            'neg-outcomes': neg_outcome_hist}
    with open(json_filepath, 'w') as file:
        json.dump(json_dict, file, indent=4)




def invertValueMapDict(vi_to_v_maps):
    v_to_vi_maps = {}
    for cat in vi_to_v_maps.keys:
        vi_to_v_map = vi_to_v_maps[cat]
        v_to_vi_maps[cat] = {v: k for k, v in vi_to_v_map.items()}
    return v_to_vi_maps

    
    



#The slice specs are of the form ('<feature-name>', '<comparator>', '<feature-value-name>')
#The two slice specs pick out two subsets of the data in dist_list to compare.
def computePosPredictionRatioScore_save(ddict_list, slice_spec_1, slice_spec_2, outcome_tag_tup = ('apple', 'other'),
                                   num_bins = None, plot_p = False):

    if num_bins == None:
        num_bins = gl_num_bins
    ddict_list_1 = ca.filterDdict(ddict_list, [slice_spec_1])
    pos_list_1 = ca.filterDdict(ddict_list_1, [(gl_outcome_tag, '==', outcome_tag_tup[0])])
    pos_hist_1 = ca.buildHistByDecile(pos_list_1, gl_prediction_bin_tag, num_bins)    
    neg_list_1 = ca.filterDdict(ddict_list_1, [(gl_outcome_tag, '==', outcome_tag_tup[1])])
    neg_hist_1 = ca.buildHistByDecile(neg_list_1, gl_prediction_bin_tag, num_bins)    

    ddict_list_2 = ca.filterDdict(ddict_list, [slice_spec_2])
    pos_list_2 = ca.filterDdict(ddict_list_2, [(gl_outcome_tag, '==', outcome_tag_tup[0])])
    pos_hist_2 = ca.buildHistByDecile(pos_list_2, gl_prediction_bin_tag, num_bins)        
    neg_list_2 = ca.filterDdict(ddict_list_2, [(gl_outcome_tag, '==', outcome_tag_tup[1])])
    neg_hist_2 = ca.buildHistByDecile(neg_list_2, gl_prediction_bin_tag, num_bins)    

    summ = 0
    denom_sum = 0
    w_binom_max_sum = 0

    ratio_1_ar = [0] * num_bins
    ratio_2_ar = [0] * num_bins
    ratio_1_ar = [0] * num_bins
    ratio_2_ar = [0] * num_bins

    for bin_i in range(num_bins):
        pos_1 = pos_hist_1[bin_i]
        neg_1 = neg_hist_1[bin_i]
        if pos_1 == 0 and neg_1 == 0:
            continue
        pos_2 = pos_hist_2[bin_i]
        neg_2 = neg_hist_2[bin_i]
        if pos_2 == 0 and neg_2 == 0:
            continue
        ratio_1 = pos_1 / (pos_1 + neg_1)
        ratio_2 = pos_2 / (pos_2 + neg_2)
        ratio_1_ar[bin_i] = ratio_1
        ratio_2_ar[bin_i] = ratio_2
        ratio_diff2 = (ratio_1 - ratio_2)*(ratio_1 - ratio_2)
        #ratio_diff2 = abs(ratio_1 - ratio_2) //prefer squared to abs for smoothed ratios
        denom = min((pos_1 + neg_1), (pos_2 + neg_2))
        weighted_ratio_diff = ratio_diff2 * denom
        denom_sum += denom

        cross_binom_1, cross_binom_2 = binomCrossProbs(pos_1, neg_1, pos_2, neg_2)
        cross_binom_1 = max(.0000000001, cross_binom_1)
        cross_binom_2 = max(.0000000001, cross_binom_2)
        nlog_cross_binom_1 = -math.log(cross_binom_1)
        nlog_cross_binom_2 = -math.log(cross_binom_2)
        cbinom_max = min(nlog_cross_binom_1, nlog_cross_binom_2)
        
        #cbinom_max = max(cross_binom_1, cross_binom_2)
        #cbinom_max = -math.log(cbinom_max)        
        ##cbinom_max = 1.0/cbinom_max
        ##cbinom_max = 1.0/max(.001, cbinom_max)
        w_binom_max = cbinom_max * denom
        w_binom_max_sum += w_binom_max 
        summ += weighted_ratio_diff
        
        #print('bin_i: ' + str(bin_i) + ' ratio_1: ' + str(ratio_1) + ' ratio_2: ' + str(ratio_2) + ' ratio_diff2: ' + str(ratio_diff2) + ' weighted_ratio_diff: ' + str(weighted_ratio_diff))
        #print('bin_i: ' + str(bin_i) + ' 1: {0:5} / {1:5} ratio1: {2:.3f}   2: {3:5} / {4:5}  ratio_2: {5:.3f}  ratio_diff^2:  {6:.3f}  denom: {7:5}  weighted: {8:.3f}  weighted_bm: {9:.3f}'.format(pos_1, neg_1, ratio_1, pos_2, neg_2, ratio_2, ratio_diff2, denom, weighted_ratio_diff, w_binom_max))
        
        print('bin_i: ' + str(bin_i) + ' 1: {0:5} / {1:5} nlog_cbinom_1: {2:.3f}   2: {3:5} / {4:5}  nlog_cbinom_2: {5:.3f}  max:  {6:.3f}  denom: {7:5}  weighted: {8:.3f}  weighted_bm: {9:.3f}'.format(pos_1, neg_1, nlog_cross_binom_1, pos_2, neg_2, nlog_cross_binom_2, cbinom_max, denom, weighted_ratio_diff, w_binom_max))

    pprs = summ / denom_sum
    ave_binom_max = w_binom_max_sum / denom_sum

    print('final_denom: ' + str(denom_sum * float(num_bins)))
    print('denom_sum: ' + str(denom_sum) + '  summ: ' + str(summ) + '   w_binom_max_sum: ' + str(w_binom_max_sum))
    print('pprs: ' + str(pprs) + '  ave_binom_max: ' + str(ave_binom_max))
    
    if plot_p:
        plt.ylim(0, 1.0)
        ind = range(num_bins)
        
        plt.plot(ind, ratio_1_ar, color='red')
        plt.plot(ind, ratio_2_ar, color='blue')
    
    plt.show()
        
    
    return pprs * 100.0, ave_binom_max


    
