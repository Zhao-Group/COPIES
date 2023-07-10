"""
 Copyright 2023 University of Illinois Board of Trustees. All Rights Reserved.
 Carl R. Woese Institute for Genomic Biology
 This file is part of CRISPR-COPIES, which is released under specific terms.  See file 'LICENSE' for more details.
"""

################################################################### 
## Primary Author:  Aashutosh Girish Boob aboob2@illinois.edu
## Co-Authors: Vassily Andrew Petrov vassily2@illinois.edu
## License: Apache License 2.0
## Repository:  https://github.com/HuiminZhao/COPIES
###################################################################

## Credit: parts of this code make use of the code or pre-trained models from
##    - https://github.com/USDA-ARS-GBRU/GuideMaker
##    - https://github.com/H2muller/CROPSR
##    - https://github.com/dDipankar/DeepGuide
##    - https://github.com/zhangchonglab/sgRNA-cleavage-activity-prediction

import random,sys,os

#paths
# directories will be relative to script source.
temporary = "../ncbi-blast-2.12.0+/tmp/"
deg_file = '../essential_genes/deg.csv'
blast_path = os.getcwd() + '/ncbi-blast-2.12.0+/bin/'

#Modules
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import re
from collections import Counter
import argparse
import os
from Bio.Blast.Applications import NcbiblastpCommandline
import distance
import scann
from Bio.SeqUtils import GC
from Bio.SeqUtils import MeltingTemp
import Bio.SeqUtils.MeltingTemp as Tm
import doench_predict
import sys
import multiprocessing as mp
import math
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from scipy.stats import spearmanr, pearsonr
from scipy.stats import gaussian_kde
import pickle
import h5py as h5
from keras.models import Sequential, Model
from keras.layers.core import  Dropout, Activation, Flatten
from keras.regularizers import l1,l2,l1_l2
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Bidirectional, BatchNormalization, MaxPooling2D, AveragePooling1D, Input, Multiply, Add, UpSampling1D
import time

NUM_THREADS = mp.cpu_count()

# change working directory to script directory
os.chdir(os.path.dirname(sys.argv[0]))

if os.path.exists(temporary):
    print("Directory exists.")
else:
    os.mkdir(temporary)

scann_time = 0
blast_time = 0

#Functions
def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name),'fasta')
    data = []
    for fasta in fasta_seqs:
        if not ('mitochondrion' in fasta.description or 'plastid' in fasta.description or 'chloroplast' in fasta.description): #Considering only nuclear chromosomes (includes plasmid/megaplasmid), removing mitochondrial and plastid/chloroplast genomes
            data.append([fasta.id, str(fasta.seq).strip().upper()])
    
    return data
    
def pam_to_search(pam, iupac_code):
    pam_string = list(pam)
    pam_seq = []
    for i in range(len(pam_string)):
        curr_char = iupac_code[pam_string[i]].split('|')
        if i == 0:
            pam_seq = curr_char
        else:
            curr_pam_seq = []
            for j in range(len(curr_char)):
                for k in range(len(pam_seq)):
                    curr_pam_seq.append(pam_seq[k]+curr_char[j])
                    
            pam_seq = curr_pam_seq
    
    return pam_seq

def one_hot(seq_list):
    d_len = len(seq_list[0])
    ll = np.frombuffer(bytes("".join(seq_list), "ascii"), dtype=np.uint8)
    ll = np.reshape(ll, (len(seq_list), d_len))
    ll = np.array(ll)
    ll[ ll == ord("A") ] = 0b1000
    ll[ ll == ord("C") ] = 0b0100
    ll[ ll == ord("G") ] = 0b0010
    ll[ ll == ord("T") ] = 0b0001
    ll = np.reshape(ll, (len(seq_list), d_len, 1))
    one_hot_list = np.unpackbits(ll, axis=2)[:,:,4:]
    del ll
    one_hot_list = np.reshape(one_hot_list, (len(seq_list), 4*d_len))
    return one_hot_list.astype(np.float32)

def check_intergenic(gtable, chrom, loc, strand, chrom_len, intergenic_space):
    flag = True
    #getting the gene list present on the chromsomose of interest
    gloi = gtable.loc[gtable['Accession'] == chrom].reset_index(drop=True)
    if len(gloi) > 0:
        gloi_start = gloi['Start'].to_numpy()
        gloi_stop = gloi['Stop'].to_numpy()

        if strand == '+':
            for i in range(len(gloi)):
                if loc > gloi_start[i] - intergenic_space and loc < gloi_stop[i] + intergenic_space:
                    flag = False
                    break
        else:
            for i in range(len(gloi)):
                if chrom_len - loc > gloi_start[i] - intergenic_space and chrom_len - loc < gloi_stop[i] + intergenic_space:
                    flag = False
                    break

    return flag

def get_gene_info(gtable, chrom, loc, strand, chrom_len, gdenslen):
    #getting the gene list present on the chromsomose of interest
    gloi = gtable.loc[gtable['Accession'] == chrom].sort_values('Start').reset_index(drop=True)
    if len(gloi) > 0:
        if strand == '+':
            curr_loc = loc
        else:
            curr_loc = chrom_len - loc

        #left gene, right gene is defined with respect to the top strand [gRNA orientation not taken into account]
        if curr_loc < gloi['Start'][0]:
            right_gene = gloi['Locus tag'][0]
            intg_region_size = '-'
            relative_orient = '-'
            left_gene = '-'
        elif curr_loc > list(gloi['Stop'])[-1]:
            left_gene = list(gloi['Locus tag'])[-1]
            intg_region_size = '-'
            relative_orient = '-'
            right_gene = '-'
        else:
            stop_index = len([x for x in list(gloi['Stop']) if x < curr_loc]) - 1
            intg_region_size = gloi['Start'][stop_index+1] - gloi['Stop'][stop_index]
            left_gene = gloi['Locus tag'][stop_index]
            right_gene = gloi['Locus tag'][stop_index+1]
            
            left_gene_strand = gloi['Strand'][stop_index]
            right_gene_strand = gloi['Strand'][stop_index+1]
            
            if left_gene_strand == right_gene_strand:
                relative_orient = 'tandem'
            elif left_gene_strand == '+' and right_gene_strand == '-':
                relative_orient = 'convergent'
            else:
                relative_orient = 'divergent'

        if curr_loc - gdenslen < 0:
            l_limit = 0
        else: 
            l_limit = curr_loc - gdenslen
            
        if curr_loc + gdenslen > chrom_len:
            u_limit = chrom_len
        else:
            u_limit = curr_loc + gdenslen
            
        gene_density = len(list(set([i for i,v in enumerate(list(gloi['Start'])) if v > l_limit]) & set([i for i,v in enumerate(list(gloi['Stop'])) if v < u_limit])))
        
    else:
        intg_region_size = '-'
        left_gene = '-'
        right_gene = '-'
        relative_orient = '-'
        gene_density = '-'
        
    return [intg_region_size, left_gene, right_gene, relative_orient, gene_density]    

def grna_search(genome, pam_l, glen, orient):
    grna_list = []
    #'for' loop for pam sequences
    for i in range(len(pam_l)):
        curr_pam = pam_l[i]
        
        #'for' loop for chromosomes
        for j in range(len(genome)):
            #top_strand
            chrom_seq = genome[j][1]
            if orient == '3prime':
                curr_grna_loc = [x - glen for x in [m.start() for m in re.finditer(curr_pam, chrom_seq)]]
            else: 
                curr_grna_loc = [m.start() for m in re.finditer(curr_pam, chrom_seq)]
                
            for k in range(len(curr_grna_loc)):
                if curr_grna_loc[k] > -1 and curr_grna_loc[k] < len(chrom_seq) - glen - len(curr_pam):
                    grna_list.append([chrom_seq[curr_grna_loc[k]:curr_grna_loc[k] + glen + len(curr_pam)], genome[j][0], curr_grna_loc[k], '+', len(chrom_seq)])
                    
            #bottom_strand
            chrom_seq = str(Seq(genome[j][1]).reverse_complement())
            if orient == '3prime':
                curr_grna_loc = [x - glen for x in [m.start() for m in re.finditer(curr_pam, chrom_seq)]]
            else: 
                curr_grna_loc = [m.start() for m in re.finditer(curr_pam, chrom_seq)]
                
            for k in range(len(curr_grna_loc)):
                if curr_grna_loc[k] > -1 and curr_grna_loc[k] < len(chrom_seq) - glen - len(curr_pam):
                    grna_list.append([chrom_seq[curr_grna_loc[k]:curr_grna_loc[k] + glen + len(curr_pam)], genome[j][0], curr_grna_loc[k], '-', len(chrom_seq)])
    
    return grna_list
   
def grna_prefilter(grna_list, glen, pam, orient, seedlen, re_grna_list, polyG_len, polyT_len, ambiguous_nucleotides, gc_limits):
    #get grna occurrence table (without PAM)
    if orient == '3prime':
        grna_without_pam = [item[0][0:glen] for item in grna_list]    
    elif orient == '5prime':
        grna_without_pam = [item[0][len(pam):] for item in grna_list]  
        
    #remove ambiguous nucleotides
    grna_without_pam_f = [word for word in grna_without_pam if not any(bad in word for bad in ambiguous_nucleotides)]
    
    grna_occurrence = pd.DataFrame.from_dict(Counter(grna_without_pam_f), orient='index').reset_index()
    grna_occurrence.columns = ['grna', 'frequency']

    #get all guide sequences occuring in genome with duplicates removed (definition of duplicate: sequence occurring in multiple places)
    complete_grna_library_wo_pam = list(grna_occurrence['grna']) 
    
    #get all guide sequences occuring in genome once
    grna_once_wo_pam_library_f = list(grna_occurrence.loc[grna_occurrence['frequency'] == 1]['grna'])
    
    #get all seed sequences occuring in genome
    if orient == '3prime':
        seed_region = [item[glen-seedlen:] for item in complete_grna_library_wo_pam]    
    elif orient == '5prime':
        seed_region = [item[:seedlen] for item in complete_grna_library_wo_pam]
    
    seed_occurrence = pd.DataFrame.from_dict(Counter(seed_region), orient='index').reset_index()
    seed_occurrence.columns = ['seed_seq', 'frequency']
    complete_seed_library_with_unique_seed = list(seed_occurrence.loc[seed_occurrence['frequency'] == 1, 'seed_seq'])

    #gRNA RE check
    re_to_check_grna_pre = re_grna_list.split(',')   
    
    if not '' in re_to_check_grna_pre:
        #Taking care of ambiguous nucleotides
        re_to_check_grna = []
        for i in range(len(re_to_check_grna_pre)):
            re_to_check_grna.extend(pam_to_search(re_to_check_grna_pre[i] ,iupac_code))
        
        #checking provided recognition sequence
        grna_wo_pam_f = [w for w in grna_once_wo_pam_library_f if not any(b in w for b in re_to_check_grna)] #f stands for filter
        
        #checking reverse complement of provided recognition sequence
        revcom_re_to_check_grna = []
        for i in range(len(re_to_check_grna)):
            revcom_re_to_check_grna.append(str(Seq(re_to_check_grna[i]).reverse_complement()))
            
        grna_wo_pam_f = [w for w in grna_wo_pam_f if not any(b in w for b in revcom_re_to_check_grna)]
    else: 
        grna_wo_pam_f = grna_once_wo_pam_library_f
        
    #GC content check
    low_limit, up_limit = [int(x) for x in gc_limits.split(",")]
    if low_limit == 0 and up_limit == 100:
        grna_wo_pam_f = grna_wo_pam_f
    else:
        grna_wo_pam_f = [w for w in grna_wo_pam_f if(low_limit <= GC(w) <= up_limit)]
        
    #gRNA polyG check
    if not polyG_len == 0:
        polyG_to_check = "G" * polyG_len
        grna_wo_pam_f = [w for w in grna_wo_pam_f if polyG_to_check not in w]
        
        polyC_to_check = "C" * polyG_len
        grna_wo_pam_f = [w for w in grna_wo_pam_f if polyC_to_check not in w]
        
    #gRNA polyT check
    if not polyT_len == 0:
        polyT_to_check = "T" * polyT_len
        grna_wo_pam_f = [w for w in grna_wo_pam_f if polyT_to_check not in w]
        
        polyA_to_check = "A" * polyT_len
        grna_wo_pam_f = [w for w in grna_wo_pam_f if polyA_to_check not in w]
        
    #get guides sequences with a unique seed
    if orient == '3prime':
        grna_wo_pam_f_seed_region = [item[glen-seedlen:] for item in grna_wo_pam_f]    
    elif orient == '5prime':
        grna_wo_pam_f_seed_region = [item[:seedlen] for item in grna_wo_pam_f]    

    unique_seed_library = list(set(grna_wo_pam_f_seed_region) & set(complete_seed_library_with_unique_seed))
    seed_to_compare_dict = dict(zip(grna_wo_pam_f_seed_region,range(0,len(grna_wo_pam_f_seed_region))))
    grna_wo_pam_f_index = [seed_to_compare_dict.get(key) for key in unique_seed_library]
    grna_wo_pam_us_f = [grna_wo_pam_f[i] for i in grna_wo_pam_f_index]
    
    return complete_grna_library_wo_pam, grna_wo_pam_us_f
   
def grna_filter(grna_list, glen, pam, orient, seedlen, re_grna_list, polyG_len, polyT_len, edit_dist, gtable, intergenic_space, gdenslen, ambiguous_nucleotides, gc_limits, dist_type):
    
    complete_grna_library_wo_pam, grna_wo_pam_us_f = grna_prefilter(grna_list, glen, pam, orient, seedlen, re_grna_list, polyG_len, polyT_len, ambiguous_nucleotides, gc_limits)
    
    #off-target using filtered_grna_library and complete_grna_library
    xb = one_hot(complete_grna_library_wo_pam)
    norm_xb = xb/np.round(np.sqrt(glen),4)
    del xb
    
    train_data_size = len(complete_grna_library_wo_pam)
    number_of_partitions = int(np.sqrt(train_data_size))
    number_of_partition_to_search = int(number_of_partitions/20)
    
    start_scann = time.perf_counter()
    searcher = scann.scann_ops_pybind.builder(norm_xb, 10, "dot_product").tree(
    num_leaves = number_of_partitions, num_leaves_to_search = number_of_partition_to_search, training_sample_size=train_data_size).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    del norm_xb
    
    xq = one_hot(grna_wo_pam_us_f)
    len_xq = len(xq)
    neighbors = searcher.search_batched(xq, leaves_to_search=number_of_partition_to_search, pre_reorder_num_neighbors=number_of_partition_to_search)[0]
    del xq
    stop_scann = time.perf_counter()
    global scann_time
    scann_time = stop_scann - start_scann
    
    unique_grna_library_mm = []
    nearest_neighbor = []
    k_to_check = 3 #knearest neighbor search
    if dist_type == 'hamming':
        for i in range(len_xq):
            knn_dist = []
            for j in range(k_to_check):
                knn_dist.append(distance.hamming(grna_wo_pam_us_f[i], complete_grna_library_wo_pam[neighbors[i][j+1]]))

            if all(i > edit_dist - 1 for i in knn_dist):
                unique_grna_library_mm.append(grna_wo_pam_us_f[i])
                nearest_neighbor.append(str(knn_dist[0]) + ',' + complete_grna_library_wo_pam[neighbors[i][1]]) 
    else:
        for i in range(len_xq):
            knn_dist = []
            for j in range(k_to_check):
                knn_dist.append(distance.levenshtein(grna_wo_pam_us_f[i], complete_grna_library_wo_pam[neighbors[i][j+1]]))

            if all(i > edit_dist - 1 for i in knn_dist):
                unique_grna_library_mm.append(grna_wo_pam_us_f[i])
                nearest_neighbor.append(str(knn_dist[0]) + ',' + complete_grna_library_wo_pam[neighbors[i][1]])
                
    if orient == '3prime':
        grna_to_compare = [item[0][:glen] for item in grna_list]    
    elif orient == '5prime':
        grna_to_compare = [item[0][len(pam):len(pam)+glen] for item in grna_list]    

    grna_to_compare_dict = dict(zip(grna_to_compare,range(0,len(grna_to_compare))))
    grna_index = [grna_to_compare_dict.get(key) for key in unique_grna_library_mm]
    grna_list_mm = [grna_list[i] for i in grna_index]

    for i in range(len(grna_list_mm)):
	    grna_list_mm[i] = grna_list_mm[i] + [nearest_neighbor[i]]
    
    del grna_to_compare, grna_to_compare_dict
    
    #select intergenic gRNA
    if len(grna_list_mm) > 0:
        index_to_keep = []
        for i in range(len(grna_list_mm)):
            if check_intergenic(gtable, grna_list_mm[i][1], grna_list_mm[i][2], grna_list_mm[i][3], grna_list_mm[i][4], intergenic_space):
                index_to_keep.append(i)

        grna_list_mm_intg = [grna_list_mm[i] for i in index_to_keep]
        #intergenic region size, adjacent genes and gene density
        for i in range(len(grna_list_mm_intg)):
            grna_list_mm_intg[i] = grna_list_mm_intg[i] + get_gene_info(gtable, grna_list_mm_intg[i][1], grna_list_mm_intg[i][2], grna_list_mm_intg[i][3], grna_list_mm_intg[i][4], gdenslen)
    
    return grna_list_mm_intg

def hr_filter(data, glen, pam, genome, hr_len, RE_hr, polyG_hr, polyT_hr):
    for i in range(len(data)):
        #get the chromosome seq based on data[i][1]; use genome
        chrom_seq = genome[[chrom_name[0] for chrom_name in genome].index(data[i][1])][1]
        
        #pam length 
        pam_len = len(pam)
        if data[i][3] == '+':
            curr_chrom_seq = chrom_seq
        else:
            curr_chrom_seq = str(Seq(chrom_seq).reverse_complement())
            
        if data[i][2] - hr_len > 0:
            LHR = curr_chrom_seq[data[i][2] - hr_len:data[i][2]]
        else:
            LHR = '-'
            
        if data[i][2] + glen + pam_len + hr_len < data[i][4]:
            RHR = curr_chrom_seq[data[i][2] + glen + pam_len:data[i][2] + glen + pam_len + hr_len]
        else:
            RHR = '-' 
            
        data[i].extend([LHR, RHR])
    
    LHR_list = [lhr_seq[-2] for lhr_seq in data]
    RHR_list = [rhr_seq[-1] for rhr_seq in data]
    
    index_to_remove = []
    #RE_check
    re_to_check_hr_pre = RE_hr.split(',')   
    if not '' in re_to_check_hr_pre:
        #Taking care of ambiguous nucleotides
        re_to_check_hr = []
        for i in range(len(re_to_check_hr_pre)):
            re_to_check_hr.extend(pam_to_search(re_to_check_hr_pre[i] ,iupac_code))
        
        for i in range(len(data)):
            for j in range(len(re_to_check_hr)):
                if re_to_check_hr[j] in LHR_list[i] or re_to_check_hr[j] in RHR_list[i]:
                    index_to_remove.append(i)

    #PolyG_check
    if not polyG_hr == 0:
        polyG_to_check = "G" * polyG_hr
        polyC_to_check = "C" * polyG_hr
        for i in range(len(data)):
            if polyG_to_check in LHR_list[i] or polyC_to_check in RHR_list[i]:
                index_to_remove.append(i)
        
    #polyT check
    if not polyT_hr == 0:
        polyT_to_check = "T" * polyT_hr
        polyA_to_check = "A" * polyT_hr
        for i in range(len(data)):
            if polyT_to_check in LHR_list[i] or polyA_to_check in RHR_list[i]:
                index_to_remove.append(i)
    
    gh_data = [i for j, i in enumerate(data) if j not in np.unique(index_to_remove)]
    
    return gh_data

def rs1_score(sequence): 
    """
    Adopted from Müller Paul, H., Istanto, D.D., Heldenbrand, J. et al. CROPSR: an automated platform for complex genome-wide CRISPR gRNA design and validation. BMC Bioinformatics 23, 74 (2022).
    """
    """
    Generates a binary matrix for DNA/RNA sequence, where each column is a possible base
    and each row is a position along the sequence. Matrix column order is A, T/U, C, G
    """
    seq = str(sequence).upper()
    seq = list(seq)
    matrix1  = np.zeros([len(sequence),4], dtype=int)
    for i,item in enumerate(sequence):
        if item == 'A':
            matrix1[i,0] = 1
        if item == 'T':
            matrix1[i,1] = 1
        if item == 'U':
            matrix1[i,1] = 1
        if item == 'C':
            matrix1[i,2] = 1
        if item == 'G':
            matrix1[i,3] = 1

    """
    Generates a binary matrix for DNA/RNA sequence, where each column is a possible
    pair of adjacent bases, and each row is a position along the sequence.
    Matrix column order is AA, AT, AC, AG, TA, TT, TC, TG, CA, CT, CC, CG, GA, GT, GC, GG
    """
    sequence = sequence.replace('U','T')
    pairwise_sequence = []
    for i in range(len(sequence)):
        if i < len(sequence)-1:
            basepair = sequence[i]+sequence[i+1]
            pairwise_sequence.append(basepair)
    matrix2 = np.zeros([len(pairwise_sequence),16], dtype=int)
    for i,item in enumerate(pairwise_sequence):
        if item == 'AA':
            matrix2[i,0] = 1
        if item == 'AT':
            matrix2[i,1] = 1
        if item == 'AC':
            matrix2[i,2] = 1
        if item == 'AG':
            matrix2[i,3] = 1
        if item == 'TA':
            matrix2[i,4] = 1
        if item == 'TT':
            matrix2[i,5] = 1
        if item == 'TC':
            matrix2[i,6] = 1
        if item == 'TG':
            matrix2[i,7] = 1
        if item == 'CA':
            matrix2[i,8] = 1
        if item == 'CT':
            matrix2[i,9] = 1
        if item == 'CC':
            matrix2[i,10] = 1
        if item == 'CG':
            matrix2[i,11] = 1
        if item == 'GA':
            matrix2[i,12] = 1
        if item == 'GT':
            matrix2[i,13] = 1
        if item == 'GC':
            matrix2[i,14] = 1
        if item == 'GG':
            matrix2[i,15] = 1

    """
    Scoring matrix
    """
    intersect = 0.59763615
    low_gc = -0.2026259
    high_gc = -0.1665878

    first_order = ['G02','A03','C03','C04','C05',
                    'G05','A06','C06','C07','G07',
                    'A12','A15','C15','A16','C16',
                    'T16','A17','G17','C18','G18',
                    'A19','C19','G20','T20','G21',
                    'T21','C22','T22','T23','C24',
                    'G24','T24','A25','C25','T25',
                    'G28','T28','C29','G30']
    first_scores = [-0.2753771,-0.3238875,0.17212887,-0.1006662,-0.2018029,
                    0.24595663,0.03644004,0.09837684,-0.7411813,-0.3932644,
                    -0.466099,0.08537695,-0.013814,0.27262051,0.1190226,
                    -0.2859442,0.09745459,-0.1755462,-0.3457955,-0.6780964,
                    0.22508903,-0.5077941,-0.4173736,-0.054307,0.37989937,
                    -0.0907126,0.05782332,-0.5305673,-0.8770074,-0.8762358,
                    0.27891626,-0.4031022,-0.0773007,0.28793562,-0.2216372,
                    -0.6890167,0.11787758,-0.1604453,0.38634258]
    first_order_scores = dict(zip(first_order,first_scores))

    second_order = ['GT02','GC05','AA06','TA06','GG07',
                    'GG12','TA12','TC12','TT12','GG13',
                    'GA14','GC14','TG17','GG19','TC19',
                    'CC20','TG20','AC21','CG21','GA21',
                    'GG21','TC22','CG23','CT23','AA24',
                    'AG24','AG25','CG25','TG25','GT27',
                    'GG29']
    second_scores = [-0.6257787,0.30004332,-0.8348362,0.76062777,-0.4908167,
                    -1.5169074,0.7092612,0.49629861,-0.5868739,-0.3345637,
                    0.76384993,-0.5370252,-0.7981461,-0.6668087,0.35318325,
                    0.74807209,-0.3672668,0.56820913,0.32907207,-0.8364568,
                    -0.7822076,-1.029693,0.85619782,-0.4632077,-0.5794924,
                    0.64907554,-0.0773007,0.28793562,-0.2216372,0.11787758,
                    -0.69774]
    second_order_scores = dict(zip(second_order,second_scores))

    # order 1 score matrix
    """ row order == A T/U C G """
    first_matrix = np.zeros([4,30], dtype=float)
    def posit(key):
        return int(key[1:])-1
    for k,v in first_order_scores.items():
        if k[0] == 'A':
            first_matrix[0,posit(k)] = v
        elif k[0] == 'T':
            first_matrix[1,posit(k)] = v
        elif k[0] == 'C':
            first_matrix[2,posit(k)] = v
        elif k[0] == 'G':
            first_matrix[3,posit(k)] = v

    # order 2 score matrix
    """ row order == AA AT AC AG TA TT TC TG CA CT CC CG GA GT GC GG """
    second_matrix = np.zeros([16,29], dtype=float)
    for k,v in second_order_scores.items():
        if k[0:2] == 'AA':
            second_matrix[0,int(k[2:])-1] = v
        if k[0:2] == 'AT':
            second_matrix[1,int(k[2:])-1] = v
        if k[0:2] == 'AC':
            second_matrix[2,int(k[2:])-1] = v
        if k[0:2] == 'AG':
            second_matrix[3,int(k[2:])-1] = v
        if k[0:2] == 'TA':
            second_matrix[4,int(k[2:])-1] = v
        if k[0:2] == 'TT':
            second_matrix[5,int(k[2:])-1] = v
        if k[0:2] == 'TC':
            second_matrix[6,int(k[2:])-1] = v
        if k[0:2] == 'TG':
            second_matrix[7,int(k[2:])-1] = v
        if k[0:2] == 'CA':
            second_matrix[8,int(k[2:])-1] = v
        if k[0:2] == 'CT':
            second_matrix[9,int(k[2:])-1] = v
        if k[0:2] == 'CC':
            second_matrix[10,int(k[2:])-1] = v
        if k[0:2] == 'CG':
            second_matrix[11,int(k[2:])-1] = v
        if k[0:2] == 'GA':
            second_matrix[12,int(k[2:])-1] = v
        if k[0:2] == 'GT':
            second_matrix[13,int(k[2:])-1] = v
        if k[0:2] == 'GC':
            second_matrix[14,int(k[2:])-1] = v
        if k[0:2] == 'GG':
            second_matrix[15,int(k[2:])-1] = v

    item_gc = sequence[0][5:-5]
    gc_count = item_gc.count('G') + item_gc.count('C')
    if gc_count < 10:
        gc_score = low_gc
    else:
        gc_score = high_gc
    first_first = np.matmul(first_matrix,matrix1)
    score_first = np.trace(first_first)
    score_second = np.trace(np.matmul(second_matrix,matrix2))
    score = (1/(1 + math.exp(-(intersect + gc_score + score_first + score_second))))
    return score
   
def one_hot_encoding(lines):
    data_n = len(lines) 
    SEQ = np.zeros((data_n, len(lines[0]), 4), dtype=int)
    
    for l in range(0, data_n):
        seq = lines[l]
        for i in range(28):
            if seq[i] in "Aa":
                SEQ[l, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l, i, 3] = 1

    return SEQ

def scores_guides_cas9(guides):

    seq = one_hot_encoding(guides)
    
    # model architecture
    SEQ = Input(shape=(28,4))
    conv_1 = Conv1D(activation="relu", padding="same", strides=1, filters=20, kernel_size=5, kernel_regularizer = l2(0.0001))(SEQ)
    bat_norm1 = BatchNormalization()(conv_1)
    pool = MaxPooling1D(pool_size=(2))(bat_norm1)
    conv_2 = Conv1D(activation="relu", padding="same", strides=1, filters=40, kernel_size=8, kernel_regularizer = l2(0.0001))(pool)
    bat_norm2 = BatchNormalization()(conv_2)
    pool_1 = AveragePooling1D(pool_size=(2))(bat_norm2)
    enc = pool_1
    dec_pool_1 =  UpSampling1D(size=2)(enc)
    dec_bat_norm2 = BatchNormalization()(dec_pool_1)
    dec_conv_2  = Conv1D(activation="relu", padding="same", strides=1, filters=40, kernel_size=8, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_bat_norm2)
    dec_pool = UpSampling1D(size=2)(dec_conv_2)
    dec_conv_1 = Conv1D(activation="relu", padding="same", strides=1, filters=20, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_pool)
    dec = Conv1D(activation="relu", padding="same", strides=1, filters=4, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_pool)
    model_seq = Model(inputs = SEQ, outputs= dec) 
    flatten = Flatten()(enc)
    dropout_1 = Dropout(0.5)(flatten)
    dense_1 = Dense(80, activation='relu', kernel_initializer='glorot_uniform')(dropout_1)
    dropout_2 = Dropout(0.5)(dense_1)
    out = Dense(units=1,  activation="linear")(dropout_2) 
    model = Model(inputs = SEQ, outputs= out)
    model.load_weights(sys.path[0] + "/model/cas9_seq_wtt.h5")
    pred_y = model.predict(seq)

    score = [-1*ele for ele in pred_y.flatten()]
    
    return score

def scores_guides_cas12a(guides):

    # one hot encoding
    seq = one_hot_encoding(guides)

    # model architecture
    SEQ = Input(shape=(32,4))
    conv_1 = Conv1D(activation="relu", padding="same", strides=1, filters=20, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(SEQ)
    bat_norm1 = BatchNormalization()(conv_1)
    pool = MaxPooling1D(pool_size=(2))(bat_norm1)
    conv_2 = Conv1D(activation="relu", padding="same", strides=1, filters=40, kernel_size=8, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(pool)
    bat_norm2 = BatchNormalization()(conv_2)
    pool_1 = AveragePooling1D(pool_size=(2))(bat_norm2)
    enc = pool_1
    dec_pool_1 =  UpSampling1D(size=2)(enc)
    dec_bat_norm2 = BatchNormalization()(dec_pool_1)
    dec_conv_2  = Conv1D(activation="relu", padding="same", strides=1, filters=40, kernel_size=8, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_bat_norm2)
    dec_pool = UpSampling1D(size=2)(dec_conv_2)
    dec_conv_1 = Conv1D(activation="relu", padding="same", strides=1, filters=20, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_pool)
    dec = Conv1D(activation="relu", padding="same", strides=1, filters=4, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_pool)
    model = Model(inputs = SEQ, outputs= dec)
    flatten = Flatten()(enc)
    dropout_1 = Dropout(0.5)(flatten)
    dense_1 = Dense(80, activation='relu', kernel_initializer='glorot_uniform')(dropout_1)
    dropout_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(units=40,  activation="relu",kernel_initializer='glorot_uniform')(dropout_2)
    dropout_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(units=40,  activation="relu",kernel_initializer='glorot_uniform')(dropout_3)
    out = Dense(units=1,  activation="linear")(dense_3)
    model3 = Model(inputs = [SEQ], outputs= out)
    model3.load_weights(sys.path[0] + "/model/cas12a_wtt.h5")

    # prediction
    test_x = seq
    pred_y = model3.predict([test_x])

    score = [-1*ele for ele in pred_y.flatten()]

    return score

def Vector_feature_to_Value_feature(position_feature_depedent_Dic,basepairLst):
    new_Dic={}
    for position in position_feature_depedent_Dic:
        for i,base in enumerate(basepairLst):
            new_Dic['%s_%s'%(position,base)]=position_feature_depedent_Dic[position][i]
    return new_Dic

def Order1(sequence):
    n=len(sequence)
    seq=sequence
    baseDic={}
    order=1
    baseLst=['A','T','C','G']
    position_independentDic={}
    position_dependentDic={}
    for i,base in enumerate(baseLst):
        baseDic[base]=np.zeros(4**order)
        baseDic[base][i]=1 
        position_independentDic['order1_IP_%s'%(base)]=0

    for i in range(len(seq)):
        for j,base in enumerate(baseDic):
            if base==seq[i]:
                position_dependentDic['order1_P%s'%(i+1)]=baseDic[base]
                position_independentDic['order1_IP_%s'%(base)]+=1

    position_dependentDic=Vector_feature_to_Value_feature(position_dependentDic,baseLst)
    Order1_positionDic=dict(list(position_dependentDic.items())+list(position_independentDic.items()))
    return Order1_positionDic

def Order2(sequence):
    seq=sequence
    BasepairDic={}
    BasepairLst=[]
    position_dependentDic={}
    position_independentDic={}
    order=2
    baseLst=['A','T','C','G']
    for base1 in baseLst:
        for base2 in baseLst:
            BasepairLst.append(base1+base2)
    for i in range(len(BasepairLst)):
        BasepairDic[BasepairLst[i]]=np.zeros(4**order)
        BasepairDic[BasepairLst[i]][i]=1
        position_independentDic['order2_IP_%s'%(BasepairLst[i])]=0
    for i in range(len(seq)-1):
        seq_pair=seq[i:i+2]
        for j,basepair in enumerate(BasepairLst):
            if seq_pair==basepair:
                position_dependentDic['order2_P%s'%(i+1)]=BasepairDic[basepair]
                position_independentDic['order2_IP_%s'%(basepair)]+=1

    position_dependentDic=Vector_feature_to_Value_feature(position_dependentDic,BasepairLst)
    Order2_positonDic=dict(list(position_dependentDic.items())+list(position_independentDic.items()))
    return Order2_positonDic

def NGGNfeature(sequence):
    NNDic={}
    seq=sequence
    BasepairDic={}
    BasepairLst=[]
    baseLst=['A','T','G','C']
    for base1 in baseLst:
        for base2 in baseLst:
            BasepairLst.append(base1+base2)
    for i in range(len(BasepairLst)):
        BasepairDic[BasepairLst[i]]=np.zeros(4**2)
        BasepairDic[BasepairLst[i]][i]=1
    for basepair in BasepairLst:
        if basepair == seq:
            NNDic['NGGN']=BasepairDic[basepair]
    NNDic=Vector_feature_to_Value_feature(NNDic,BasepairLst)
    return NNDic

def Temper(sequence):
    seq=sequence
    seq_7=seq[:7]
    seq_8=seq[7:15]
    seq_5=seq[15:20]
    TDic={}
    TDic['T20']=Tm.Tm_NN(seq)
    TDic['T7']=Tm.Tm_NN(seq_7)
    TDic['T8']=Tm.Tm_NN(seq_8)
    TDic['T5']=Tm.Tm_NN(seq_5)
    return TDic

def complement(sequence):
    seq=''
    for rec in sequence:
        if rec=='A':
            rec='T'
        elif rec=='T':
            rec='A'
        elif rec=='C':
            rec='G'
        else:
            rec='C'
        seq+=rec
    seq=seq[::-1]
    return seq

def feature(sequence,NGGN):
    Order1Position=Order1(sequence)
    Order2Position=Order2(sequence)
    Temprature=Temper(sequence)
    NGGN_sequence=NGGNfeature(NGGN)
    seq_feature=dict(list(Order1Position.items())+list(Order2Position.items())+list(Temprature.items())+list(NGGN_sequence.items()))
    seq_feature['GC']=float(GC(sequence))/100
    seq_feature_name=sorted(Order1Position.keys())+sorted(Order2Position.keys())+sorted(Temprature.keys())+sorted(NGGN_sequence.keys())
    seq_feature_name.append('GC')
    return seq_feature_name,seq_feature

def get_feature_main(sgRNA_seq_list):
    sgRNAfastaDic={}
    sgrnaFeatureDic={}
    for i in range(len(sgRNA_seq_list)):
        sgRNAfastaDic[i]=str(sgRNA_seq_list[i]).upper()
        sgrnaFeatureDic[i]={}
                
    featureName=[]
    for sgrna_name in sgRNAfastaDic:
        sgrnaSequence=sgRNAfastaDic[sgrna_name][4:24]
        NGGNSequence=sgRNAfastaDic[sgrna_name][24]+sgRNAfastaDic[sgrna_name][27]
        featureName,featureValue=feature(sgrnaSequence,NGGNSequence)
        sgrnaFeatureDic[sgrna_name]=featureValue
                
    my_data = pd.DataFrame(sgrnaFeatureDic).T 
    my_data = my_data.rename_axis('sgRNAID').reset_index()
    return my_data

def perform_normalization(activity):
    max_value=np.nanmax(activity)
    min_value=np.nanmin(activity)
    scaler=max_value-min_value
    return (activity-min_value)/scaler

def GBR_pred_main(sgRNA_seq_list, id_column, model, normalization):
    my_data = get_feature_main(sgRNA_seq_list) 
    gbm_tuned=pickle.load(open(model,'rb'))
    
    predictors = [x for x in my_data.columns if x!=id_column]

    activity=gbm_tuned.predict(my_data[predictors])
    if normalization:
        activity=perform_normalization(activity)
        
    result = pd.concat([my_data[id_column], pd.Series(activity, name='score')], axis=1)
    return list(result['score'])

def score_guides_bacteria(model_name, sgRNA_seq_list):
    varlist = {'model': model_name.split('(')[1].split(')')[0], 'normalization': 'Yes'}    
    id_column='sgRNAID'
    model='model/Cas9_sgRNA_activity_GBR.pickle' if varlist['model']=='Cas9' else 'model/eSpCas9_sgRNA_activity_GBR.pickle'
    output = GBR_pred_main(sgRNA_seq_list, id_column, model, varlist['normalization'])
    return output

def write_fasta(name, sequence_df):
    out_file = open(name, "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df['DEG_id'][i] + '\n')
        out_file.write(sequence_df['Sequence'][i] + '\n')
    out_file.close()

def ambg_nt_replacement(seq_list, ambiguous_nucleotides):
    for i in range(len(seq_list)):
        text = seq_list[i]
        if any(s in text for s in ambiguous_nucleotides):
            text = text.replace('M', 'A')
            text = text.replace('R', 'A')
            text = text.replace('W', 'A')
            text = text.replace('S', 'C')
            text = text.replace('Y', 'C')
            text = text.replace('K', 'G')
            text = text.replace('V', 'A')
            text = text.replace('H', 'A')
            text = text.replace('D', 'A')
            text = text.replace('B', 'C')
            text = text.replace('N', 'A')
        
            seq_list[i] = text
        
    return seq_list

#IUPAC Code
iupac_code = {
  "A": "A",
  "T": "T",
  "G": "G",
  "C": "C",
  "M": "A|C",
  "R": "A|G",
  "W": "A|T",
  "S": "C|G",
  "Y": "C|T",
  "K": "G|T",
  "V": "A|C|G",
  "H": "A|C|T",
  "D": "A|G|T",
  "B": "C|G|T",
  "N": "A|C|G|T",
}

def main():
    genome_file = args.Genome
    gene_file = args.Gene_table
    output_file = args.Output_file
    pam = args.PAM
    orient = args.Orientation
    glen = args.Guide_Length
    seedlen = args.Seed_Length
    re_grna_list = args.RE_grna
    gc_limits = args.GC_grna
    polyG_len = args.polyG_grna
    polyT_len = args.polyT_grna
    intergenic_space = args.intspace
    edit_dist = args.edit_dist
    dist_type = args.dist_type
    gdenslen = args.gene_density_len
    hr_len = args.HR_Length
    re_hr_list = args.RE_hr
    polyG_hr = args.polyG_hr
    polyT_hr = args.polyT_hr
    protein_file = args.protein_file
    org_ge = args.blast_org
    backbone_region = args.backbone_complementarity_check
    distal_end = args.distal_end_len
    on_target_score_name = args.on_target

    #Data Processing
    genome = read_fasta(genome_file)
    total_gene_table = pd.read_csv(gene_file, sep = '\t')
    gene_table = total_gene_table[total_gene_table['assembly_unit'] == 'Primary Assembly'][['# feature','class','chromosome','genomic_accession','start','end','strand','locus_tag','product_accession']].reset_index(drop=True)
    gene_table.columns = ['# feature','class','#Name','Accession', 'Start', 'Stop', 'Strand', 'Locus tag','Protein product']
    refined_gene_table = gene_table[gene_table['# feature']=='gene'][['Accession', 'Start', 'Stop', 'Strand', 'Locus tag']].reset_index(drop=True)
    pam_library = pam_to_search(pam,iupac_code)
    ambiguous_nucleotides = list(iupac_code.keys())[4:]
    if " " in re_grna_list:
        re_grna_list = re_grna_list.replace(" ", "")
    if " " in re_hr_list:
        re_hr_list = re_hr_list.replace(" ", "")
    if " " in backbone_region:
        backbone_region = backbone_region.replace(" ", "")

    if seedlen >= glen:
        print('Seed length should be less than the guide length.')
        sys.exit()

    low_limit, up_limit = [int(x) for x in gc_limits.split(",")]
    if low_limit < 0 or up_limit > 100:
        print('GC range is not valid. Please enter the value between 0 and 100. The input format is as follows: --GC_grna 10,90')
        sys.exit()

    if pam == 'NGG' and orient == '3prime':
        if distal_end < hr_len:
            print('Please enter a value greater than the length of the homology arms for the distal end length.')
            sys.exit()

    #Obtaining harbors
    grna_list = grna_search(genome, pam_library, glen, orient)
    grna_data = grna_filter(grna_list, glen, pam, orient, seedlen, re_grna_list, polyG_len, polyT_len, edit_dist, refined_gene_table, intergenic_space, gdenslen, ambiguous_nucleotides, gc_limits, dist_type)
    del grna_list

    if len(grna_data) > 0:
        grna_hr_data = hr_filter(grna_data, glen, pam, genome, hr_len, re_hr_list, polyG_hr, polyT_hr)
        del grna_data

        #Cleaning and Labeling dataframe
        grna_hr_df = pd.DataFrame(grna_hr_data, columns = ['Guide with PAM', 'Accession', 'Location', 'Strand', 'Chromosome Length', 'Distance, Closest off-target', 'Intergenic Size', 'Left Gene', 'Right Gene', 'Relative Orientation', 'Gene Density', 'Left HR', 'Right HR'])
    else:
        print('No harbors can be obtained after applying the specified constraints. Try relaxing the edit distance criteria.')
        sys.exit()
   
    if len(grna_hr_data) > 0:
        del grna_hr_data
        
        if orient == '3prime':
            guide_seq = grna_hr_df['Guide with PAM'].str[:glen]
            pam_seq = grna_hr_df['Guide with PAM'].str[glen:]
        elif orient == '5prime':
            guide_seq = grna_hr_df['Guide with PAM'].str[len(pam):]
            pam_seq = grna_hr_df['Guide with PAM'].str[:len(pam)]

        grna_hr_df.insert(loc=0, column='PAM', value=pam_seq)
        grna_hr_df.insert(loc=0, column='Guide Sequence', value=guide_seq)
        del grna_hr_df['Guide with PAM']
        
        chrom_name_df = gene_table.drop_duplicates('Accession').reset_index(drop=True)[['#Name','Accession']]
        grna_hr_df = grna_hr_df[grna_hr_df['Accession'].isin(list(chrom_name_df['Accession']))].reset_index(drop=True) #removing gRNA if accession ID not in gene table as intergenic criteria cannot be checked

        chrom_name_list = []
        for i in range(len(grna_hr_df)):
            chrom_name_list.append(chrom_name_df.loc[chrom_name_df['Accession'] == grna_hr_df['Accession'][i], '#Name'].iloc[0])

        grna_hr_df.insert(loc=3, column='Chromosome', value=chrom_name_list)

        self_comp = []
        stem_len = 4
        for i in range(len(grna_hr_df)):
            fwd = grna_hr_df['Guide Sequence'][i]
            rvs = str(Seq(fwd).reverse_complement())
            L = len(fwd)-stem_len-1

            folding = 0
            for j in range(0,len(fwd)-stem_len):
                if GC(fwd[j:j+stem_len]) >= 0.5:
                    if fwd[j:j+stem_len] in rvs[0:(L-j)] or any([fwd[j:j+stem_len] in item for item in backbone_region]):
                        folding += 1

            self_comp.append(folding)

        grna_hr_df.insert(loc = 3, column='Self-Complementarity', value = self_comp)
        grna_hr_df.insert(loc = 3, column='GC Content', value=[GC(w) for w in list(grna_hr_df['Guide Sequence'])])

        #Remove gRNA located at the end of the chromosomes
        ind_to_remove = []
        for i in range(len(grna_hr_df)):
            if grna_hr_df['Location'][i] < distal_end or grna_hr_df['Location'][i] > grna_hr_df['Chromosome Length'][i] - distal_end:
                ind_to_remove.append(i)

        grna_hr_df = grna_hr_df.drop(ind_to_remove).reset_index(drop=True)

        #Adding Sequence ID to track gRNA in visualization (as Bokeh does not allow copying)
        grna_hr_df.insert(0, 'ID', range(1, 1 + len(grna_hr_df)))

        #On target scores
        if len(grna_hr_df) > 0:
            if on_target_score_name == 'Doench et al. 2016':
                if pam == 'NGG' and orient == '3prime':
                    on_target_seq = []
                    for i in range(len(grna_hr_df)):
                        if glen < 24:
                            on_target_seq.append(grna_hr_df['Left HR'][i][glen-24:] + grna_hr_df['Guide Sequence'][i] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                        else:
                            on_target_seq.append(grna_hr_df['Guide Sequence'][i][-24:] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])

                    grna_hr_df['On-target Score'] = doench_predict.predict(np.array(ambg_nt_replacement(on_target_seq, ambiguous_nucleotides)), num_threads=1)
                    
                else:
                    grna_hr_df['On-target Score'] = 'NA'
                
            elif on_target_score_name == 'CROPSR':
                if pam == 'NGG' and orient == '3prime':
                    on_target_seq = []
                    for i in range(len(grna_hr_df)):
                        if glen < 24:
                            on_target_seq.append(grna_hr_df['Left HR'][i][glen-24:] + grna_hr_df['Guide Sequence'][i] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                        else:
                            on_target_seq.append(grna_hr_df['Guide Sequence'][i][-24:] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                    
                    grna_hr_df['On-target Score'] = np.vectorize(rs1_score)(ambg_nt_replacement(on_target_seq, ambiguous_nucleotides))
                    
                else:
                    grna_hr_df['On-target Score'] = 'NA'
                    
            elif on_target_score_name == 'DeepGuide (Cas9)':
                if pam == 'NGG' and orient == '3prime':
                    on_target_seq = []
                    for i in range(len(grna_hr_df)):
                        if glen < 22:
                            on_target_seq.append(grna_hr_df['Left HR'][i][glen-22:] + grna_hr_df['Guide Sequence'][i] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                        else:
                            on_target_seq.append(grna_hr_df['Guide Sequence'][i][-22:] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                            
                    grna_hr_df['On-target Score'] = scores_guides_cas9(ambg_nt_replacement(on_target_seq, ambiguous_nucleotides))
                    
                else:
                    grna_hr_df['On-target Score'] = 'NA'
                
            elif on_target_score_name == 'DeepGuide (Cas12a)':
                if pam == 'TTTV' and orient == '5prime':
                    on_target_seq = []
                    for i in range(len(grna_hr_df)):
                        if glen < 27:
                            on_target_seq.append(grna_hr_df['Left HR'][i][-1:] + grna_hr_df['PAM'][i] + grna_hr_df['Guide Sequence'][i] + grna_hr_df['Right HR'][i][0:27-glen])
                        else:
                            on_target_seq.append(grna_hr_df['Left HR'][i][-1:] + grna_hr_df['PAM'][i] + grna_hr_df['Guide Sequence'][i][0:27])

                    grna_hr_df['On-target Score'] = scores_guides_cas12a(ambg_nt_replacement(on_target_seq, ambiguous_nucleotides))

                else:
                    grna_hr_df['On-target Score'] = 'NA'
					
            elif on_target_score_name == 'sgRNA_ecoli (Cas9)':
                if pam == 'NGG' and orient == '3prime':
                    on_target_seq = []
                    for i in range(len(grna_hr_df)):
                        if glen < 24:
                            on_target_seq.append(grna_hr_df['Left HR'][i][glen-24:] + grna_hr_df['Guide Sequence'][i] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                        else:
                            on_target_seq.append(grna_hr_df['Guide Sequence'][i][-24:] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                    
                    #sgRNA ecoli scoring
                    grna_hr_df['On-target Score'] = score_guides_bacteria(on_target_score_name, ambg_nt_replacement(on_target_seq, ambiguous_nucleotides))
                    
            elif on_target_score_name == 'sgRNA_ecoli (eSpCas9)':
                if pam == 'NGG' and orient == '3prime':
                    on_target_seq = []
                    for i in range(len(grna_hr_df)):
                        if glen < 24:
                            on_target_seq.append(grna_hr_df['Left HR'][i][glen-24:] + grna_hr_df['Guide Sequence'][i] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                        else:
                            on_target_seq.append(grna_hr_df['Guide Sequence'][i][-24:] + grna_hr_df['PAM'][i] + grna_hr_df['Right HR'][i][0:3])
                    
                    #sgRNA ecoli scoring
                    grna_hr_df['On-target Score'] = score_guides_bacteria(on_target_score_name, ambg_nt_replacement(on_target_seq, ambiguous_nucleotides))
            
            else:
                grna_hr_df['On-target Score'] = 'NA'
			
        if org_ge and protein_file:
            #Adding essentiality information
            organism_list = org_ge.split(',')

            #procuring Essential Gene Database file
            deg_database = pd.read_csv(deg_file)

            db = os.path.join(temporary, 'RefOrg.faa') # BLAST database
            blastout = os.path.join(temporary, 'blast.tab')  # BLAST output

            eg_df = deg_database[deg_database.Organism.isin(organism_list)].reset_index(drop=True).iloc[:,0:2]

            #create RefOrg file 
            ref_org = os.path.join(temporary, 'RefOrg.fasta') 
            write_fasta(ref_org, eg_df)

            blast_start = time.perf_counter()
            #Creating Blast Database
            blastdb_cmd = '{}makeblastdb -in {} -parse_seqids -dbtype prot -out {}'.format(blast_path, ref_org, db)
            os.system(blastdb_cmd)

            #Blast
            cmd_blastp = NcbiblastpCommandline(cmd = blast_path + 'blastp', query = protein_file, out = blastout, outfmt = 6, db = db,  num_threads=NUM_THREADS)
            stdout, stderr = cmd_blastp()
            blast_stop = time.perf_counter()
            global blast_time
            blast_time = blast_stop - blast_start

            results = pd.read_csv(blastout, sep="\t", header=None)
            headers = ['query', 'subject',
                        'pc_identity', 'aln_length', 'mismatches', 'gaps_opened',
                        'query_start', 'query_end', 'subject_start', 'subject_end',
                        'e_value', 'bitscore']

            results.columns = headers
            #Change BLAST parameters here (Update it to 50% if organism of interest is phylogenetically close)
            blast_gene_table = gene_table[(gene_table['# feature']=='CDS') & (gene_table['class']=='with_protein')].reset_index(drop=True)
            results_filtered = results.loc[(results['e_value'] < 1e-5) & (results['pc_identity'] >= 50)].reset_index(drop=True)
            eg_loc_df = blast_gene_table[blast_gene_table['Protein product'].isin(np.unique(list(results_filtered['query'])))].reset_index(drop=True)

            chrom_len_array = []
            for i in range(len(chrom_name_df)):
                for j in range(len(genome)):
                    if chrom_name_df['Accession'][i] == genome[j][0]:
                        chrom_len_array.append(len(genome[j][1]))

            chrom_name_df['Length'] = chrom_len_array

            chr_eg_zone = []
            for i in range(len(chrom_name_df)):
                curr_chr_eg_data = eg_loc_df.loc[eg_loc_df['Accession'] == chrom_name_df['Accession'][i]].sort_values('Start').reset_index(drop=True)

                zone_info = []
                ini_flag = 0
                if np.shape(curr_chr_eg_data)[0] > 0:
                    for j in range(len(curr_chr_eg_data)):
                        if ini_flag == 0:
                            zone_info = str(0) + '-' + str(curr_chr_eg_data['Start'][j])
                            chr_eg_zone.append([chrom_name_df['Accession'][i], zone_info])
                            ini_flag = 1

                        if j == len(curr_chr_eg_data) - 1:
                            zone_info = str(curr_chr_eg_data['Stop'][j]) + '-' + str(chrom_name_df['Length'][i])
                        else:
                            zone_info = str(curr_chr_eg_data['Stop'][j]) + '-' + str(curr_chr_eg_data['Start'][j+1])

                        chr_eg_zone.append([chrom_name_df['Accession'][i], zone_info])
                else:
                    zone_info = str(0) + '-' + str(chrom_name_df['Length'][i]) #no essential genes on that chromosome
                    chr_eg_zone.append([chrom_name_df['Accession'][i], zone_info])

            chr_eg_zone = pd.DataFrame(chr_eg_zone, columns = ['Acc','Loc'])

            zone = []
            site_loc = []
            for i in range(len(grna_hr_df)):
                if grna_hr_df['Strand'][i] == '+':
                    grna_loc = grna_hr_df['Location'][i]
                    site_loc.append(grna_loc)
                else:
                    grna_loc = grna_hr_df['Chromosome Length'][i] - grna_hr_df['Location'][i]
                    site_loc.append(grna_loc - glen - len(pam))

                for j in range(np.shape(chr_eg_zone)[0]):
                    if chr_eg_zone['Acc'][j] == grna_hr_df['Accession'][i]:
                        curr_zone_bound = chr_eg_zone['Loc'][j].split('-')
                        if grna_loc > int(curr_zone_bound[0]) and grna_loc < int(curr_zone_bound[1]):
                                zone.append(j+1)

            grna_hr_es_df = grna_hr_df
            grna_hr_es_df['Zone'] = zone
            del grna_hr_es_df['Location']
            grna_hr_es_df.insert(loc = 6, column='Location', value = site_loc)

            pd.DataFrame(grna_hr_es_df).to_csv(output_file, index = False) #Safe Harbor Data Output

        else:
            site_loc = []
            for i in range(len(grna_hr_df)):
                if grna_hr_df['Strand'][i] == '+':
                    site_loc.append(grna_hr_df['Location'][i])
                else:
                    site_loc.append(grna_hr_df['Chromosome Length'][i] - grna_hr_df['Location'][i] - glen - len(pam))

            del grna_hr_df['Location']
            grna_hr_df.insert(loc = 6, column='Location', value = site_loc)

            pd.DataFrame(grna_hr_df).to_csv(output_file, index = False) #Harbor Data Output

    else:
        print('No Safe Harbors can be obtained after applying the specified constraints. Try relaxing the criteria.')

    print("=================================")
    print("scann time, blast time")
    print("{},{}".format(scann_time, blast_time))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--Genome', help="Genome filename", required=True)
    parser.add_argument('-t', '--Gene_table', help="Gene table filename", required=True)
    parser.add_argument('-out', '--Output_file', default = 'output.csv', help="Name of the output file")
    parser.add_argument('-p', '--PAM', type=str, default='NGG', help="A short PAM motif to search for, it may use IUPAC ambiguous alphabet. Default: NGG.", required=True)
    parser.add_argument('-o', '--Orientation', choices=['3prime', '5prime'], default='3prime', help="PAM position relative to target: 5prime: [PAM][target], 3prime: [target][PAM]. For example, PAM orientation for SpCas9 is 3prime. Default: 3prime.")
    parser.add_argument('-l', '--Guide_Length', type=int, choices=range(10, 28, 1), metavar="[10-27]", default=20, help="Length of the guide sequence. Default: 20.")
    parser.add_argument('-sl','--Seed_Length', type=int, choices=range(0, 27, 1), metavar="[0-27]", default=10, help='Length of a seed region near the PAM site required to be unique. Specified length should be less than the guide length. Default: 10.')
    parser.add_argument('--RE_grna', type=str, default='', help='Undesired recognition sequence of restriction enzymes in guide RNA')
    parser.add_argument('--GC_grna', type=str, default='0,100', help="GC content limits of the gRNA. Recommended range: '25,75'.")
    parser.add_argument('--polyG_grna', type=int, choices=range(0, 11, 1), metavar="[0-10]", default=0, help='Length of consecutive G/C repeats not allowed in the guide sequence. Default value of 0 implies poly_G rule is not applied.')
    parser.add_argument('--polyT_grna', type=int, choices=range(0, 11, 1), metavar="[0-10]", default=0, help='Length of consecutive T/A repeats not allowed in the guide sequence. Default value of 0 implies poly_T rule is not applied.')
    parser.add_argument('--intspace', type=int, default=300, help='Minimum distance of gRNA from any gene. Default is 300bp. Value is dependent on the organism of interest. Example: Prokaryotes: 300 bp, Fungi: 400 bp.')
    parser.add_argument('--edit_dist', type=int, default=6, choices=range(0, 11, 1), metavar="[0-10]",  help='Minimum number of mismatches allowed in the guide region to classify a sequence as candidate gRNA. Default value is 6.')
    parser.add_argument('--dist_type', choices=['hamming', 'levenshtein'], default='hamming', help="Select the distance type. Default: hamming.")
    parser.add_argument('-gd_l', '--gene_density_len', type=int, default=10000, help='Size of the region from the gRNA site to calculate gene density. Default is 10000bp. Value is dependent on the organism of interest.')
    parser.add_argument('-hr_l', '--HR_Length', type=int, choices=range(5, 1001, 1), metavar="[5-1000]", default=50, help="Length of the homology arms. Default: 50bp.")
    parser.add_argument('--RE_hr', type=str, default='', help='Undesired recognition sequence of restriction enzymes in the homology arm.')
    parser.add_argument('--polyG_hr', type=int, choices=range(0, 11, 1), metavar="[0-10]", default=0, help='Length of consecutive G/C repeats not allowed in the homology arm. Default value of 0 implies poly_G rule is not applied.')
    parser.add_argument('--polyT_hr', type=int, choices=range(0, 11, 1), metavar="[0-10]", default=0, help='Length of consecutive T/A repeats not allowed in the homology arm. Default value of 0 implies poly_T rule is not applied.')
    parser.add_argument('--backbone_complementarity_check', type=str, default='', help='Complementarity check if the guide RNA will form secondary structure with the backbone.')
    parser.add_argument('--protein_file', type=str, default='', help="Fasta file containing protein sequences.")
    parser.add_argument('--blast_org', type=str, default='',  help="Name of the oprganism/s to blast proteins against to identify probable essential genes.")
    parser.add_argument('--distal_end_len', type=int, default=5000,  help="Remove guide RNA located within this distance from the end of the chromosome. Value is dependent on the organism of interest. Note for NGG PAM, enter a value greater than the length of the homology arms.")
    parser.add_argument('--on_target', type=str, default='Doench et al. 2016', help="Model to calculate on-target scores. Options: Doench et al. 2016, CROPSR, DeepGuide (Cas9), DeepGuide (Cas12a), sgRNA_ecoli (Cas9), sgRNA_ecoli (eSpCas9).")
    args = parser.parse_args()
    main()
