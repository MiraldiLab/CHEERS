'''
:File: CHEERS_normalize.py
:Author: Blagoje Soskic, Wellcome Sanger Institute, <bs11@sanger.ac.uk>
        Updated: Tareian Cazares and Faiz Rizvi, University of Cincinnati
:Last updated: 10/06/2021

This script is used to normalize counts within peaks in the dataset:
1) load output of featureCounts *_ReadsInPeaks.txt (each txt file is an analyzed sample that contains 4 columns without header: chr   start   end count )
2) correct counts per peak for library size by scaling it to the sample with the largest sum of counts
3) remove the bottom 10th percentile of peaks with the lowest read counts
4) quantile normalize the library size-corrected peak counts
5) Euclidean normalization to obtain a cell type specificity score

outputs :
outputPrefix_normToMax.txt
outputPrefix_normToMax_quantileNorm.txt
outputPrefix_normToMax_quantileNorm_euclideanNorm.txt

Usage:
    python CHEERS_normalize.py --input ~/peak/counts/per/sample/ --prefix prefix --outdir ~/output/directory/
'''

import os
import glob
import numpy as np
import pandas as pd
import argparse
import logging

# logging level set to INFO
logging.basicConfig(format='%(asctime)s | %(message)s',
                    level=logging.INFO)

LOG = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description = "Data normalization for CHEERS disease enrichment",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input", help = "path to files with read counts per peak")
parser.add_argument("--prefix", help = "file prefix")
parser.add_argument("--outdir", help = "directory where to output results")

args = parser.parse_args()

LOG.info('CHEERS: Adapted for use with maxATAC predictions')


def filter_matrix(counts_matrix):
    # Get the sum of the counts per peak across all samples
    sum_peak_counts = counts_matrix.sum(axis=1)

    # Find the bottom 10th percentile value
    bottom = np.percentile(sum_peak_counts, 10)

    # Get only the peaks with a sum across of peak counts across all samples greater than bottom 10th percentile
    counts_matrix = counts_matrix.iloc[np.where(sum_peak_counts > bottom)]

    # Reset index to build the bed formated file
    return counts_matrix.reset_index()


def count_depth_normalize(counts_matrix):
    # Find the total number of reads
    filesSums = counts_matrix.sum() #create list of names

    # create norm factor
    # Get the max read depth across all samples
    maxSum = max(filesSums)

    # Create a vector of normalized values based on the max read count / each samples total read count
    # This will create a vector where the values are a proportion of the max sequencing depth
    norm_value_vector = [maxSum / x for x in filesSums]

    # Normalize each column by the normalization factor specific to its library size compared to the max size
    return counts_matrix.divide(norm_value_vector)


# Define functions
def quantile_norm(normData):
    regions = normData.loc[:, :'end']
    normDataClean = normData.drop(['chr', 'start', 'end'], axis=1)
    rank_mean = normDataClean.stack().groupby(normDataClean.rank(method='first').stack().astype(int)).mean()
    normDataCleanQuantile = normDataClean.rank(method='min').stack().astype(int).map(rank_mean).unstack()
    normDataCleanQuantileFinal = pd.concat([regions, normDataCleanQuantile], axis=1)
    return normDataCleanQuantileFinal


def euclidean_norm(normDataCleanQuantileFinal):
    normDataQuantile = normDataCleanQuantileFinal.drop(['chr', 'start', 'end'], axis=1)
    normDataSquare = np.square(normDataQuantile)
    normDataQuantile['eucl_norm'] = np.sqrt(normDataSquare.sum(axis=1))
    normDataQuantileEucl = normDataQuantile.iloc[:, 0:].div(normDataQuantile.eucl_norm, axis=0)
    dim = normDataQuantileEucl.shape[1] - 1
    final = normDataQuantileEucl.iloc[:, :dim]
    return final


def import_counts_directory(input_files):
    """Import a directory of BEDgraph files as a matrix

    This function will for loop through all .bed files found by glob and import them into a single counts matrix.
    The input files must have been generated using the same reference peak set or there will be missing and overlapping
    intervals in the output matrix.

    args:
    input_files (list): A list of input .bed files to merge into a matrix

    return:
    counts_matrix (dataframe): A matrix where rows are peaks and columns are file specific counts per peak
    """
    data_list = []

    # open and load all counts per files.
    for filename in input_files:
        base_filename = os.path.basename(filename)

        # Import bed file as dataframe. Rename counts as filename
        df = pd.read_table(filename, names=["chr", "start", "end", base_filename])

        # Filter for specific chromosomes
        df = df[df["chr"].isin(chrList)]

        # Set the index to be unique peaks
        df = df.set_index(["chr", "start", "end"])

        # Add dataframe to a list to be concatenated
        data_list.append(df)

    return pd.concat(data_list, axis=1)

# Body

# Get the number of input files detected
input_files = glob.glob(os.path.join(args.input, '*.bed'))

LOG.info(f'{len(input_files)} filed ending in .bed detected')

# Chrs to keep for the analysis
chrList = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13',
           'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

LOG.info('Limiting analysis to the following chromosomes' + "\n" + "\n".join(chrList))

'''
Import counts files as a matrix
'''
LOG.info(f'Importing counts files as a matrix')

counts_matrix = import_counts_directory(input_files)

counts_matrix.to_csv(os.path.join(args.outdir, args.prefix + "_counts.tsv"), sep="\t", index=False)

'''
Normalize counts to account for sequencing depth.
'''
LOG.info('Normalizing counts matrix to account for sequencing depth')

# Normalize each column by the normalization factor specific to its library size compared to the max size
counts_matrix = count_depth_normalize(counts_matrix)

'''
remove bottom 10th percentile of the peaks
'''
LOG.info('Filtering counts matrix to remove bottom 10th percentile of peaks by sum of counts')

# Reset index to build the bed formated file
counts_matrix = filter_matrix(counts_matrix)

LOG.info('Writing normalized counts matrix to file')

# Write normalized matrix to file
counts_matrix.to_csv(os.path.join(args.outdir, args.prefix + "_counts_normToMax.tsv"), sep="\t", index=False)

'''
quantile normalize the library size-corrected peak counts to make comparisons between different peaks possible
'''
# quantile normalisation and output
LOG.info('Quantile normalizing filtered counts matrix')

normDataQuantile = quantile_norm(counts_matrix)

fileName = os.path.join(args.outdir, args.prefix + "_counts_normToMax_quantileNorm.tsv")

LOG.info('Writing quantile normalized and filtered counts matrix to file')

normDataQuantile.to_csv(fileName, header=1, index=None, sep='\t')

'''
To assess specificity score perform euclidean normalization
'''
# Euclidean normalization and output
LOG.info('Euclidean normalize the quantile normalized and filtered counts matrix')

euclideanNorm = euclidean_norm(normDataQuantile)
regions = normDataQuantile.loc[:, :'end']
euclideanNorm = euclideanNorm.round(4)
euclideanNormFinal = pd.concat([regions, euclideanNorm], axis=1)

fileName3 = os.path.join(args.outdir, args.prefix + '_counts_normToMax_quantileNorm_euclideanNorm.tsv')

LOG.info('Write the euclidean + quantile normalized + filtered counts matrix to file')

euclideanNormFinal.to_csv(fileName3, header=1, index=None, sep='\t')

LOG.info('Normalization Complete!')
