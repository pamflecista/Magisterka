### Using convolution neural networks (CNN) for better understanding of human genome

<br></br>

**This repository is a part of my master's project called "Identification of 
chromatin regions active in human brain using neural networks".**

<br></br>

### Versions

- Python 3.6
- pandas 1.0.1
- numpy 1.18.1
- pytorch 1.3.1

<br></br>

### Example dataset

Example of trained network is in */data/custom40* directory.

**Calculating integrated gradients**
<br></br>
To calculate integrads based on example model and set of sequences just run:

```
python3 calculate_integrads.py \
        --model custom40_last.model \
        --seq extreme_custom40_train_1.fasta \
        --baseline CHOSEN-BASELINE
```
CHOSEN-BASELINE depends on what baseline you want to use for calculating 
integrated gradients (see: https://arxiv.org/abs/1703.01365 for details 
of the method), select one of the options:
- *zeros* - use zeros array as baseline
- *fixed* - use the same set of random sequences as the baseline for each 
sequence
- *random* - use different random set of sequences as the baseline for each 
sequence
- *test-balanced-8-3_baseline.npy* - use pre-calculated balanced baseline 
(each 3 nucleotides in the given position occur exactly once across all 64. baseline sequences)

As the output new directory called *integrads_NETWORK_SEQUENCES_BASELINE_TRIALS-STEPS* is created
(*integrads_custom40_extreme-custom40-train-1_CHOSEN-BASELINE_10-50* if the default data was used). 
Inside there are result files:
- integrads_all.npy - numpy array with calculated gradients
- params.txt - file with parameters of the analysis

**Plotting seqlogos**
<br></br>
To plot seqlogos based on the calculated integrads run:
```
python3 plot_seqlogo.py \
integrads_custom40_extreme-custom40-train-1_CHOSEN-BASELINE_10-50/integrads_all.npy \
--global_ylim \
--one \
--clip NUM
```
Options *global_ylim*, *one* and *clip* are optional:
- *global_ylim* - 
set the same y axis range for all sequences from the given array
- *one* - plot only one nucleotide in one position 
(the one from the original sequence)
- *clip* - subset of nucleotides to plot: +-NUM from the middle 
of the sequences, by default NUM=100

As the output new directory with plots is created.


