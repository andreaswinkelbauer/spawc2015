# On Quantization of Log-Likelihood Ratios for Maximum Mutual Information

This code package corresponds to the following scientific paper:

Andreas Winkelbauer and Gerald Matz, “On quantization of log-likelihood ratios for maximum mutual information,” in Proc. 16th IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC 2015), Stockholm, Sweden, June 2015.

We provide a MATLAB implementation of the algorithms described in the above paper. The use of these algorithms is exemplified by MATLAB scripts that generate the numerical results presented in this paper.

## Content of Code Package

This paper describes two algorithms which are implemented as MATLAB functions in the files ``design_LLR_quant_distribution.m`` (quantizer design based on LLR distribution) and ``design_LLR_quant_samples.m`` (quantizer design based on LLR samples). These files contain an extensive inline documentation.

The files ``figure2.m`` and ``figure3.m`` use the functions mentioned above and produce the data shown in Figures 2 and 3 in our paper. To this end, the files ``ib_limit.m`` and ``quant_mse.m`` provide helper functions which are tailored to this specific use case and should not be used for other purposes.

This code package also includes a ``LICENSE`` file and this ``README.md`` file.

## Paper Abstract

We consider mutual-information-optimal quantization of log-likelihood ratios (LLRs). An efficient algorithm is presented for the design of LLR quantizers based either on the unconditional LLR distribution or on LLR samples. In the latter case, a small number of samples is sufficient and no training data are required. Therefore, our algorithm can be used to design LLR quantizers during data transmission. The proposed algorithm is reminiscent of the famous Lloyd-Max algorithm and is not restricted to any particular LLR distribution.

## Acknowledgements

This research has received funding from the WWTF project TINCOIN (ICT12-054) and from the EU FP7 project NEWCOM# (GA FP7-ICT-318306).

## License and Referencing

This code package is licensed under the GPLv2 license. If you use this code in any way (for research or otherwise), please cite our original paper as listed above. If you are a BibTeX user, you may want to use the following code:

```
@InProceedings{winkelbauer2015a,
  Title                    = {On Quantization of Log-Likelihood Ratios for Maximum Mutual Information},
  Author                   = {Winkelbauer, Andreas and Matz, Gerald},
  Booktitle                = {Proc. 16th IEEE Int. Workshop on Signal Processing Advances in Wireless Communications (SPAWC 2015)},
  Year                     = {2015},
  Month                    = {June}
}
```
