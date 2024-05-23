# SERcaf
<br />
Speech emotion recognition (SER) aims to recognize human emotions through in-depth analysis of audio signals. However, it remains challenging to encode emotional cues and to fuse the encoded cues effectively. 
<br />
In this study, dual-stream representation is developed, and both full training and fine tuning of different deep networks are employed for encoding emotion patterns. 
Specifically, a cross-attention fusion (CAF) module is designed to integrate the dual-stream output for emotion recognition. Using different dual-stream encoders 
(full-training a text processing network and fine-tuning a pre-trained large language network), the CAF module is compared to other three fusion modules on three databases. 
The SER performance is quantified with weighted accuracy (WA), unweighted accuracy (UA), and F1-score (F1S). 
<br />
Experimental results suggest that the CAF outperforms the other three modules and leads to promising performance on the databases. 
It is also found that fine-tuning a pre-trained large language network achieves superior representation than full-training a text processing network. 
<br />
In a future study, improved SER performance could be achieved through the development of a multi-stream representation of emotional cues 
and the incorporation of a multi-branch fusion mechanism for emotion recognition.

**Our contribution**
<br />
(1) Dual-stream representation is designed. It fully trains a text processing network and fine-tunes a language processing network for encoding emotional cues in audio.
<br />
(2) A novel feature fusion module CAF is developed. It enables feature dimensionality alignment and generates a more informative representation for emotion recognition. 
<br />
(3) The proposed SER framework is validated on three databases and achieves promising performance.

**Citation**
<br />
The work is under review. If it is helpful, please cite
<br />
Yu S, Meng J, Fan W, Chen Y, Zhu B, Yu H, Xie Y, Sun Q. Speech emotion recognition using dual-stream representation and cross-attention fusion. 2024.
