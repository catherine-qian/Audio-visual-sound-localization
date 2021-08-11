# Audio-visual-sound-localization

source code for the ICASSP2021 paper：“Multi-target DoA Estimation with an Audio-visual Fusion Mechanism”

If you use this code, please cite:

@inproceedings{qian2021multi,
  title={Multi-target DoA Estimation with an Audio-visual Fusion Mechanism},
  author={Qian, Xinyuan and Madhavi, Maulik and Pan, Zexu and Wang, Jiadong and Li, Haizhou},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4280--4284},
  year={2021},
  organization={IEEE}
}




1. Clone this repository 
   git clone https://github.com/catherine-qian/Audio-visual-sound-localization.git

2. Download the extracted features from https://nusu-my.sharepoint.com/personal/e0675871_u_nus_edu/Documents/Features
   and put the features under data/
   (you may specify the datapath in dataread.py)

3. Run python main_sslr.py -model 'MLP3' to get the results






