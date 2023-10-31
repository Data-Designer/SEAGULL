<h1 align="center"> Cross-Domain Recommendation via Progressive Structural Alignment </h1>

<p align="center">
  <!-- <img src="https://img.shields.io/badge/ChuangZhao-BCWF-orange">
  <img src="https://img.shields.io/github/stars/ChuangZhao/SEAGULL">
  <img src="https://img.shields.io/github/forks/ChuangZhao/SEAGULL">
  <img src="https://img.shields.io/github/issues/ChuangZhao/SEAGULL">  
  <img src="https://img.shields.io/github/license/ChuangZhao/SEAGULL"> -->
  <a href="https://github.com/Data-Designer/SEAGULL">
    <img src="https://img.shields.io/badge/ChuangZhao-SEAGULL-orange">
  </a>
  <a href="https://github.com/Data-Designer/SEAGULL/stargazers">
    <img src="https://img.shields.io/github/stars/Data-Designer/SEAGULL">
  </a>
  <a href="https://github.com/Data-Designer/SEAGULL/network/members">
    <img src="https://img.shields.io/github/forks/Data-Designer/SEAGULL">
  </a>
  <a href="https://github.com/Data-Designer/SEAGULL/issues">
    <img src="https://img.shields.io/github/issues/Data-Designer/SEAGULL">
  </a>
<!--   <a href="https://github.com/Data-Designer/SEAGULL/graphs/traffic">
    <img src="https://visitor-badge.glitch.me/badge?page_id=Data-Designer.SEAGULL">
  </a> -->
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<a href="https://github.com/Data-Designer/SEAGULL#contributors-"><img src="https://img.shields.io/badge/all_contributors-1-orange.svg"></a>
<!-- ALL-CONTRIBUTORS-BADGE:END -->
  <a href="https://github.com/Data-Designer/SEAGULL/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Data-Designer/SEAGULL">
  </a>
</p>


## About Our Work

Update: 2023/09/14: We have created a repository for the paper titled *Cross-Domain Recommendation via Progressive Structural Alignment*, which has been submitted to the *IEEE Transactions on Knowledge and Data Engineering (TKDE)* journal. In this repository, we offer the original sample datasets, preprocessing scripts, and algorithm files to showcase the reproducibility of our work.

![image-20230914222242414](https://s2.loli.net/2023/09/14/ZujVotqlGLhrxcR.png)

## Requirements

- Python == 3.8
- Pytorch == 1.11.0
- DGL == 0.9.1
- gensim == 3.8.3
- nltk == 3.7
- stanfordcorenlp == 3.9.1.1

## Data Sets

The structure of the data set should be like

```
Douban
|_ douban_feature_raw
|  |_ bookreviews_cleaned.txt
|  |_ books_cleaned.txt
|  |_ moviereviews_cleaned.txt
|  |_ movies_cleaned.txt
|  |_ music_cleaned.txt
|  |_ musicreviews_cleaned.txt
|  |_ users_cleaned.txt
|_ douban_feature
|_ douban_movie
|_ douban_book
|_ douban_music
Amazon
|_ ...
|_ ...
```

Due to file size limitations, we have not uploaded all of the data. The Amazon data can be obtained from [this website](https://jmcauley.ucsd.edu/data/amazon/), while the Lenovo data is commercially licensed and requires you to request access from us.

## RUN

```powershell
# unzip all files into the douban_feature_raw directory
# preprocess could be found in GADTCDR, movie should be 20.
python main.py # main file
```

## Contact

If you have any questions, please contact me via [zhaochuang@tju.edu.cn](zhaochuang@tju.edu.cn).

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
Thanks to the data preprocessing piplines in GADTCDR https://github.com/FengZhu-Joey/GA-DTCDR
<table>
  <tr>
    <td align="center"><a href="https://data-designer.github.io/"><img src="https://avatars.githubusercontent.com/u/26108487?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Chuang Zhao</b></sub></a><br /><a href="#ideas-ZhiningLiu1998" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Data-Designer/JOC/commits?author=Data-Designer" title="Code">💻</a></td>
  </tr>
</table>


This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
