# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박준수](https://github.com/UpstageAILab)             |            [손은혜](https://github.com/UpstageAILab)             |            [임환석](https://github.com/UpstageAILab)             |            [황은혜](https://github.com/UpstageAILab)             |            [윤소영](https://github.com/UpstageAILab)             |
|                            팀장, 모델링 및 데이터 분석                             |                            모델링 및 데이터 분석                             |                            담당 역할                             |                            발표자료 작성 도움
                             |                            담당 역할                             |

## 0. Overview
### Environment
- 주피터 노트북
- Upstage GPU

### Requirements
- GPU

## 1. Competiton Info

### Overview

- 의료·금융·보험·물류 등 다양한 산업에서 대량의 문서 이미지를 자동으로 식별, 처리하기 위해 활용되는 CV 분야의 핵심 태스크인 문서 타입 이미지 분류(Image Classification) 문제를 다룹니다.

### Timeline

2025. 10. 31 ~ 2025. 11.12

## 2. Components

### Directory

- 팀원별 폴더 만들어서 업로드


```
├── Limhwanseok
│   ├── augmentation.ipynb (리더보드 제출 코드)
│   ├── image_vision.ipynb
│   └── swin_large_augmentation.ipynb (리더보드 제출 코드)
├── parkjunsu
│   ├── baseline.ipynb
│   ├── baseline_v1.py
    ├── baseline_vit_.ipynb
    ├── classcification.ipynb
    ├── classcification_V2.ipynb
    ├── frist_classification.py
    ├── object_dec_and_aug.ipynb
    ├── object_dec_and_aug_v2.ipynb  
│   └── vit_augmented_baseline.ipynb
└── soneunhye
    ├── 01_1st_EDA.ipynb
    ├── 01_1st_EDA_based_0.6038_scored.ipynb
    ├── 02_1st_EDA_based_0.5901_scored.ipynb
    ├── 02_2nd_EDA.ipynb
    ├── 03_1st_EDA_based_0.6261_scored.ipynb
    ├── 04_1st_EDA_based__0.6483_scored.ipynb
    ├── 05_2nd_EDA_based_0.6113_scored.ipynb
    ├── 06_2nd_EDA_based_0.4992_scored.ipynb
    ├── 07_2nd_EDA_based_0.6041_scored.ipynb
    ├── 08_2nd_EDA_0.6669_scored.ipynb
    ├── 09_no_submission.ipynb
    ├── 10_2nd_EDA_0.6827_scored.ipynb
    ├── 11_2nd_EDA_0.6670_scored.ipynb
    ├── 12_2nd_EDA_0.7402_scored.ipynb
    ├── v1_object_dec_and_aug_v2.ipynb 
    └── v2_object_dec_and_aug_v2.ipynb    
```

## 3. Data descrption

### Dataset overview

train datasets 이미지 레이아웃 확인 
객체 이미지 + 문서이미지
대부분 캡쳐본으로 깨끗하고 노이즈가 많이 없는 상태

test datasets 이미지 레이아웃 확인
회전, 반전, 흐림, 이미지 혼합(섞임), 잘림 등 다양한 유형의 노이즈 및 변형 포함

<img width="1033" height="264" alt="image" src="https://github.com/user-attachments/assets/496940bc-3746-4f05-8adf-4763daafb983" />


### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- ViT (Swin Large, 384)
  

### Modeling Process


<img width="1222" height="412" alt="image" src="https://github.com/user-attachments/assets/3edeeb48-64c1-49a3-95a6-72384e785882" />


## 5. Result

### Leader Board

<img width="817" height="607" alt="image" src="https://github.com/user-attachments/assets/3bb173d1-3456-40fb-aded-d7c4a3f343fa" />


### Presentation

- https://docs.google.com/presentation/d/1AdBPM1qlEKxrxkfW6FrzrgiNRj-qfuXR/edit?pli=1&slide=id.g399552807a2_2_70#slide=id.g399552807a2_2_70

## etc

### Meeting Log

<img width="720" height="1085" alt="image" src="https://github.com/user-attachments/assets/35914120-c137-44d9-93ca-c27a22ce2ad3" />

<img width="433" height="241" alt="image" src="https://github.com/user-attachments/assets/6a3946c0-ee8c-4f30-85fc-c595bbbb01e6" />



