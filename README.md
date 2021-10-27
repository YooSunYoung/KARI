# KARI

KARI SAR image object detection

## Preprocessing

`Preprocessing.preprocess.py`

전처리는 두 가지 종류의 데이터에 대해 총 네 단계에 걸쳐 이루어집니다.

### 데이터 종류
1. 원본 h5 이미지
2. Sub-band analysis 를 통해 얻은 총 4개의 밴드의 이미지

### 처리 단계 및 결과물
```
PARSE_DIRECTORY_STRUCTURE = 0
IMAGES_TO_NUMPY = 1
CROP_IMAGES = 2
RESIZE_CROPPED_IMAGES = 3

phase = RESIZE_CROPPED_IMAGES
```
메인 함수의 `phase` 변수에 들어가는 숫자를 조절하면 해당 단계를 실행합니다.
1. Parse Directory Structure
    - 전체 데이터셋의 디렉토리 구조를 파싱해서 json 파일에 담습니다.
      </br> 해당 결과물은 `data/directory_structure/` 폴더에 담겨있습니다.
      
2. Image to Numpy
    - 각 원본 이미지들을 넘파이 오브젝트로 바꾸어줍니다.
      </br> 이 때, 각 이미지는 지역, 날짜 상관 없이 데이터 그룹(선별/비선별) 순서대로 저장됩니다.
      </br> 타겟의 위치와 크기도 레이블로 함께 저장합니다.
    - 결과물은 npz 파일입니다.
    
3. Crop Images
    - 원본 이미지를 조각냅니다.
    - 이미지를 조각 내면서 레이블도 함께 보정하여 업데이트해줍니다.
    - 이 때, 원본 이미지에서의 조각 이미지의 좌상단 픽셀 위치도 함께 저장합니다.
    - 결과물은 npz 파일입니다.
    
4. Resize Images
    - 이미지의 크기를 일정하게 조정합니다.
    - 이미지의 크기와 함께 레이블의 위치도 보정해줍니다.
    - 결과물은 npy 파일입니다.
    
## Data Summary

1. train/test/validation set splitting strategy
2. train-set summary
3. test-set summary
4. validation-set summary

## Machine Learning

1. loss function
2. Faster RCNN
3. load cropped and resized image and labels
4. train hyper-parameters

> ref: https://github.com/FurkanOM/tf-faster-rcnn
