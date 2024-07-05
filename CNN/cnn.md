# Convolution Neural Network
---

### 1. 기존에는 무엇이 문제였는가?

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/674dce47-3aac-4338-a3ec-dac71a5bf6e8)


CNN의 등장이전 Image Pattern학습은 Fully Connected Layer로 이루어졌다. 
사진과 같은 2차원 정보를 사용하기에는 다음과 같은 문제가 발생했다.

1. 이미지 크기에 비례한 고차원 매개변수의 문제

   Fully Connected Layer는 Input Node -> Output Node의 연결이 모두 연결된 구조이다. 

   만약, 입력한 사진의 크기가 너비 W, 높이 H, 채널 C를 가진다면 첫 입력 계층의 파라미터수는 W x H x C이다.

   너비 640, 높이 480, RGB채널을 입력으로 준다면 640 x 480 x 3 = 921,600개의 입력 node가 필요하다.

   만약, 그 다음 layer의  Node를 128개로 가정한다면, 학습에 필요한 매개변수는 921600 x 128 = 117,964,800이다.

   이미지를 Flattent을 하면서 Fully connected Layer에 필요한 입력의 수는 증가하고 그만큼 학습에 필요한 파라미터 수도 증가한다.

   만약, Data Set이 [1024, 1024, 3]을 가지는 컬러 사진 4장이라면, Fully connected Layer의 파라미터 수에비해서 데이터셋이 적기에 과적합이 일어날 수 있다.

   Fully connected Layer의 파라미터 수가 증가한다는 것은 그만큼 다양한 상황을 표현할 수 있다는 것이지만, 반대로 한 상황에 대해서만 과적합도 잘한다는 의미이다. 

   즉, 입력되는 사진의 크기가 클수록 필요한 연산량은 증가하고 학습시에 과적합의 위험이 존재한다. ==>> 수정 필요

   
3. 공간적 정보 소실의 문제

   ![3](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/9d209cd4-2e69-4f6c-8c3e-aed5aceade23)


   1차원만 입력 받을 수 있는 Fully Connected Layer의 입력으로 사용하기위해서,
  
   2차원인 사진을 강제로 펼쳐서(Flatten) 1차원으로 만들어줘야한다.

   이런 과정이 사진만이 갖고있는 공간적 정보를 훼손할 수 있다.

   각 사진 속 인접한 Pixel끼리 연관이 있을텐데, 이를 무시하고 강제로 펼치면서 훼손되는 것이다.

   예시로, 위의 사진은 우리 집 강아지 뚱자이다.

   눈이나 코의 Pixel은 주변 Pixel들은 눈이나 코와 관련된 또는 구분하기위한 정보를 담음 Pixel일 수 있다.

   하지만 Flatten을하면서 그 위치에 있었기에 의미가 있었던 Pixel끼리의 연관성을 깨버렸다. 

  
5. 변형에대한 적응성의 문제

   ![5151](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/304d3525-7eba-4fde-ad8c-b14003d22913)


   위의 사진은 우리 집 고양이 아리이다.

   위가 원본이고 아래는 2번 90도만큼 회전한 것이다.

   같은 사진 속에 고양이의 위치나 각도가 변하더라도 고양이라고 인지할 수 있어야한다.

   하지만, Fully Connected Layer는 공간적 정보 훼손과 위치와 크기, 회전에대한 불변성이 부족하기에 성능이 떨어지는 문제가 있다.

   사진의 Pixel을 강제로 펼쳤기에, 고양이의 위치가 변하면 펼친 pixel 값의 순서도 다르다.

   즉, 입력되는 값이 달라진다는 것이다.

   Fully Connected Layer는 이를 다른 입력(패턴)으로 받아들일 수 있기에 성능이 떨어지는 것이다.

   회전과 크기 변환도 같은 이유이다. 


### 2. CNN의 탄생 과정

[[Paper]](https://ieeexplore-ieee-org-ssl.openlink.ajou.ac.kr/document/6795724)


1번 항목을 통해서 사진을 잘 학습하기 위해서 필요한 것에대해 간략하게 파악할 수 있다.

인공지능이 사진을 잘 파악하기 위해서 다음과 같은 가정을 한다. 

 1. Locality 가정

 2. Stationarity 가정

 3. Translation invariance & Translation equivariance





long range dependency -> cnn, vit 비교 
