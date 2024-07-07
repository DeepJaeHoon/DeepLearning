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

   
2. 공간적 정보 소실의 문제

   ![3](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/9d209cd4-2e69-4f6c-8c3e-aed5aceade23)


   1차원만 입력 받을 수 있는 Fully Connected Layer의 입력으로 사용하기위해서,
  
   2차원인 사진을 강제로 펼쳐서(Flatten) 1차원으로 만들어줘야한다.

   이런 과정이 사진만이 갖고있는 공간적 정보를 훼손할 수 있다.

   각 사진 속 인접한 Pixel끼리 연관이 있을텐데, 이를 무시하고 강제로 펼치면서 훼손되는 것이다.

   예시로, 위의 사진은 우리 집 강아지 뚱자이다.

   눈이나 코의 Pixel은 주변 Pixel들은 눈이나 코와 관련된 또는 구분하기위한 정보를 담음 Pixel일 수 있다.

   하지만 Flatten을하면서 그 위치에 있었기에 의미가 있었던 Pixel끼리의 연관성을 깨버렸다. 

  
3. 변형에대한 적응성의 문제

   ![5151](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/304d3525-7eba-4fde-ad8c-b14003d22913)


   위의 사진은 우리 집 고양이 아리이다.

   위가 원본이고 아래는 2번 90도만큼 회전한 것이다.

   같은 사진 속에 고양이의 위치나 각도가 변하더라도 고양이라고 인지할 수 있어야한다.

   하지만, Fully Connected Layer는 공간적 정보 훼손과 위치와 크기, 회전에대한 불변성이 부족하기에 성능이 떨어지는 문제가 있다.

   사진의 Pixel을 강제로 펼쳤기에, 고양이의 위치가 변하면 펼친 pixel 값의 순서도 다르다.

   즉, 입력되는 값이 달라진다는 것이다.

   Fully Connected Layer는 이를 다른 입력(패턴)으로 받아들일 수 있기에 성능이 떨어지는 것이다.

   회전과 크기 변환도 같은 이유이다. 


---
### 2. CNN의 탄생 과정


1번 항목을 통해서 사진을 잘 학습하기 위해서 필요한 것에대해 간략하게 파악할 수 있다.

사진 속 고양이가 다양한 위치나 크기, 각도로 존재해도 인공지능이 "고양이"라고 답을 줬으면 하는 것이다. 

인공지능이 사진을 잘 파악하기 위해서 다음과 같은 가정을 한다.[[Paper]](https://ieeexplore-ieee-org-ssl.openlink.ajou.ac.kr/document/6795724)

 1.  locality of pixel dependencies 가정

 2.  Stationarity of statistics 가정

사진을 잘 이해시키기 위해서, 위의 가정을 고려하여 설계한 것이 Convolution Neural Network이다. 

우선, Stationarity of statistics 가정을 살펴보자.

### 2.1 Stationarity of Time-Series

정상성(Stationarity)은 시계열 분야에서 쓰이는 용어이다.

시간에 관계없이 데이터의 확률 분포는 일정하다 or 시계열의 확률적인 성질들이 시간의 흐름에따라 변하지 않는다. 

비정상성을 띈다는 것은 시간에 따라 시계열 데이터의 확률 분포가 달라진다는 것을 의미한다.

확률적인 성질은 평균, 분산과 같은 것을 의미한다.

정상성의 의미에따라 시계열은 추세(장기적으로 증가, 감소하는 경향성), 계절성(계절의 요인을 받아 일정기간 비슷한 패턴)을 가지지 않아야한다.

과거의 변동폭과 현재의 변동폭이 같아야한다. 

추세 또는 계절성이 있다는 것은 관측 시점에따라서 확률 특성이 변하는 것을 의미한다.

분산이 증가 또는 감소하는 것 역시 정상성을 만족하지 못한 시계열 데이터이다.

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/e35bb7a8-ae9c-49f5-8269-0e9ba2ffbbc2)


위의 그림에서 정상성을 가지는 그래프는 추세와 계절성이 보이지 않는 (B),(G)이다. 

(B)는 급등한 구간 이전까지는 정상성을 확실히 만족한다.

(G)는 계절성이 보이는 것 같지만 등락에 정해진 기간이 없어서 정상성을 가진다.

(A, C, E, F) = 어떠한 추세(Trend)에의해 변하므로 정상성을 띄지 않는다. 


(D, H) = 계절성을 가지고 있어서 정상성을 띄지 않는다. 

(I) = 추세 + 계절성을 가지고 있어서 정상성을 띄지 않는다. 


![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/71d434ac-a4f6-4223-b057-3dda747a4873)

회색선이 시계열이 자료의 값이다.

t1, t2, t3, t4 모두 동일한 확률분포를 가지고 있다. 이것이 시계열에서 정상성이다.  


### 2.2 Stationarity of Image

시계열에서 정상성(Stationarity)은 시간에따라 시계열 데이터가 가지는 확률 분포는 동일하다는 성질이었다.

사진(영상)에서 정상성(Stationarity)은 무엇을 의미하는 것일까? 

![코](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/b6b695c5-e6b8-44cf-b952-9e7019126df3)


사진 속 위치에 상관없이 모두 동일한 특징(패턴)을 얻을 수 있다. 

사진의 한 영역에 대한 통계적 특성이 다른 부분에서도 동일할 것이라 가정한다

다른 표현으로, 사진의 한 부분에서 특정 특징이 발견되면, 다른 위치에서도 비슷한 특징이 존재할 수 있다.

예시로, 사진 속 집사들의 코에 빨강 사각형과 파랑 사각형이 있다. 

사각형은 사진의 한 영역을 의미하며, 사각형 영역안의 Pixel들끼리만 연관성 있다는 가정을 하자.

사각형은 CNN이 사진을 관심있게 볼 집중 영역이며 나머지 영역은 참고하지 않는다.

여기서, 사각형의 역할이 사람의 코에대한 특징을 추출하는 역할을 한다고 해보자. 색상은 빨강과 파랑으로 구분해서 그렸지만 동일한 필터이며 위치가 다름을 보여주기 위해서 색상을 구분해서 그렸다. 

빨강 사각형 영역이 가지는 특징이 파랑 사각형이 가지는 특징과 동일(유사)하다는 가정이다.

![vzvzvzvz](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/4f8b76f6-2148-4a74-ac78-2ea9e1c387f9)


사각형은 어떠한 확률 분포를 가지고 있다. 이 확률 분포를 위치에 상관없이 공유하기에, "코"에대한 특징을 위치에 상관없이 구할 수 있다.

즉, 어떠한 확률 분포를 가진 사각형이 사진 영역 안을 관찰하면서 지나갈때마다, 사각형이 가지는 확률분포는 변함이 없기에 사진 영역 전부를 지나갈 때까지 모든 Pixel은 사각형의 확률분포를 사용한다. 

그래서 빨강 사각형 위치에 있을때나 파랑 사각형 위치에 있을때나 동일한 "코"라는 특징을 얻을 수 있다. 

후술하겠지만, 동일한 Filter(Kernel이라고도 부르며 여기서는 사각형을 의미)가 사진 안에서 위치에 상관없이 동일한 확률 특성 or 특징을 얻을 수 있음을 의미한다. 

Stationarity 가정이 있기에 CNN은 Parameter Sharing을 할 수 있고, Stationarity 가정은 Convolution이라는 수학 연산자로 만족된다.

---

회귀 분석을 할 때 오차의 정규성, 등분산성을 만족해야 그 회귀 분석 결과를 믿을 수 있는 것













Translation invariance & Translation equivariance
long range dependency -> cnn, vit 비교 
