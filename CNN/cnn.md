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

   만약, Data Set이 [1024, 1024, 3]을 가지는 컬러 사진 4장이라면, Fully connected Layer의 파라미터 수에비해서 데이터셋이 적기에,

    차원의 저주 현상으로 과적합이 일어날 수 있다.

   Fully connected Layer의 파라미터 수가 증가한다는 것은 그만큼 다양한 상황을 표현할 수 있다는 것이지만, 반대로 한 상황에 대해서만 과적합도 잘한다는 의미이다. 

   
3. 공간적 정보 소실의 문제

   ![3](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/9d209cd4-2e69-4f6c-8c3e-aed5aceade23)


   1차원만 입력 받을 수 있는 Fully Connected Layer의 입력으로 사용하기위해서,
  
   2차원인 사진을 강제로 펼쳐서(Flatten) 1차원으로 만들어줘야한다.

   이런 과정이 사진만이 갖고있는 공간적 정보를 훼손할 수 있다.

   각 사진 속 인접한 Pixel끼리 연관이 있을텐데, 이를 무시하고 강제로 펼치면서 훼손되는 것이다.

   예시로, 위의 사진은 우리 집 강아지 뚱자이다.

   눈이나 코의 Pixel은 주변 Pixel들은 눈이나 코와 관련된 또는 구분하기위한 정보를 담음 Pixel일 수 있다.

   하지만 Flatten을하면서 그 위치에 있었기에 의미가 있었던 Pixel끼리의 연관성을 깨버렸다. 

  
4. 변형에대한 적응성의 문제

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

#### 2.1 Stationarity of Time-Series

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


#### 2.2 Stationarity of Image

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

#### 2.3 Locality Of Pixel Dependencies

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/780ee6c4-512b-4515-8b98-eb3656091964)

사진에서 pixel의 종속성은 특징이 있는 작은 지역으로 한정된다. 

위의 사진을 보면 "코"라는 특징은 파란색 사각형 안에 있는 pixel에서만 표현되고 해당 pixel끼리만 관계를 가진다고 할 수 있다.

빨간색 사각형안의 pixel은 파란색 사각형안의 pixel과는 종속성(연관)이 없다는 가정이다.

즉 이미지를 구성하는 특징들은 사진 전체가 아닌 일부 지역에 근접한 pixel로만 구성되고 근접한 pixel끼리만 종속성(연관성)을 가진다는 가정이다.

---

### 3. Convolution이란 무엇인가?

왜 합성곱을 사용하는지 의문을 가진적이 있는가?

합성곱은 어떠한 신호나 정보를 목적(변환, 분해, 필터링, etc...)에 맞게 설계하기 위함이다. 

사진의 edge만 추출하고자 사진 pixel에다가 edge detection filter를 합성곱할 수 있다. 

![3](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/2fb3aefc-cc87-4bfa-b5b0-c626f94ae054)

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/47555329-b152-47fd-ae71-b534578e5b9d)


기존에는 사진을 처리할 때, 고정된 값을 가지는 filter(kernel)을 사용하였다. 

이러한 filter를 사용하여 사진에대한 분류나 특징을 잡으려고 하였다.

filter를 수식에서 w라고 한다면, 원본 사진 f에다가 2D Convolution을 적용하여 특징을 추출한 사진 g를 얻는 방식이었다. 

하지만, filter를 설계할 때 사람의 직관이나 굉장한 노가다가 필요한 작업중 하나였다. 

사람들은 이러한 수고를 줄이고자 자동으로 개발자의 목적에 맞는 filter를 설계하는 알고리즘을 개발하고자한 것이 CNN이다.

![Convolution_of_box_signal_with_itself2](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/cc65e16d-8144-4073-aa0c-47660bdfd752)

![vzvdd](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/25ec8315-2bf5-4c0f-ae39-8c8c51667394)

Convolution을 간략하게 이해해보자.

수학적으로 간단하게 설명하면 입력함수를 고정해두고, 대상함수를 반전시켜 $\tau$만큼 평행이동시키면서 서로 곱해서 새로운 결과를 얻는 것이다. 

이걸 보고 이해하는 사람은 거의 없다. 밑의 예시로 다시 이해해보자.

얼룩지고 더러운 바닥이 있다고 가정해보자. 이때, 우리는 청소부고 물걸래질로 더러운 바닥을 청소하는게 목적이다.

물걸래로 바닥을 문대서 닦으면 깨끗한 바닥이 된다. 이 과정이 Convolution이라고 볼 수 있다. 

물걸래는 수식에서 g(t - $\tau$)이고 더러운 바닥이 f(t)이며, 깨끗한 바닥이 (f*g)(t)이다. 

더러운 바닥을 깨끗하게 변환시키고싶어서 고정해두고, 물걸래의 면적이 $\tau$만큼 이동하면서 지나간 면적만큼 더러운 바닥과 서로 연산돼서 깨끗한 바닥((f*g)(t))을 만든 것이다. 

Image에서는 어떠한 의미일까? 

특징을 추출하거나 분류하고싶은 사진(더러운 바닥)을 고정해두고, Filter(물걸래)가 사진을 훑고 지나가면서 새로운 특징(깨끗한 바닥)을 만들어낸다. 
수학적으로는 사진이 f(t), Filter가 g(t - $\tau$), 사진의 특징만 추출한 Feature Map이 (f*g)(t)이다. 

**Convolution Neural Network은 개발자의 목적(Object Detection, Segmetation, etc..)에 맞게 어떠한 사진(음성 기록이나 시계열 자료)에 대해 목적에 달성하기 위한 적절한 Filter를 설계(학습)하는 것이 목표다.**

---

### 4. Convolution Neural Network의 연산 과정

![unnamed](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/22c78e93-7cbc-4cda-9a66-4eb8a55b2a61)

padding이니 polling이니 여기다 적어

feature map 의미, 숫자 크면 뭐가 좋나


---

### 5. CNN이 가지는 특성 
CNN을 설계하기위한 가정들과 연산 과정 덕분에 얻게되는 다양한 특징들이 있다.

#### 5.1 Sparse Connectivity

![vzvzvzzvvzvzvzvzv](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/4cf797b1-9b6f-4eb8-82ba-dbb356336c50)



#### 5.2 Parameter Sharing

CNN은 각 Layer마다 사진 또는 Feature Map을 대상으로 동일한 Filter를 사용한다. 

즉, Filter가 가지고 있는 가중치를 공유한다.

Filter를 왜 공유할 수 있는걸까? 그 이유는 항목 2.2의 Stationarity 가정이 있기에 가능하다.

동일한 Filter로 사진 안에서 위치에 상관없이 동일한 특징을 얻을 수 있기에, 

Covolution 연산처럼 사진에다가 똑같은 Filter를 슥슥 문질러주면 위치와 상관없이 동일한 특징을 얻을 수 있다.

![코](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/b6b695c5-e6b8-44cf-b952-9e7019126df3)


Parameter Sharing의 이점은 무엇일까?

위의 사진에서 인공지능이 어떻게 처리하는게 좋을까?

1. 위치에 상관없이 둘 다 "코"라는 특징 추출

2. 위치를 고려하여 "정면에서 본 오른쪽 코", "사선에서 본 왼쪽 코"라는 특징을 추출

1번과 2번 무엇이 더 효율적인가?

2번은 위치마다 서로 다른 특징이라고 인지를 시켜줘야해서, 위치마다 서로 다른 Filter를 적용해줘야하는 문제가 생긴다. 

1번과 같이 위치에 상관없이 동일한 특징을 얻을 수 있게 동일한 Filter를 사용해서(Parameter Sharing) 특징을 얻는다.

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/284588b4-570c-4a1c-a5ab-3ac4815d863a)


동일한 Filter를 공유하기에 Fully Connected Layer와 비교해서 필요한 Parmeter 수의 감소와 memory도 아끼고 연산량도 아낄 수 있고 statistical efficiency가 증가한다.

statistical efficiency의 의미는 더욱 효율적인 모델은 상대적으로 데이터 수를 적게 학습시켜도 더 좋은 성능을 내야한다는 의미다.

Parameter Sharing으로 한장의 Feature Map을 만드는데 동일한 특징을 여러곳에서 볼 수 있어서,

Fully Connected Layer에비해 동일한 학습 데이터셋 상황에서도 더 많은 데이터를 학습한 효과를 갖게 되고 결국 statistical efficiency가 향상된다.

Fully Connected Layer는 사진의 크기, 위치, 각도등이 변하면 새로운 pattern으로 인식하여 새롭게 배워야하지만,

CNN의 Filter는 Stationarity 가정 덕분에, 가중치를 공유하는 Filter가 다양한 위치에서 동일한 특징을 뽑아낸다.

가중치를 공유하는 Filter가 사진을 훑으면서 학습하기에 사진 한 장으로 더 많은 학습 데이터를 보는 느낌이다.  

#### 5.3 Translation invariance & Translation equivariance

Invariance와 equivariance의 의미는 각각 불변성과 가변성을 의미한다. 

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/e6f50400-6279-4c69-9d3a-7fc3bb28f4e2)


Translation invariance는 사진의 위치가 변해도 출력은 동일하다는 것이다.

Translation equivariance는 사진의 위치가 변하면 출력도 변한다는 것이다.

Classification에서 위 사진의 고양이가 어느 위치에 있더라도 "고양이"라고 출력을 내고싶어한다.

이것이 Translation invariance를 의미한다. 

그렇다면, Convolution Neural Network는 Translation invariance일까?

CNN의 핵심 연산인 Convolution을 먼저 알아보자.

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/acdc3831-6bb5-4cf4-af44-ee22d1965d4d)

사진 속 고양이의 위치가 11시 방향인 사진과, 5시 방향인 사진이 있다고 가정하자. 

이 사진을 동일한 Filter로 Convolution한 결과는 어떨까?(Stationarity 가정에 의한 Parameter Sharing)

위의 사진에서 보이듯이 똑같은 값의 결과가 위치만 다르게해서 나온다.(Stationarity 가정에 의한 위치가 달라도 동일한 특징을 뽑아냄)

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/97af3512-f66b-4bfb-824d-d9b9e3f2e810)


즉, Convolution의 연산 자체는 위치에따라 결과의 위치도 다르게 나오는 Translation equivariance이다. 

그렇다면 CNN은 어느 부분에서 Translation invariance의 성질을 가질 수 있을까?



![Screenshot_20240709_155937_Samsung Notes](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/8640b05f-295d-4b1f-a5bd-d5d3cf8a1c96)

첫 번째가 Pooling이다. 

만약 위의 그림처럼 어떠한 입력 [0, 1, 0, 0]이 있다. 이 값을 이동 시켜서 [0, 0, 1, 0]로 이동한 입력이 있다고하자.

이 둘에게 2 x 2 Max Pooling을 취해보면 둘 다 "1"이라는 값으로 똑같이 나온다.

위치가 변했음에도 둘 다 똑같은 결과를 보여준다. 

Pooling 영역 안에서 값들의 순서가 바뀌든 말든 결과가 동일한 Translation invariance 성질을 보여준다. 

K x K Pooling size 안에서는 Translation invariance 연산이다. (Pooling 종류 Max든, Min이든, Mean이든 상관 X)

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/9c2e95a6-9ac0-43d5-9ea7-acd0ab253120)


두 번째가 Parameter Sharing이다. 

동일한 Filter를 사용해서 Convolution 연산은 앞에서 Translation equivariance라고 하지 않았던가?

수학적 연산으로는 Translation equivariance이 맞다. 

모순되게도, Filter의 Parameter Sharing 덕분에 Translation Invariant 성질도 어느정도 갖게된다. 

Parameter Sharing을 하지 않으면, 위치마다 다른 Filter를 사용해서 사실상 Translation Invariant 성질을 잃는다.

예시로, 사진 속 같은 고양이라도 위치가 다르다고 고양이라고 인지를 못할 수 있다.

입력 위치가 변하면 최종 결과가 달라지는 Translation equivariance이다.

Parameter Sharing을 해서 다른 위치에 있더라도 고양이라고 인지하게 해줄 수 있다면 Translation Invariant 성질이 생긴 것이라 볼 수 있다.

Parameter Sharing은 연산은 Translation equivariance하게, 결과는 Translation Invariant한 상황이다.

이러한 성질을 가질 수 있게하는 다른 방법이 또 있을까?

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/d97a7ba8-7bd8-4935-b544-734daec663ac)

위의 사진을 보면 서로 같은 패턴이지만 위치가 다른 Feature Map 여러장을 가지고 있다. 

임의의 Filter로 Convolution을 하게 되면, 같은 출력 결과가 다른 위치에 나오게 될 것이다.(Translation equivariance)

하지만, Pooling과 그 결과를 Fully Connected Layer에 통과하고 Softmax같은 활성화 함수를 지나면 동일한 확률 결과를 출력하는 것을 볼 수 있다. (Translation Invariant)

**사진속 객체의 위치와 상관없이 Pattern이 동일하다면 같은 출력을 보여주는 Translation Invariant 성질을 가지게 되었다.**

정리하면, Convolution 연산 과정 자체는 Translation equivariance한 상태이다.

Convolution이 Translation equivariance한 성질은 Feature Map의 크기 안에서만 허용된다. 

**Fully Connected Layer의 경우에 Translation equivariance한 입력을 주면 결과도 Translation equivariance이다. (이 부분 맞는지 검토 필요)**

**Translation Invariant한 입력을 주면 결과도 Translation Invariant한 입력을 가진다.**

Pooling의 경우 Pooling Size만큼 Small Translation Invariant한 성질을 가지게 해준다.

그렇기에 Pooling 후에 Fully Connected Layer 통과하면 Translation Invariant 성질을 점점 가질 수 있다.

또한, CNN에서 Softmax같은 활성화 함수도 Translation Invariant 성질을 가질 수 있게한다.

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/c11e1c3e-97e3-4e7e-b015-e26dfcb3abba)

파란색 이미지 하단 왼쪽에 눈과 코가 있다고 가정한다.

녹색은 Convolutuin Layer의 Feature Map으로, 눈 채널과 코 채널 각각 하단 왼쪽에서 눈과 코 위치를 따라 큰 활성화 값이 출력된다.

노란색은 더 깊은 Convolutuin 연산 결과의 feature map으로 face 채널과 leg 채널 등이 있는데, 

이전 feature map에서 큰 활성화 값이 나온 왼쪽 하단 영역의 눈, 코 채널을 합쳐 face 채널의 왼쪽 하단에서 큰 활성화 값이 출력된다.

여기까지는 각 입력에서 특징 위치와 동일하게 출력이 되겄기에 Translation Equivariant하다.

하지만 Fully Connected Layer를 거치고 마지막 label 확률을 출력하는 부분에서는 위치와 상관없이 human body가 detect되었다.

이러한 과정이 Translation Invariant 하다고 할 수 있다.

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/4c901f1b-8da4-4106-8ab8-ac6a9a787e36)

눈, 코 특징이 사진의 왼쪽 상단에 존재한다.

Convolution 연산은 Equivariant해서 서로 다른 위치에 있는 특징을 입력으로 넣으면,
 
Feature Map에서도 각 특징을 서로 다른 위치에 놓이게 되는 Translation Equivariant이다.

하지만, Fully Connected Layer와 Softmax를 통과한 값은 똑같이 human body가 나왔다. 

Convolution Layer를 지나 Fully Connected Layer와 Softmax를 거친 결과는

특징의 위치와 상관없이 무조건 특징이 포함된 라벨의 확률 값을 높게 출력한다.

즉, 객체가 어디에 위치하던 객체의 확률 값은 높게 나오는 Translation Invariant이다.

정리하자면 Convolution 연산의 Equivariance한 특성과 파라미터를 공유하는 성질이 CNN 자체가 Translation Invariant 특성을 갖게 된다.

#### 5.4 Translation invariance가 항상 옳은 것일까?

![image](https://github.com/DeepJaeHoon/DeepLearning/assets/174041317/ea61ca77-8a85-4a71-bf5f-825f23e739f8)

위치 정보가 중요한 경우에 Translation Invariance가 반가운 성질은 아닐 것이다.

위의 사진은 Translation Invariance의 문제를 극단적으로 묘사한 것이다.

오른쪽 이목구비가 완전 뭉개진 사진을 CNN은 "얼굴"이라고 인지하는 문제를 지적한 것이다.

이런 문제는 Translation Invariance의 성질을 가진 CNN에서 나타나는 문제이다.

위치 정보가 중요한 경우에 Parameter Sharing을 하지 않는 방식도 있다.

동일한 특징이어도 위치마다 다른 Filter를 적용하는 방식이다.

Filter가 위치 정보를 내포하고 있으며 Translation Invariance의 성질을 버리는 것이다.

다만, Pooling을 통해서 Small Translation Invariance을 어느정도 챙길 수 있다. 

---

covolution이 왜 cnn 가정을 만족할 수 있찌?

cnn 오차 역전파와 Fully connected layer로 표현 가능한지, 

Inductive bias에 대해서

receptive field

layer 층수별로 어떠한 feature를 뽑나 

long range dependency -> cnn, vit 비교 

local feature 추출 내용
