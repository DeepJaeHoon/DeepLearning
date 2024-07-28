---
### 기존에는 무엇이 문제였는가?

![grgrgr](https://github.com/user-attachments/assets/72ce2337-5f2c-4980-81fa-2dc6561beec4)

Transformer는 자연어 처리, 주식 예측이나 경로 예측 같은 시계열 분석에도 유용하게 쓰이고 있다. 

Transformer를 개발하기 이전에는 어떠한 방식으로 위의 문제를 해결했을까?

RNN을 사용하여 위의 문제를 해결하려 하지만 시계열의 입력 길이가 길수록 gradient vanishing 문제가 발생하였다.

RNN의 gradient vanishing 문제를 해결하고자 LSTM을 사용하였지만 완전히 해결하지는 못했다.

![fszas](https://github.com/user-attachments/assets/657c2415-104b-48c8-be16-4e480bcec1a2)

자연어의 경우 입력되는 시계열의 길이와 출력되는 시계열의 길이가 다른 문제가 있어서 Seq2Seq를 도입했다.

Seq2Seq는 Encoder와 Decoder 개념을 도입했다. 시계열을 RNN 계열을 사용해서 feature를 뽑는 방식이다.

예시로 Encoder에는 한국어를 RNN으로 Feature를 뽑고 Decoder로 넘겨주면, 최종 출력은 한국어를 영어로 번역해주는 구조이다. 

하지만 기존 Seq2Seq의 경우 Encoer의 값을 Decoder로 넘겨주는 값(Context Vector)이 고정된 크기를 가져 정보 손실이 있다.

![gwgweg](https://github.com/user-attachments/assets/93cdd182-9f49-43c4-bc77-70c2a3d3c264)

위의 문제를 극복하고자 Attention이라는 개념이 도입했다. Decoder에서 매 시점마다 Encoder의 feature를 Attention을 통해 참조하는 구조다.

하지만 이러한 방식으로 고정된 크기의 Context Vector가 가지는 문제를 극복해도 문제가 남아있다. 

RNN 계열의 구조는 시간이 흐를수록 과거 정보가 흐릿해지는 구조를 가지고 있는 것이다.  

흐릿해진 정보를 담고 있는 feature에 Attention을 하고 있으니, RNN에 의한 Seq2Seq의 근본적인 문제는 해결을 못한 것이다. 

![image](https://github.com/user-attachments/assets/07f9ed89-cc89-44ac-8629-cad278094949)

시간이 지날수록 과거 정보가 흐릿해지는 것을 보완하고자 Bidirectional RNN이 나왔다.

하지만, 거리에 의존적(long term dependency)이라는 문제가 남아있다. 

거리에 의존적이다는 0초~10초 자료를 RNN을 통해서 계산하면, 10초에서 0초의 feature는 7초에서의 0초 feature보다 흐릿하게 남아있다.

시간(순서)상 먼 거리일수록 정보가 흐릿해지는 것을 의미한다. Bidirectional은 반대로 10초에서 시작해서 0초로 RNN 통과하는 방식이다.

이 역시, 0초에서 10초의 feature는 7초에서의 10초 feature보다 흐릿하게 남아있다.

Bidirectional 방식으로 위의 0초에서 10초, 10초에서 0초 feature를 concat을 해도 첫 입력정보가 갈수록 흐려지는 거리 의존성 문제는 남아있다. 

또한, RNN 계열의 방식은 다음 시계열의 정보를 알기위해서 현재와 과거의 정보에대해서 연산해주는 과정이 필연적이다.

이러한 연산 방식은 직렬 방식이라 병렬로 해결할 수 없어서 연산 속도가 느리다는 단점이 남아있다.

RNN 계열을 사용하는 한, 병렬 문제, 거리 의존성과 기울기 소실, 과거 정보 소실 문제는 남아있기에 RNN 계열에서 벗어날 방식을 제안한게 Transformer이다. 

Transformer는 RNN, CNN을 아예 사용하지 않고 Attention만으로 시계열 정보를 효과적이게 처리했다.

---
# ENCODER

![제목 없음fsdfsd](https://github.com/user-attachments/assets/d072ceb6-5e7c-474e-8453-0368b7f86de5)

Transformer의 Encoder 구조는 위와 같은 구조를 가진다. 

어떠한 시계열 정보가 Input Embedding Layer에서 Toeken화가 진행된다. 

어떠한 문장이 주어지면 word2vec으로 model이 이해할 수 있는 언어로 바꿔주거나 각 시계열 step마다 가진 변수들을 추상화하는 역할이다. 

정리하면. 문장이나 어떠한 시계열(주식, 주행 경로)을 model이 이해할 수 있는 언어로 바꿔주는 단계이다. 

Positional Encoding은 Embedding Layer의 결과에 시간 순서 정보를 추가해주는 역할이다.

그 이후, Multi Head Attention을 통해서 주어진 시계열에 대해 문맥 정보를 파악하여 Decoder로 넘겨주는 역할이다. 

Transformer에서 Attention이 중요하기에 먼저 설명하고 나머지에 대해서 설명하겠다. 

## 1.1 Attention의 의미는 뭘까?

![image](https://github.com/user-attachments/assets/298ad028-44e6-4dbb-9bf1-7e18f892a6cf)

Attention은 무엇인가? 

Attention은 단어 의미 그대로 model이 누구한테 더 집중해서 참고를 해야하나 지시하는 역할이다. 

만약 Transformer를 공부하고자 하는 학생이 Transformer와 관련된 내용을 집중하지, 생명과학II에 집중하지 않듯이 말이다.

그럼 model이 무언가에 집중하라는 개념을 어떤 방법으로 구현을 했을까?

대표적이고 주로 쓰이는 방식이 아래의 scaled dot product attention이다.

그림에서 Mask(opt.)가 있다. 이는 Padding Mask를 적용할건지 또는 Diagonal Mask를 적용할지 여부이다.

Mask에 관한 내용은 후술하겠다. 

![image](https://github.com/user-attachments/assets/65747181-7411-4e6f-af84-38fe2ead5fa7)

수식을 보면 Q, K, V가 있을 것이다. 이것들이 attention의 핵심적인 역할을 한다.

Q는 Query라고 하며, 조사(질문)하고 싶은 대상이다.

K는 Key라고 하며, Query가 조사를 위해서 참고할 수 있는 자료라고 할 수 있다. 

V는 Value라고 하며, Key가 가지고 있는 정보(내용물)라고 볼 수 있다. 

예시를 들어보겠다. 

![vzvz](https://github.com/user-attachments/assets/5927691d-87bf-435f-83c0-b0cb62e4db67)

웅이는 지금 "Transformer"가 무엇인지 몰라서 누군가한테 배우고싶은 상황이다.
이때, "Transformer"가 Query의 역할을 한다. 

웅이 주변 사람중에 "김지우", "여수 독고", "강건마"한테 Transformer가 뭐야? 라고 물어보자.

지금은 "김지우", "여수 독고", "강건마" 각각이 Key의 역할을 하는 것이다.

질문을 들은 "김지우", "여수 독고", "강건마"는 자기 자신들이 갖고 있는 지식을 활용해서 답변을 내놓을 것이다. 

이때, "김지우", "여수 독고", "강건마"가 가지고 있는 지식들이 Value의 역할을 한다. 

이들은, 질문에 대해서 자신들이 가지고 있는 지식들중 **유사한 지식**들에 대해서 혼합한다.

이때, 각자 보유한 지식중에서 Transformer와 가장 유사한 지식들에 가중치를 높게 쳐서 혼한하여 대답을 줄 것이다.

"김지우": Transformer 그거 아가들 보는 영화 아니야?

"여수 독고": Transformer는 변압기를 말하는거지? V = IR이라는 공식이 있는데 ...

"강건마": Attention is all you need 아시는구나?! 

이들의 대답이 곧 Attention의 결과라고 볼 수 있다. 또한 알 수 있는 것이 "김지우"와 "여수 독고"의 대답은 도움되지 않는다. 

즉, Key에 관한 정보를 이상한 정보를 주면 model은 학습하는데 어려움을 겪을 수 있다는 것을 알 수 있다. 

## 1.2 Attention의 유사도는 어떻게 구현하지?

![내적](https://github.com/user-attachments/assets/ed2d8a73-d7fc-4b2f-a826-e35961372196)

Query와 Key, Value는 Attention 이전 Embedding Layer에서 나온 값을 Query, Key, Value 값으로 바꿔주는 연산이 있다.

수학적으로 표현하면 다음과 같다. E는 Embedding Layer에서 나온 값이라 하자.

Query = ${W_q  E}$ 

Key = ${W_k E}$ 

Value = ${W_v E}$ 

${W_q}$ , ${W_k}$, ${W_v}$는 학습 가능한 행렬이라고 볼 수 있다. 

Query가 Key를 통해서 어떠한 정보들이 나한테 중요(유사)한지를 확인하는 방법을 선형대수학의 내적을 통해 구현했다. 

위의 영상과 같이 Query와 Key의 transpose를 행렬곱 연산을 하면 Query와 Key의 유사도를 구하는 내적 연산을 할 수 있다.

정리하면, ${Q K^T}$의 결과는 Query가 Key와 얼마나 유사한가?를 연산한 결과이다.

## 1.3  Attention의 Scaled의 의미는 뭐지?

수식의 분모에 $\sqrt{d_k}$는 무엇일까? 값 자체는 Key가 가지는 hidden dimension 차원에 root를 씌운 것이다. 

아래 영상을 보면 $\sqrt{d_k}$을 나눠주는 효과를 알 수 있다. 

![scaled](https://github.com/user-attachments/assets/cbc3d277-3253-4ab2-9600-955112910b28)

내적 연산은 값을 서로 곱하고 더한다는 연산이기에 내적 결과 값이 큰 숫자를 가질 수 있다.  

우리는 Attention Score를 구하기 위해서 ${Q K^T}$의 결과에 SoftMax 함수를 통과시킬 것이다. 

SoftMax의 그래프를 확인해보자.

![ggredrg](https://github.com/user-attachments/assets/91a41ddd-5fe0-4728-9a3d-ec67c31736e3)

왼쪽은 수식과 같이 SoftMax에대한 그래프이고, 오른쪽은 SoftMax함수와 Softmax 함수의 미분 그래프이다. 

SoftMax에 들어가는 값이 엄청 작거나, 엄청 크면 오른쪽 미분 그래프에서 확인할 수 있듯이 기울기가 0으로간다.

즉, Gradient Vanishing 문제가 생길 수 있다. Scaled 과정은 Gradient Vanishing을 예방하기 위한 조치이다. 

만약 Query와 Key의 i번째 값은 평균이 0이고 분산이 1인 정규분포 ${N(0, 1)}$를 가진다고 해보자.

평균 

= ${E(Q_i K_i)}$ = ${E(Q_i)E(K_i)}$ = 0x0 = 0

분산

= ${Var(Q_i K_i)}$ = $E({Q_i^2 K_i^2})$ - $E({Q_i K_i})^2$ = $E(Q_i^2)E(K_i^2)$  

= $(Var(Q_i) + E(Q_i)^2)(Var(K_i) + E(K_i)^2)$ = 1

위의 결과에 의하여 아래의 수식으로 표현 가능하다.

$$\sum_{i=1}^{d_k} Q_i K_i  \sim {N(0, d_k)}$$

즉, 정규분포 ${N(0, 1)}$를 가지는 Query와 Key를 내적하면 평균이 0이고 분산이 ${d_K}$인 분포를 가진다.

여기다가 $\sqrt{d_k}$을 나눠준다면 ${N(0, 1)}$을 가지기에 상대적으로 SoftMax 미분 값이 0에 있을 확률이 줄어든다.

## 1.4 Attention Score가 뭘까?

![softmax](https://github.com/user-attachments/assets/6c578eb0-dee0-42dd-950c-f4593657c2b7)

지금까지 과정은 $\\frac{Q K^T}{\sqrt{d_k}}\$ 만 적용했다. 

이 값을 SoftMax 함수인 $\frac{\exp(x_i)}{\sum_{j} \exp(x_j)}\$ 여기에 적용해야한다. 

그러면 Query와 Key가 얼마나 유사한가에 대해서 0과 1 사이의 확률 가중치로 표현할 수 있게된다. 

Attention Score의 크기는 다음과 같이 표현된다.

$\text{Attention Score} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)\$

$\text{Query} \in \mathbb{R}^{query_{seq} \times D}$,  $\text{Key} \in \mathbb{R}^{key_{seq} \times D}$

그러므로 위의 Query와 Key를 내적하게 되면 아래의 차원을 갖게된다.

$QK^T \in \mathbb{R}^{query_{seq} \times key_{seq}}$

즉, Query의 각 seq 요소들이 Key의 전체 seq의 요소들과 얼마나 유사한지를 따진 Attention Score가 나온다.

## 1.5 Attention의 결과는?

![value](https://github.com/user-attachments/assets/032655fb-ee1f-4186-9f31-5cc85b85224c)

이 Attention Score를 Value와 곱해주면 최종적인 Attention 연산이 끝나게 된다. 

Query, Key가 얼마나 유사한지 Attention Score를 통해서 알 수 있고, Key 가지고 있는 정보를 유사한만큼 가져오는 과정이라 볼 수 있다. 

많이 유사할수록 더 많은 정보를 가져오고, 유사하지 않으면 적게 가져오는 것이다. 

Attention은 Query와 Key의 유사도를 구해서 Key가 가지고 있는 정보를 유사도 비율만큼 집중해서 새로운 정보를 만드는 과정이다. 

지금까지의 과정은 Single Head Attention이라 불리는 과정이다. 

## 1.6 Multi Head Attention이란?

## 2.1 Positional Encoding 왜 필요할까?

Transformer의 주장은 RNN이나 Conv와 같은 시계열 network를 사용하지 않아도 Attention만 사용해서 시계열 자료를 예측할 수 있다고 한다.

하지만 Section 1에서 봤듯이 Attention의 과정은 내적 연산으로만 이루어진다. 

내적은 입력 순서와 상관없이 항상 동일한 연산 결과를 내놓는다. 

(1, 2)와 (3, 4)를 내적한 결과나 (2, 1)와 (4, 3)을 내적한 결과는 같다. 

문장을 예시로 들어보자.

원본: The chef cooked the meal. 변형: The meal cooked the chef.

원본: The police arrested the thief. 변형: The thief arrested the police.

원본: The cat chased the mouse. 변형: The mouse chased the cat.

원본과 변형 문장의 단어는 아예 똑같지만 순서만 다르다. 하지만, 순서만 달라졌다고 의미는 아예 바뀌었다. 

내적은 원본 문장과 변형 문장이 서로 다른게 아닌 아예 똑같은 것이라고 인지한다.

순서를 반연하려고 RNN 계열을 쓰자니 long term dependency와 직렬 연산이라는 단점을 안고 가야하고,

그냥 진행하자니 순서를 몰라서 엉뚱한 결과를 초래할 수 있다.

그렇기에 임의로 순서 정보를 반영해주는 Positional Encoding이라는 개념을 도입했다. 

## 2.2 Positional Encoding의 조건은?

순서 정보만 주면 모든 문제가 해결될 것 같지만 생각만큼 쉽지는 않다. 

순서 정보를 주는 방법은 Embedding Layer의 결과에 순서 값을 더해주는 것이다.

![image](https://github.com/user-attachments/assets/b22f7646-76f1-407f-ae42-1d1cab426a90)

Embedding Layer 결과를 E, Positional Encoding의 값을 P라고 하면 위의 사진처럼 순서에 맞게 더해주는 것이다.

하지만, Positional Encoding의 값으로 무엇을 쓰냐가 문제이다. 

Positional Encoding을 위해서 다음의 조건이 필요하다. 아래 설명에 위치 차이라는 말은 시간 순서 차이라고 받아들이면 된다.

1. Positional Encoding의 값은 입력되는 시계열의 길이와 상관없이 고유한 값을 줘야한다.
   첫 번째 문장은 길이가 20줄이고, 두 번째 문장은 길이가 10줄이라고 가정하자.
   첫 번째 문장과 두 번째 문장의 10번 째 줄까지는 동일한 Positional Encoding 값이 들어가야한다. 

2. Positional Encoding의 값은 입력 값에 비해서 너무 크면 안된다.
   만약 embedding vector = [0, 1, 2, 3]인데 positional encoding = [1025, 502, -2145, -325]라고 하면,
   이 둘이 더할 때 원래 시계열의 의미보다 순서 정보의 의미가 더 커지게 된다. 입력 값이 왜곡된다고 볼 수 있다. 
   
3. Positional Encoding의 값은 빠르게 증가, 감소하면 안된다.
   Positional Encoding이 빠르게 증가된 시점에서 gradient vanishing or explosiong 같은 문제가 발생할 수 있다.

   ![fds](https://github.com/user-attachments/assets/ed992275-59bb-4daf-87b6-0759733f1050)

   위 그림의 계산그래프의 오차 역전파 과정을 보라. Transformer의 Encoder 과정만 직접 시각화했다.

   x는 어떠한 시계열 입력, W(x)는 Embedding Layer, E는 Embedding Layer의 결과, P.E는 Positional Endocing

   Z는 Embedding 결과에 순서 정보를 입힌 결과, F(Z)는 Multi Head Attention, G(K)는 Feed Forward 부분이다.

   빨강 부분으로 시각화한 부분을 보면, Positional Encoding의 값이 급격하게 증가하거나 감소하면 기울기가 0이나 무한대로 갈 수 있다.

   그렇기에 Positional Encoding의 값은 급격한 변화가 있어선 안된다. 

4. 위치 차이에 의한 Positional Encoding값의 차이를 거리로 이용할 수 있어야한다.
   
   예를 들어 0번째, 1번째 Positional Encoding 값의 차이가 1번째, 2번째 Positional Encoding값의 차이와 유사해야한다.
   
   이유는 위치 차이 1만큼 났을 때, Positional Encoding 또한 같은 거리 만큼 차이 나는 것을 인식할 수 있다.
   
   따라서 등간격으로 배열된 부드러운 곡선 형태를 사용한다.

5. Positional Encoding 값은 위치에 따라 서로 다른 값을 가져야 합니다.

   위치 정보를 나타내는 만큼 서로 다른 값을 나타내어야 학습할 때 의미 있게 사용할 수 있다.

위의 조건을 만족해야만 의미있는 Positional Encoding이라 할 수 있다. 

위치가 증가할수록(시간이 흐를 수록) Positional Encoding 값도 크게 설계해보자.

![image](https://github.com/user-attachments/assets/031b0557-199f-4829-9865-495abe8062eb)

문제는 Positional Encoding 값 입력 값에 비해 너무 크면 안된다는 것, Positional Encoding의 값이 빠르게 증가되면 안된다는 것에 위배된다.

그럼 값이 크고 빠르게 증가하는 것이 문제니, 정규화를 해서 넣어주면 해결될까?

![image](https://github.com/user-attachments/assets/f04448e8-cc41-411d-9122-9df6ef15f5e4)

Positional Encoding의 값은 입력되는 시계열의 길이와 상관없이 고유한 값을 줘야한다는 조건에 맞지 않게된다. 

입력 데이터의 길이에 따라 정규화 해주는 값이 (max_len -1)로 정해지기 때문에 고정된 위치값을 가질 수 없다.

Positional Encoding의 값을 scalar 형태가 아닌 vector 형태로 바꿔본다면?

![image](https://github.com/user-attachments/assets/de017d19-167e-4387-978b-340f61c3ad39)

이 그림에서는 이진수 형태($2^d$)로  Positional Encoding을 표현했다. d는 hidden dimension 차원이다.

$2^d$ 보다 작은 모든 길이에 대하여 동일한 Positional Encoding 값을 가질 수 있고 가변적인 길이에도 같은 위치라도 동일한 값을 가질 수 있다.

또한 값도 0~1 사이라 매우 적절해 보인다. 

![image](https://github.com/user-attachments/assets/f5985bdb-670f-46cc-b945-a4685ef1c66c)

문제는 파란선과 검은선의 차이는 파란선은 점들 사이의 간격이 일정하지만 검은 점들 간 간격은 일정하지 않다.

파란선은 y = x이다. (0, 0), (0.33, 0.33), (0.66, 0.66), (1, 1)이라하자. 

검은선은 2차원 좌표에 대한 이진수이다. (0, 0), (0, 1), (1, 0), (1, 1)이라하자.

즉, 파란선은 입력 데이터의 길이에 따라 정규화 해준 값이고, 검은선은 지금 다루고 있는 이진수 기법이다.

문제는 "위치 차이에 의한 Positional Encoding값의 차이를 거리로 이용할 수 있어야한다"를 위반한다.

이진수 좌표로 거리를 구해보자

(0, 0) -> (0, 1) = 1

(0, 1) -> (1, 0) = $\sqrt{2}$

(1, 0) -> (1, 1) = 1

(0, 0) -> (0, 1)의 거리 차이와 (0, 1) -> (1, 0)는 동일하지 않다. 지금은 작아보여도 차원이 커질수록 문제가된다.

그렇기에 다른 해결 방안이 필요하다. 

## 2.3 Sinusodal Positional Encoding


## 3.1 Padding Mask

## 4.1 Multi Head Attention 

## 4.2 Self Attention

## 4.3 Cross Attention

## 5.1 Point Wise FFN


## 4.1 Attention의 문제점

---
### DECODER

![gwsrgr](https://github.com/user-attachments/assets/2e99cf6b-a4ed-43ab-ab24-7ed26532029f)

#### Masked Attention
---
### TRANSFORMER의 한계는?
