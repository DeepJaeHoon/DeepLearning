# Attention Is All You Need
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
### ENCODER

![제목 없음fsdfsd](https://github.com/user-attachments/assets/d072ceb6-5e7c-474e-8453-0368b7f86de5)

Transformer의 Encoder 구조는 위와 같은 구조를 가진다. 

어떠한 시계열 정보가 Input Embedding Layer에서 Toeken화가 진행된다. 

어떠한 문장이 주어지면 word2vec으로 model이 이해할 수 있는 언어로 바꿔주거나 각 시계열 step마다 가진 변수들을 추상화하는 역할이다. 

정리하면. 문장이나 어떠한 시계열(주식, 주행 경로)을 model이 이해할 수 있는 언어로 바꿔주는 단계이다. 

Positional Encoding은 Embedding Layer의 결과에 시간 순서 정보를 추가해주는 역할이다.

그 이후, Multi Head Attention을 통해서 .......................................

#### Attention

![image](https://github.com/user-attachments/assets/298ad028-44e6-4dbb-9bf1-7e18f892a6cf)

Attention은 무엇인가? 

Attention은 단어 의미 그대로 model이 누구한테 더 집중해서 참고를 해야하나 지시하는 역할이다. 

만약 Transformer를 공부하고자 하는 학생이 Transformer와 관련된 내용을 집중하지, 생명과학II에 집중하지 않듯이 말이다.

그럼 model이 무언가에 집중하라는 개념을 어떤 방법으로 구현을 했을까?

대표적이고 주로 쓰이는 방식이 아래의 scaled dot product attention이다.

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

이들은, 질문에 대해서 자신들이 가지고 있는 지식들중 **유사한 지식**들에 대해서 혼합해서 다음과 같은 결과를 준다.

"김지우": Transformer 그거 아가들 보는 영화 아니야?

"여수 독고": Transformer는 변압기를 말하는거지? V = IR이라는 공식이 있는데 ...

"강건마": Attention is all you need 아시는구나?! 

이들의 대답이 곧 Attention의 결과라고 볼 수 있다. 또한 알 수 있는 것이 "맹헤지"와 "여수 독고"의 대답은 도움되지 않는다. 

즉, Key에 관한 정보를 이상한 정보를 주면 model은 학습하는데 어려움을 겪을 수 있다는 것을 알 수 있다. 

![내적ㄹ](https://github.com/user-attachments/assets/ff653ea5-68b5-4649-a4e1-2430f722f265)


Query가 Key를 통해서 어떠한 정보들이 나한테 중요한지를 확인하는 방법을 선형대수학의 내적을 통해 구현했다. 

#### Padding Mask

#### Attention의 문제점

#### Positional Encoding

#### Pading Mask
---
### DECODER

![gwsrgr](https://github.com/user-attachments/assets/2e99cf6b-a4ed-43ab-ab24-7ed26532029f)

#### Masked Attention
---
### TRANSFORMER의 한계는?
