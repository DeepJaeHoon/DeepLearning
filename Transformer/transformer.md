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

거리에 의존적이라는 의미는 0초~10초 자료를 RNN을 통해서 계산하면, 10초에서 0초의 feature는 7초에서의 0초 feature보다 흐릿하게 남아있다.

시간(순서)상 먼 거리일수록 정보가 흐릿해지는 것을 의미한다. Bidirectional은 반대로 10초에서 시작해서 0초로 RNN 통과하는 방식이다.

이 역시, 0초에서 10초의 feature는 7초에서의 10초 feature보다 흐릿하게 남아있다.

![image](https://github.com/user-attachments/assets/d8c687d1-7bc0-4444-8dfb-4b0225b94a8c)

Bidirectional 방식으로 위의 0초에서 10초, 10초에서 0초 feature를 concat을 해도 첫 입력정보가 갈수록 흐려지는 거리 의존성 문제는 남아있다. 

또한, RNN 계열의 방식은 다음 시계열의 정보를 알기위해서 현재와 과거의 정보에대해서 연산해주는 과정이 필연적이다.

이러한 연산 방식은 직렬 방식이라 병렬로 해결할 수 없어서 연산 속도가 느리다는 단점이 남아있다.

RNN 계열을 사용하는 한, 병렬 문제, 거리 의존성과 기울기 소실, 과거 정보 소실 문제는 남아있기에 RNN 계열에서 벗어날 방식을 제안한게 Transformer이다. 

Transformer는 RNN, CNN을 아예 사용하지 않고 Attention만으로 시계열 정보를 효과적이게 처리했다.

---
### ENCODER
---
### DECODER
---
### TRANSFORMER의 한계는?
