# CoCoFL: Communication- and Computation-Aware Federated Learning via Partial NN Freezing and Quantization

## Abstract

Synchronous FL은 모든 디바이스의 훈련은 서버에서 제공한 동일한 deadline에 마춰서 끝나야함.

제한된 디바이스에서 sota모델의 일부분만을 학습하여 사용하는 것은 학습을 비효율적으로 하고 모델을 좋게 만드는 것을 방해한다.

게다가 이런 방식은 각 디바이스의 정확도의 불공정성을 유발하여 문제가 있다. 왜냐면 큰 디바이스는 전체 모델의 대부분은 가지고 훈련하고 전체 모델 업데이트에 더욱 많이 기여를 하기 때문에 공정성이 떨어진다. 특히 다양한 디바이스에 걸처 class label의 분포가 치우쳐 있을 경우 = non iid Ex) 하나의 device의 훈련셈플에는 class1이 유독 많다, 디바이스1이랑 2의 정보에 의존성

CoCoFL은 모든 디바이스에서 full NN을 유지한채로 학습한다. 각 디바이스의 리소스 차이를 반영하기 위해, 특정 층을 freezes하고 quantizes한다. 이외의 층은 일반적인 학습을 따른다.

이러면 모든 제한된 로컬 디바이스가 FL system에 크게 기여하고 리소스를 잘 활용 가능하다. fairness가 보존된다. 


## Introduction

하드웨어의 발전에 의해 열악한 디바이스에서 조차 학습이 가능해졌다. 그래서 FL이 등장하였고, 이는 개인정보에 대한 보증에서 큰 이점이 있다.

기존에 등장한 FL model들은 각 로컬 디바이스의 리소스, 계산능력, 통신 한계등을 고려하여, 각 디바이스들에게 축소된 모델 즉, 글로벌 모델의 일부를 부여하여 각 디바이스에서 학습을 하도록 했다.

여태 나온 FL model의 자세한 구조는 다를지 모르지만, 일반적인 구조는 위에 말한 바를 같이 디바이스에 적합한 축소 모델을 뿌리는 것이다. 비록 이 구조는 모든 디바이스가 훈련에 참가하게 하지만, 그들의 데이터를 효과적으로 학습하지 못한다. 특히 non-iid dataset에 대해 치명적이다.

우리가 실험해본 결과 기존의 모델축소 FL에서 non iid dataset을 가지고 훈련할때, 정확도가 굉장히 낮고, 간단히 몇개의 디바이스(문제가 있는)를 제외하고 학습하니깐 더 좋은 성능을 내더라. 
왜냐 제한된 디바이스는 전체 모델의 업데이트를 안 좋은 쪽으로 끌고 가기 때문에(학습을 효율적으로 하지 못해서) 그것을 없애니 좋더라.

CoCoFL은 각각의 모든 디바이스가 그들의 리소스와 관계 없이 전체 모델에 대해 gradient를 구함. train full model that apply partial freezing and quantization

Freezing and quantization는 필요하는 리소스를 줄이고, 효과적으로 학습한다.

Freezing : gradient구하는 계산량 감소, activation 저장량 감소, 업데이트할 파라미터수 감소
quantization : freezen layer 에서의 계산속도 향상

따라서 우리는 디바이스의 리소스에 따라 모델의 크기를 건드리는게 아니고, 얼마나 Freezing and quantization 할지를 결정하는 것이다.

CoCoFL은 서버와 독립적으로 런타임에 로컬디바이스의 가용 리소스에 따라 “훈련된 레이어”를 선택한다

요약해서 우리가 보여줄 것은 총 4가지이다

1. 기존의 subset based FL은 안 좋다. 단순히 학습에 방해되는 몇가지 디바이스를 빼는 것 만으로 성능이 향상된다.
2. 기존의 subset-based FL에 비해 CoCoFL은 더 좋은 성능 (모든 디바이스가 모두 비슷한 중요도로 업데이트에 기여 = 공정성,  높은 정확도)를 보여줌
3. 제한된 디바이스에서도 전체 모델을 가지고 학습이 가능케 하는 Freezing and quantization 의 방식
4. CoCoFL에 전반적인 과정을 보여줌


## System Model and Problem Definition

System model :


조정을 책임지는 서버, 클라이언트로서의 디바이스 이렇게 구성된다.
각 디바이스는 로컬데이터에 독점 접근을 한다, 당연,,, 로컬 데이터는 로컬 디바이스에만 있응께

훈련은 각 동기적 라운드마다 주기적으로 FL 방식으로 이뤄진다, 각 라운드마다 디바이스 서브셋을 만든다, 선택된 디바이스는 가장 최근 모델의 파라미터 모두를 서버로부터 다운받음, 로컬 데이터로 로컬 디바이스에서 글로벌 모델을 업데이트, 각 디바이스는 업데이트 모델을 서버에 업로드, 서버는 FedArg를 수행.

서버는 늦게 도착한 업데이트 정보를 버림. 즉 제시간에 항상 업데이트가 이뤄져야함


Device model :


디바이스의 성능은 하드웨어, 소프트웨어 그리고 훈련상황에 따라 결정된다.
훈련중 필요한 메모리는 소프트웨어, 훈련 상황에 따라 갈린다.

A(r,c) : subset of training layers in round r on device c

A : whole configuration in model

Tc : training time in c device

Mc : require memory in c device

디바이스 하드웨어에서 CoCoFL 방식으로 profiling을 해서(분석해서) Tc, Mc를 구할 것이다.

서버에서 클라이언트의 전송은 서버의 높은 전송력으로 커버됨으로 무시. 
하지만 클라이언트에서 서버의 전송은 여러가지에 영향을 받음.

S(r,c) : communication constrain, in round r on device c

그래서 우리는 통신제약을 클라이언트에서 서버로의 업로드가능한 비트의 수의 최대치로 정한다. 서버 -> 클라이언트는 무시

라운드 r, 디바이스 c에서 포함되지 않는 레이어는 모두 동결, 양자화된다. 이는 안 바뀌므로 업로드할 필요가 없다.
디바이스에서 서버에게 업로드 해야하는 파라미터의 크기를 해석적으로 유도하거나, 각 레이어 당 매개변수를 계산해 얻는다.

그리고 공정성 또한 측정을 해 볼 것이다.


## Partial Freezing and Quantization

우리는 일반적인 구조를 가지고 CoCoFL을 적용할 것이다. Convolutional layer, Batch Norm layer, ReLU layer를 하나의 블럭으로 볼것이다.
그리고 이 블럭이 훈련시간의 대부분을 차지한다.

블럭이 동결, 양자화 되는 최소의 단위임.

Freezing : 

block에는 3가지 종류가 있다.
1. Frozen block : trained block의 prediction을 얻기 위해 forward pass만을 수행한다. 이 블럭은 업데이트가 필요 없기에 backprop가 필요 없다
2. Frozen block with backward pass : 이 블럭은 이전 레이어에 훈련할 것이 껴 있어, backprop가 필요한 frozen block이다.
3. Trained block : 훈련해야하기 때문에 forward, backward pass 모두 필요하다.

이렇게 frozen을 사용하면 기존 상황보다 계산량이 줄어들고, 훈련시간이 줄어들고 필요없는 데이터를 저장을 안하니 메모리 사용또한 줄어든다.


Fusion :

Frozen block에서 BN, convolution operation을 합치겠다.
합칠 수 있는 이유는 동결된 블럭(레이어)의 값이 훈련도중에 절때 변할 일이 없으므로 이를 선형 함수로 구하는 것에 문제가 없다.
1.3 번 종류의 블럭은 동결됐기 때문에, forward pass에서 값을 구하는 과정의 연산수를 줄이기 위해 이와 같은 것이 가능하다.

Quantization :

우리는 frozen된 블럭에 한에서만 양자화 하겠다.
1,3번 타입의 forward pass의 값을 양자화, 3번의 trained block을 위한 intermediate gradient 역시 양자화. Float32 -> int8
그러면 작은 메모리, 적은 훈련시간이 든다.

당연히 양자화를 하면 이로 인한 정보 누락이 생기고 이로 인해 quantization noise 가 발생된다.  그리고 초기 라운드에 frozen, fusion 블락에서 에러가 발생. 하지만 이 손실보다 얻는 이득이 더 크다.


## Overall CoCoFL Algorithm

부분적인 동결, 양자화는 훈련에 필요한 소통, 계산, 메모리 리소스량을 조정할 수 있따.
CoCoFL에서는 디바이스의 가용 리소스에 맞는 training configuration(= 일부가 동결 + 양자화 된 모델)을 선택한다

디바이스는 그들의 가용 리소스를 기반으로 훈련 가능한 무작위 configuration을 고른다.

각 디바이스가 자신의 리소스의 맞는 configuration을 고르게 할라면 각 configuration마다 필요한 리소스의 량을 알아야함
근데 n개의 블락 모델이라면 총 2^n개의 configuration이 존재. 너무 이를 designed- time에 측정하기에는 너무 오랜 시간이 걸림.
그래서 우리가 사용하는 configuration을 하나의 연속적인 train block만이 존재하는 configuration으로만 제한 한다. 이러면 구하기 훨씬 쉬워진다.
디바이스는 각 라운드마다 자신의 리소스에 돌아갈 수 있는 모든 configuration subset을 생성, 그중에서 가장 리소스를 잘 활용해 큰 정확도를 내는 것만을 다시 추림. 거기서 랜덤하게 선택


## summary

기존 FL의 문제 : 모든 디바이스 공정하게 학습에 참여하지 못한다(= 디바이스마다 학습에 사용하는 모델 사이즈가 작다) => 효율적으로 학습하지 못한다(증명 : 일부 디바이스를 제거하니 성능이 향상되더라), non iid에서 치명적

1. 기존 FL에 비해 더 좋은 성능,  공평성 보장 => 모든 디바이스가 full-precision으로 학습  동결됐지만… 어쨌든 전부쓴다!
2. Non iid에서 더 좋음 ?? 모르겟음
3. Fusion CL and BN
4. Quantization




기존 FL는 디바이스가 각자의 최대 리소스를 활용할 수 있는 축소 모델을 서버로 부터 받아 이를 훈련, 업로드함
서버는 이제 모든 클라이언트에서 온 update model을 아주 공정하게 반영하지 않고, 차등적으로 scale factor를 부여해 글로벌 모델 업데이트.
이는 좋지 않다. 일단 각 디바이스별 fairness하다, 각 클라이언트가 효율적으로 학습하지 못한다.
2번째 이유는, 실험을 통해 입증했다. 어떤 실험? 기본 FL method에서 constrain device를 업데이트에 제거하는 것만으로 성능이 올라가더라  => 몇몇 클라이언트는 전체 업데이트에 해를 가하는 구나 => 안 좋게 학습했구나.

그래서 CoCoFL 제안. 이 방식은 디바이스의 리소스와 상관없이 모든 디바이스가 full precision model train
여기서 리소스 별로 조정하는 것이 디바이스가 다운로드할 모델의 사이즈가 아니라 각 디바이스가 학습을 할 모델의 freezing 정도이다.
동결된 블락은 학습과정에서 업데이트 되지 않음. 그러니 동결된 블락에 필요했던 데이터가 많이 줄어든다. => 전체적인 학습속도 역시 빠르다

일단 블락이란 모델에서 연속적으로 나타나는 레이어를 다시 한번 묶은 단위이다. 여기서는 resnet. Moilenet, densenet에서 실험했고 하나의 블락에는 CL, BN, ReLU가 포함되어 있다. 이 블락이 동결의 최소 단위이다.

이제 동결된 레이어의 2가지 방식을 적용한다. Fusion, Quantization

Fusion : 동결된 블락의 내부 정보는 학습도중 절때 바뀌지 않는다. 그러므로 이를 통과하는 과정을 선형식으로 한번에 묶어서 처리하자.
Quantization : 동결된 블락의 정보를 양자화 float > int하자 그러면 계산량, 속도, 메모리 다 줄어든다.

block의 종류는 총 3가지 frozen, frozen with backless, train

