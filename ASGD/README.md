# Asynchronous-stochastic-gradient-descent

<br/><br/>

## Abstract

Dnn을 활용한 speech recognition model이 기존의 GMM-HMM모델에 비해 좋은 preformance를 거둠 


하지만 DNN의 엄청난 양의 파라미터로 인해 training cost가 크다 > 오래걸린다.

게다가 병렬화도 어려운 상황이다. > SGD의 빈번한 모델 업데이트, 모델 업데이트의 의존성

그리하여 ASGD를 제시 > 병렬적으로 gradient 계산후, 비동기적으로 모델 업데이트

<br/><br/>

## 1. Introduction

dnn을 활용한 speech recognition이 SOTA를 이뤘다.
<br/><br/>
DNN layer의 초기화는 기존 BP algorithm 기반의 unsupervised pre-train 한 것을 가지고 layer by layer로 진행한다.
하지만 좋은 performance에도 단점은 존재하는데, 바로 DNN 구조에서 발생하는 많은 파라미터이다.
그로인해 자연스럽게 training cost가 증가한다.


<br/><br/>
기존의 SOTA였던 GMM-HMM은 동기적 요소가 존재하지 않고, 각 프로세스가 독립적이라  parallelism이 자유롭다, 
하지만 DNN은 구조 특성상 parallelism이 어렵다.
그 이유는 model update의 동기적 요소 ,sequential update(minibatch1 > 2 > 3) ,때문이다.
직접적인 training data parallelism도 좋은 결과를 얻지 못함.
<br/><br/>

> ASGD는 동기적 요소를 일정수준 배재하고 training cost를 얻는 것에 focus한다.

동기적인 요소를 배재해, 일정수준의 parallelism을 얻고 이를 통해 속도 향상을 꾀한다.

SGD > sequential process : forwardness, backdrop 
<br/><br/>
sum up in server with computed gradient from each GPU. If each GPU has different training speed, wait until entire GPU data are arrvied => synchronous cost가 높다

<br/><br/>

ASGD > Calculate gradient : parallel    |        update model : Asynchronously


<br/>

BP에서 batch size는 performance, efficiency 관점에서 중요한 요소이다. (BP가 minibatch based임)
각 GPU, CPU에서의 communication cost가 크다.


<br/><br/>

### 2.1 Deep neural network used in speech recognition

DNN architecture는 파라미터가 엄청 많은 구조이다!


<br/><br/>

## 2.2 stochastic gradient descent(SGD)

> GD : 모든 데이터를 가지고 gradient 를 계산해 sum up loss를 가지고 모델을 업데이트
<br/><br/>

사용하는 용량이 너무큼, shooting이 없음

> SGD : 데이터의 부분(minibatch)를 가지고 수행
<br/><br/>

Shooting, data redundancy를 효율적으로 처리


<br/><br/>

## 2.3 Analysis of minibatch size

Training cost를 줄이기 위해 직관적인 방법은 minibatch size를 늘리는것 > 하지만 잘 작동하지 않음
<br/><br/>
일부에서는 minibatch size가 커졌는데도 training cost가 증가하는 부분이 있음 (특수한 경우)
<br/><br/>
그리고 minibatch size는 GPU의 성능에 의해 상향선이 정해진다.
<br/><br/>
GPU의 성능을 넘어선 batch size는 안하는것만 못함 > little speed up , degrade performance significantly

<br/><br/>

## 3 Asynchronous SGD

SGD variation algorithm
Server(cpu)-client(gpu) architecture
<br/><br/>

“서버에서 데이터, 모델을 받을 때”, “서버에 모델을 업데이트 할때” Mutex를 사용해 업데이트 충돌을 방지
<br/><br/>

GPU를 client처럼 사용
<br/><br/>

이전에는 large minibatch size가 효과가 없다고 했는데, ASGD에서는 성능 향상을 보인다.

<br/><br/>

SGD는 병렬적으로 처리된 각 GPU grdient를 통합해야하므로, 모든 GPU의 전송을 기다려야 한다. Synchronous cost
하지만 ASGD는 이를 없앤다.

<br/><br/>

그로인해 training cost가 줄어든다.

<br/><br/>

Communication cost는 큰 변화 없다. 그래서 여전히 bandwidth 의 한계에 부딛힌다.

<br/><br/>

## 4. Experiments

ASGD를 가지고 HMM-DNN을 학습할 때, 초기에는 작은 배치로 부터 시작해 진행 할 수록 배치사이즈가 커지게끔 훈련함
<br/><br/>

ASGD는 large minibatch에서 강하고, small minibatch에서는 더 약하해진다. Small minibatch는 잦은 모델 업데이트 때문에 그러함. > communication cost때문에, ASGD의 문제가 아니라 small minibatch 자체의 문제임
<br/><br/>
Small minibatch는 학습 초기에 조금만 사용되므로, 실험에서 small minibatch의 영향력은 적을 것이다.
<br/><br/>

ASGD가 통신 비용을 줄여주지만 여전히 bandwidth가 bottleneck이다.

<br/><br/>


## 요약

Large minibatch로 업데이트 횟수 자체를 줄여 통신 비용을 감소시킴.
<br/><br/>
ASGD는 모든 미니배치에 대해 서버의 클라이언트 모델만 업데이트 시키고, 일정량의 그레디언트가 쌓이면 서버 모델을 업데이트함 > 업데이트 수 줄음 > communication cost 줄음 > training cost 줄음
<br/><br/>



ASGD의 단점 : 여러 버전의 모델을 가지고 집계해 업데이트함, accurancy 가 굉장히 낮음, staleness : 오래되어 이상한
<br/><br/>

> ex ) GPU1 : model1 , GPU2 : model3  | 각 번호는 미니배치가 몇번 적용된 모델인지를 나타냄.
gradient를 집계한다는 것은 동일한 버전하에 의미가 있다. 극단적인 경우 gradient를 집계 시 0으로 수렴되는 상황이 나타날 수 있다. 동일한 버전에서는 서로의 방향이 비슷한 경우가 대부분임. 하지만 버전이 다르면 방향이 정반대일 수 있음
<br/><br/>



여전히 communication cost에 대한 문제 존재
<br/><br/>

Distribute leaning 에서 크게 communication cost, synchronization의 두 관점에서 평가한다. ASGD는 synchronization에 대해 집중한 알고리즘으로 이에 대해 성능 향상도 애매하고, 여러가지 문제를 해결 못한 모델이다.
Speed up도 애매해, accurancy도 애매해 그래서 현재는 버려진 알고리즘이다.

16 GPUs 16 times speed up little decrease accurancy
16 GPUs 8 times speed up increase or maintain accurancy
이런 것이 좋은 알고리즘으로 평가 받는다

