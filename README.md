# FEDHM : Efficient Federated Learning for Heterogeneous Model via Low-rank Factorization

## Abstract

FL할때, 각각의 디바이스가 같은 모델 구조, 크기를 사용함, 이는 디바이스의 특성을 무시하고 같기 때문에 성능저하로 이어짐.  
각 클라이언트(디바이스)의 차이를 무시하지 말아야 좋은 성능을 기대할 수 있다.  
<br>
그래서 우리는 heterogeneous low-rank model을 클라이언트에 뿌리고, 결과를 full-rank model로 집계하는 방법을 소개한다.
<br>
계산복잡성이 다양한 모델을 로컬에서 훈련후, 글로벌 모델에서 집계  
이 기술은 communication cost를 줄인다. 각 클라이언트의 resource차이를 고려한 모델을 분배  
<br>

## Introduction  

FL : 로컬 모델이 각자의 데이터 공유 없이 사용해 모델를 훈련시키고, 그 결과 만을 중앙에서 집계하는 학습방법  

각 클라이언트가 종류가 다른 디바이스 라면 resource 차이가 생긴다. - 훈련,통신의 속도,용량 차이  

각 클라이언트 환경에 맞는 서로 다른 모델을 분포하자(각 디바이스의 리소스를 최대한 활용가능하게 커스터마이징한 모델).
이를 HeteroFL model이라 칭한다.  
클라이언트의 상황을 고려한 서로 다른 모델을 분포했기 때문에 heterogenity about local device resource 에 대한 해소가 있지만 model heterogeneity문제가 추가 발생한다. 
e.g. convergence stability, communication overheads, and model aggregation  

기존의 존재했던, HeteroFL은 pruning 하고 채널별로 집계를 한다.  

> Structured pruning (channel pruning) : 채널(층)을 뽑아내 날려버림(파라미터를 0으로 또는 아예 삭제), 층 자체를 날리므로 pruning 비율을 크게 못함, 층자체를 날림 = 메트릭스가 줄어듬 > inference속도 개선


> Untructured pruning : 의미 없는 노드들을 뽑아내 날려버림(파라미터를 0으로 또는 아예 삭제),  pruning 비율을 높게 설정가능, 층은 아직 살아있고 중간중간 0 이 채워짐 = 매트릭스는 안줄어듬 > inference속도 유지


  
기존 heteroFL의 문제는  
1. 가지가 많이쳐진 작은 로컬모델은 훈련을 잘 못한다
2. 작은 모델의 정보가 무시될 수 있다, 큰 모델의 로컬데이터가 종속되고 균일하지 않은 분포일 경우, 큰 모델이 대해 과적합 문제가 발생한다. 그러면 중앙 모델은 로컬 큰 모델에 과적합 될 수 있다.

이의 해결방안으로 Split—Mix 방식도 제안되었지만 성능이 좋지는 않음, 앙상블 기법을 통한 under training(Over fit) 해소 하려 했지만, 성능이 많이 떨어짐  
  
기존의 pruning 방식, split-Mix를 사용한 두 FL model은 좋은 성능을 발휘하지 못했다.  


그래서 federated model compression mechanism in FL, FEDHM을 제안한다.  
FEDHM은 큰 모델로 부터 각 디바이스에 맞게 분해, low-rank factorization한 작은 모델로 압축하는 기술이다.  
1. Factorize global model(DNN) to heterogeneous local model, fitting each device capacity and property
2. Receive heterogeneous local model from selected client and gathered local data transfer to full-rank model

Low-rank > full-rank ,  full-rank > low-rank  둘다 글로벌 모델에서 일어남  


Row-rank factorization을 시도하는 첫 논문임.  
  
FEDHM consists of three main components, i.e., (1) local factorized training, (2) model shape alignment, and (3) model aggregation.
<br>

## 2. Federated Learning on Heterogeneous Devices

<br>

### 2.1 overview of FEDHM

FEDHM의 순서는 다음과 같다.

1. Factorize global model to heterogeneous local model. And each local model’s dimension is different. It suit in each heterogeneous local device resource making them fully using each resource. 
2. Each local device receive factorized local mode and train with own data independently
3. After local training, global model receive local model from local device. And then transfer to full-rank model and aggregate it.

<br>

### 2.2 Local Factorized Training
<br>

#### 2.2.1 Convolution Layer Factorization


Convolution layer를 지난후에 (m,W,H) > (m,n,k,k)  이런식으로 변한다. 즉 파라미터 수가 m*n*k*k개 이다.

이 4D를 2D으로 unrolling 한 후에 factorization을 진행한다. Cost 줄음 (m,n,k,k) > (mk,nk)
 unrolling이 끝났으면 reshaping, truncated SVD을 이용하여 분해를 진행하여 2개의 매트릭스, 즉 2개의 작은 CNN이 완성된다.
과정 (m,n,k,k) > (mk,nk) > (mk,r),  (r,nk) >(m,r,k,1) , (r,n,1,k)로 바뀐다
Unrolling >  truncated SVD   > reshaping  순으로 진행
4D를 unrolling 하지 않고 바로 인수분해 때리는 접근은 unrolling을 하는 우리 방법보다 좋지 않았다.

Decompose, composite은 따로 추가 공부가 필요함. 지금은 그냥 그렇다고 하고 넘어가자

<br>

#### 2.2.2 Hybrid Network Architecture

인수분해 네트워크는 “approximation”이라는 것을 기억하자. 반드시 vanilla에 비해 오류가 나올 수 밖에 없다.
Approximation error는 모든 층에서 나타나는데, 처음에 발생한 오류가 전파 된 것이다. 
간단한 해결책 : 뒷쪽 레이어만 인수분해하자.

CNN은 뒷쪽의 파라미터가 총 파라미터의 절반 이상을 차지한다. (CNN의 특수성) 그리서 뒤의 레이어가 값을 결정하는데 더 큰 영향이 있다.

따라서 뒷쪽만 인수분해 해도 압축률을 잃지 않는다 ??? 무슨 뜻?

앞쪽은 기존의 W를 그대로 사용 뒷쪽은 인수분해된 U,V를 사용해 나타냄

W = {w1,w2,…,wn,    Vn+1,U+1,Vn+2,Un+2,……,Vm,Um}
n층까진 그대로, n+1부터 마지막층 까지는 인수분해
N은 하이퍼파라미터로 초기에 설정해야함.

Initialization and Regularization of Factorized Layers 논문에서 다룬 FL에서 좋게 작용하는 테크닉인 Frobenius Decay와 spectral initialization을 채택하여 사용함

<br>

#### 2.2.3 Initialization and Regularization of Factorized Layers

spectral initialization (SI) and regularizing with Frobenius decay (FD)로 low-rank model을 초기화하면 향상된 결과를  얻을 수 있다.

<br>

### 2.3 Model Shape Alignment and Aggregation

각 클라이언트의 factorized local model을 가지고 rank(행렬의 기저 수) ratio와 softmax temperature을 이용한 softmax를 통해 global model을 업데이트 시킨다.

<br>

## 3. Theoretical Analysis
<br>

### Computational Complexity and Model Size

기존의 FC, Conv에 low-rank factorization을 수행한 결과 파라미터수가 줄어들었다.

기존에 존재했던 다른 방식의 factorization과도 비교를 해본결과, 입력차원이 출력차원보다 많이 작지 않은 이상 우리 것이 model size, computational complexity가 낮았다.

통신 비용과 파라미터수는 비례하다. 그러므로 파라미터수가 낮아졌으므로 통신 비용또한 줄었다.

통신, 계산 비용 모두 줄음

<br>

### Convergence Analysis

2개의 가정을 가지고 수렴을 분석한다.
1. 사용하는 loss function은 연속적인 미분가능한 Lipschitz continuous gradients 을 사용, 그리고 하한선 존재.
SG의 모멘텀도 특정 상수에 하한이 있다.

2. 분해된 U,V 역시 norm에 대해 제한이 되고, 집계 라운드 마다 full-rank, low-rank model의 차이 역시 특정 norm에 제안 된다.  

이 2가지 가정을 통해 분석을 완성함.

<br>

## summary

1. FL에 문제가 있다 : heterogeneity , 이를 해결하기위해 heteroFL
2. 기존 pruning, split-mix에 이런 해결 못한 문제가 있다.
3. Low-rank factorization을 사용한 FEDDM을 제안. 
4. 주요 기술 : low-rank factorization, hybrid architecture, 기존의 regulatiuon+ initalization 채용
5. 순서 설명 + 집계 방식 끝

<br>

단점
1. Decompose(factorizate), composite(aggregate) 비용이 너무 크다. 큰 모델을 분해하는 일은 작은 일이 아니다.
2. Approxiamation 오류가 너무 크다. Because of SVD and hybrid architecture

