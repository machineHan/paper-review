# Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour

## Abstract

Distributed synchronous SGD : 그냥 미니배치를 각 GPU에 나눠서 gradient를 계산해 aggregate(sum up, average) 해서 전체 모델 업데이트  
이것이 효과적이기 위해서는 각 GPU에 할당된 sub-minibatch size 역시 커야함 = Large minibatch  

Minibatch size를 키워보겠다!  
단순히 minibatch size만 키웠더니 > validation error가 높아지더라 > 그래서 linear scale rule을 적용했더니, acuurancy의 손실없이 validation error가 줄더라  
훈련초기 문제가 있어 warm up phase도 추가  
<br>


## 1. Introduction

최근 DNN관련 모델의 크기, dataset의 size가 너무 크다.
그리고 사이즈 증가가 pre-training을 사용하는 모든 분야에 미치는 긍정적 영향이 여러 실험들에서 증명 되었다.
그에 따라 자연히 훈련관련 코스트가 증가함.  

그래서 이 논문은 scaling된 large minibatch with SGD의 효용성을 입증할 것.  

Synchronous SGD는 흔해졌지만, 우리처럼 8k개의 데이터를 accurancy를 유지하거나, 빠르게 처리한 결과가 없다.  
Linear scale rule은 기존에 있었던 기술이지만 조명되지 않았었다. 그리고 warm up phase를 추가해 훈련초기의 장애를 극복했다.  

아까 말했다 싶이, batch size가 커질수록 optimization이 줄어든다. > 전역최솟값을 잘 못 또는 느리게 찾아간다.  

그래서 이 논문에서 large size에서도 optimization을 유지하고싶어함  
<br>


## 2. Large Minibatch SGD

<br>

### 2.1 Learning Rates for Large Minibatches

Small minibatch 대신 large minibatch를 사용해 accurancy감소 없이 speed up을 성공해내겠다.  

> Linear scaling rule : k배의 minibatch사용시, learning rate에도 k를 곱한다

그외의 파라미터는 냅둠.
이로 인해 배치크기가 커져도 accurancy, training curve가 비슷해짐  

Small minibatch size : n  vs   Large minibatch size : kn  을 비교하는 해서 자신의 접근법을 설명함  
η = ηˆ * k 이면 배치 사이즈가 달라도 가중치의 결과값이 비슷하다는 것을 대충 유추가능함  

그래서 large minibatch의 학습률 small eta에 k배 즉 linear scaling을 하는것이다.  

∇l(x, Wt) ≈ ∇l(x, Wt+j ) 이 가정하에 정해지는 위의 내용이지만 전제가 맞지 않는 부분이 2부분 있다.
1. 미니배치 크기가 일정 수준을 넘어설때
2. 학습초기(파라미터들의 변화가 심할때)

2번의 상황은 warm phase로 개선시킴  

이전 몇몇 논문에서 같은 논리가 몇번 등장했지만, 부족한 것이 대부분이여서 linear scaling rule에 몇가지를 추가해 완벽하게 만들엇다.  
<br>

## 2.2 Warmup

학습 초기에만 의도적으로 학습률을 낮춰 훈련을 함, 이후에는 일반적으로 진행  

Constamt warmup
> 초기의 학습률을 매우 작은 상수로 지정해서 앞의 소수의 에폭만 훈련, 이후에 kη로 복구

Gradual warmup
> 위의 전략은 이미지넷 라지 배치를 사용하는 우리 상황에 잘 맞지 않음. 그래서 학습률을 초기에 천천히 증가시켜 우리가 원하는 에포크 이후에는 kη이 되도록 설정한다. 이는 우리 데이터셋에 맞는 웜업이였다.

<br> 

### 2.3 Batch Normalization with Large minibatch

batch에 대한 normalization은 각 배치에 dependency가 존재한다.  

batch가 없는 경우, 각 셈플은 아무 것도 처리가 안된, 즉 독립적인 데이터를 가지고 loss를 구하게 된다. 그래서 독립성이 있다.  

하지만 BN을 하게 되면 셈플이 소속된 배치의 통계치에 대해 스케일되므로 이 셈플은 배치에 대해 의존성이 생긴다.(같은 데이터가 다른 배치에 들어가면 값이 달라지니깐)  

그래서 독립성을 셈플이 아니고 배치에 부여한다. 각 셈플은 다른 셈플에 대해 의존적이지만 각 배치는 서로 각 배치 끼리 독립적이기 때문이다. 이러면 dependency assumtion은 유지가 된다. 즉 독립단위를 셈플에서 배치로 옮겨 생각한다.  

그래서 로스에 대한 식을 보면 L(x,w) 에서  Lb(x, w)로 작은 b가 붙어 배치에 의존적이라는 것을 표현하게 끔 고쳤다.  

이렇게 되면 communication cost가 줄어든다 ( 100만개의 sample gradient sum up  vs  1만개의 minibatch total gradient sum up), worker내부에서, 통신 없이 계산할 수 있는 부분이 더 많아짐. 기존의 식과 차이점은 로스가 셈플단위에서 배치 단위로 바뀐것이다.  

그래서 워커에서 사용하는 데이터 양을 분산 학습의 파라미터로 보지말고 배치놈의 파라미터로 보자.  
Norm을 할때, 모든 워커의 정보를 통합해서 하는 것이 아닌 한 워커의 내부에서 즉 kn에 대한 놈이 아니고 n에 대한 놈으로 해야한다!


<br>

## summary 

minibatch size가 커지면 확실한 이득이 있다. 를 간단한 scaling으로 가능하게 하는 방식에 대해 배웠다.

배치놈을 할 경우 기존의 로스 계산에 다르게 접근해야하는 이유
> BN이 적용되지 않은 데이터들은, 스케일 안된 데이터 셈플, 독립적이게 로스가 구해진다.  독립적인 데이터를 가지고 구한 로스를 가지고 sum up하여 업데이트를 진행한다.


> 하지만 BN이 적용된다면 각 셈플은 소속 배치에 통계치에 의해 스케일 되어진다. 한마디로 배치에 종속적인 셈플 데이터로 로스를 구하게 된다. 종속적인 데이터를 가지고 훈련하면 안된다. “종속적이다” => “다른 셈플에 영향을 받았다”  = > “훈련 순서에 영향을 받는다”  =>  “학습이 어렵다” 라고 해석할 수 있다. 그래서 이제 로스를 배치단위로 구해서 이를 업데이트 해야한다. 배치끼리는 독립성을 유지하기 때문이다.

통신 비용이 줄어든 이유는 각 워커 내부에서도 계산 가능한 정보들이 늘었기 때문이다. 공부하기!

