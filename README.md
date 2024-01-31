# Efficient Neural Network Training via Forward and Backward Propagation Sparsification

## Abstract

Sparse training은 훈련을 가속하기 위한 자연스러운 방법. 하지만 backprop의 체인룰 때문에 좋은 성능을 내기가 어려움, backprop는 이전 단계의 dense한 정보를 가지고 훈련함

<br/>

> 그래서 우리는 진짜 완벽한 sparsification 해보겠다.

우리는 기존의 backprop의 chain rule을 가지고 하는 방식과 다르게 한다.
<br/>

Weight update + model parameter update 이렇게 크게 두가지로 나뉜다.
Weight update는 기존의 체인룰과 같다. 단지 sparsiciation이 적용된 상태에서 하는 것
<br/>

Model parameter update는 체인룰과 완전 다른 것을 사용할것이다. 이 방식은 backprop가 필요 없고 앞의 2 layer의 forward Pass 만 요구한다.


<br/>


## Introduction

pruning으로 network sparsiciation을 시작했다. Pruning을 적용한 모델은, 무시가능한 약간의 성능하락과 큰 모델 크기 감소를 이뤘다.
<br/>


기존의 sparsiciation을 크게 두가지로 나눌 수 있다. Parametric and non-parametric
<br/>

Sparse network training을 자세히 보면 실용적인 상황, 일반적인 플렛폼 위에서 accelerating이 어려운 이유를 알 수 있다.
<br/>
 
기존의 방식들은 네트워크 sparsiciaiton에만 목적을 두고 서브네트워크을 만드는 데 필요한 추가 계산 비용 대한 것은 염두하지 않음. 모델이 작아진건 맞는데, 만드는데 사용하는 추가 비용이 크다.
<br/>

> Parametric : 매번 새로운 weight가 pruning될 때마다 처음부터 retrain

> Non-parametric : pruning 된 weight가 0에 한없이 가깝지만 0이 아니여서 여전히 dense backprop가 진행됨

다른 방식들 중 몇몇은 처음에 항상 큰 모델로 부터 시작해서 점차 줄여나간다. 그래서 초기단계의 계산비용, 모델 크기, 메모리 사용량이 큰 것으로 인한 한계가 분명 존재한다.
<br/>


그래서 우린 efficient channel-level parametric sparse neural network training method를 제안한다.
 => perfect sparse training

<br/>

## 3. Why Existing parametric methods cannot achieve farcical speedup?

마스크를 통해 prunning channel이 0에 한없이 가까워지지만 0이아님. 그래서 dense computation이 진행됨
<br/>

Computational cost가 작게 줄어듬
<br/>


## 4.1 channel-level sparse training
기존의 channel-level sparse training은 이런 방식으로 이뤄진다. 라는 내용을 담고 있음

## 4.2.1 Fitter update via completely sparse computation

그냥 우리는 pruning chaanel fitte를 0으로 만들것이다. 이 상황에서 prunning는 업데이트 자체가 필요없고 not Running channel fitter 의 backprop 중에 prunning channel fitter를 지나온 그레디언트도 필요 없어지므로 효율적인 계산이 가능하다.
<br/>

결론은, fitter update는 chain rule을 사용하되 completely sparse backprop for prunning channel fitter가 가능하다.
<br/>


##4.2.2 Structure Parameter Update via Variance Reduced Policy Gradient

PEG는 forward pass를 통해 gradient를 추정할 수 있다.  Complete sparse상황에서 PEG로 로스를 구하는 것은 효율적으로 진행된다. 하지만 기존 PEG는 다른 논문에서 다룬바와 같이 높은 분산을 가지고 있다.
<br/>

그래서 VR-PEG를 제안한다. VR-PEG는 분산이 제한된다
 이유는 이해 못하겠음
Abstract 에서 VR-PEG는 2개의 forward pass만을 사용한다 했는데 그것역시 확인하지 못함.
강화학습에 대한 내용
