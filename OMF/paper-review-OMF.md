# Provable Dynamic Fusion for Low-Quality Multimodal Data

## Before Read

Generalization error : 머신러닝에서 일반화 오류라고 하면, 보지 못한 데이터에 대해 알고리즘이 얼마나 정확하게 결과를 낼 것이냐를 측정한 것이다.  

훈련 알고리즘은 유한한 데이터에 대해 평가 받기에, generalization error에 대한 평가가 sampling error에 민감하다.  

그 결과 현재 데이터에 대한 예측 오류가 새로운 데이터 측정 능력에 대한 정보를 많이 전달하지는 못한다. 

Generalization error, expected loss, risk 값은 Loss function을 통해 구할 수 있다.  


<br>

## Abstract

Multimodal fusion의 과제 중 하나는 각 모달사이의 연관성을 빠르게 파악하고 이에 토대로 유연한 상호작용을 하는 것이다. 각 모달리티의 가치를 모두 사용하고 낮은 질의 멀티모달 데이터의 영향을 완화시키기 위해, dynamic  multimodal fusion이 제안되었다.  

dynamic  multimodal fusion이 자주 쓰임에도 불구하고, 이론적 증명이 부족하다. 이 논문은 일반적인 관점에서 이론적인  분석을 한다.  

그리고 classification task에서 성능과 robustness를 향상시키는 QMF라는 fusing method를 제안한다.  


<br>


## Introduction

기존에 존재하던 fusing method는 data의 질을 간과한다. 최근 연구에서 low-quality의 Multimodal data를 fusion하는 것이 좋지 않다는 것을 실험, 이론적으로 증명했다.   

높은 noise와 데이트셋 내부에서의 불균형한 정보의 quality의 영향으로, Multi-modal이 항상 uni-modal보다 좋은 것이 아니라고 실험을 통해 보였다.

이론적으로, 제한된 데이터의 양에서 Multi-modal learning의 장점이 사라진다는 것을 보였다.  

이러한 상황에서 각 모달리티를 잘 사용하고, 질이 낮은 데이터의 영향을 줄이기 위해, Dynamic fusing method가 제안됐다.  

Dynamic multimodal fusion이 놀라운 성과를 보여주고 있지만, 이론적인 이해가 결여된 채 사용되고 있다.  

이 논문에서는 강력한 multi-modal fusion method의 기준과 이점을 증명할 것이다.

이 연구에서 제안된 Quality-aware Multimodal Fusion(QMF)는 decision-level fusion의 연장선이다. framework의 핵심은  각 Modality의 quality를 특성화하기 위해 Energy-based uncertainty를 활용한다.  


이 논문에서 개재되는 바를 요약하자면 다음과 같다.

- 이 논문은 강력한 multimodal fusion의 기준과 이점을 이해하기 위한 이론적인 framework를 만든다.  
먼저,   decision fusion method의 일반화 오차 한계(= 일반화 능력)를 Rademacher complexity 관점에서 분석한다. 그리고 어떤 상황에서 static fusion보다 dynamic fusion이 좋은지를 찾는다. <br> 결과 부터 말하자면, Multimodal fusion의 fusion weight가 unimodal generalization error와 음의 상관관계가 있을 때, static 보다 dynamic fusion이 더 좋은 성능을 낸다.

- 이론적인 분석하에, dynamic fusion의 일반화능력이 uncertainty estimation의 성능과 동일하다는 것이 보여진다. 이것은 새로운 dynamic fusion algorithm을 평가하고 설계하는 원리를 직접적으로 암시한다.

- 위의 분석에 영향을 받아, QMF를 제안한다. 이것은 
더 좋은 일반화 능력을 제공한다.


<br>

## Related works

### Multimodal Fusion

크게 3가지로 나눠진다. Early, intermediate, late fusion  

이전 연구에 intermediate fusion이 representation learning에 더 이점이 있다고 하지만, late fusion 방식이 단순하고, 해석에 용이하여 좀 더 널리 쓰이고 있다.  

이전에 “dynamic weighting mechanism” 이라는 기술은 제시한 dynamic based fusion 연구가 진행된 적이 있다. 이 연구에서 확실히 상황에 맞는 fusion방식이 더 좋다는 것을 보여줬다.  

Uncertainty-based multimodal fusion method가 많은 테스크에서 큰 이점을 보이고 있다.

<br>

### Uncertainty Estimation

반복하지만, fusion method에 대한 논리적인 이해가 충분하지 않다. 이로 인해 안정성이 중요한 분야에선 사용하지 않는다.  

Uncertainty estimation은 머신러닝 모델에서 나온 결과값이 얼마나 틀리는 지에 대해 나타낸다.

Predictive confidence, DST, Energy score ….

<br>

## Theory

### Prelinimaries

x = input data = {x1,x2,..,xm}  m = number of modality  
y = label

X,Y,Z = input space, target space, latent space

(x, y) ∈ X × Y

h = X -> Z : multimodal fusion
g = Z -> Y : mapping to target label

목표는 f = h*g(x) 인 multimodal model이 unseen dataset에서 좋은 성능을 내개끔 하는 것이다.

D = (x,y)
Dtest, Dtrain = joint distribution of D

<br>

## When and How Dynamic Multimodal Fusion Help

순서는 다음과 같다
1. Rademacher complexity 를 이용해 dynamic late fusion의 generalization error bound를 분석한다. 그리고 bound를 3가지 요소로 분리
2.  분리된 요소를 가지고, 어떤 상황에서 dynamic 방식이 좋은지를 찾아낸다.

우린 밑에서 설명한 basic setting에 따라 분석을 시작한다.

<br>

#### Basic setting

scenario : M개의 모달리티, 이진분류 task  
M개의 unimodal classifier을 준비한다. 최종 prediction은 각자의 modality에 따라 decision * weight 을 하여 계산된다. (Ensemble-like late fusion)  

Static fusion과 대조적으로, dynamic fusion의 decision weight는 다른 셈플에 따라 다양하게 생성된다.   

- w(m,static) : decision weight on static fusion in m modality, constant value
- w(m,dynamic) : decision weight on dynamic fusion in m modality, function of input value x


#### Theorem 1 : Generalization bound of multimodal fusion

> generalization bound : 머신러닝에서 훈련 알고리즘을 정할때, Generalization error가 가장 적은 hypothesis를 골라야한다. 하지만 Generalization error는 unknown distribution에 의존하므로 정확한 값을 구할 수 없다. <br>그리하여 대안으로 empirical error를 이용해 Generalization error의 bound를 정하고, 이를 통해 비교를 한다. 이를 generalization bound라 칭한다.


다음은 multimodal fusion의 generalization bound 이다.  

![식 2](https://github.com/machineHan/paper-review-tree/assets/154798552/234160df-d1da-4cfb-8e6a-f450ee894080)

식을 보면 multimodal classifier의 generalization error는 모든 unimodal의 component의 평균성능에 의해 제한된다.

> unimodal component : 1)empirical loss, 2)model complexity, 3)covariance between fusion weights and unimodal loss

이제 dynamic fusion이 어느 상황에서 static Fusion 보다 tighter bound를 가지는지(=좋은지) 대한 증명을 하겠다.

일단 위의 bound식에서 static Fuison은 Term-Cov는 0이다. Static fusion에서 사용되는 unimodal classifier wight가 상수기 때문이다.

이런 상황에서 dynamic이 static 보다 좋을려면, dynamic fusion에서 Term-C/L의 합이 더 작거나, Term-Cov가 음의 값이라면, dynamic fusion이 static fusion보다 좋은 결과를 낸다고 증명할 수 있다.

<br>

#### Theorem 2 : scenario that dynamic fusion is better than static fusion


![식 4](https://github.com/machineHan/paper-review-tree/assets/154798552/aa120414-7d71-411a-9a13-bc9054c5c4fc)
우리는 이 식을 만족하는 상황을 찾아야한다.

![식 5](https://github.com/machineHan/paper-review-tree/assets/154798552/d3fa4aac-06af-4d47-a98c-604c51404145)
![식 6](https://github.com/machineHan/paper-review-tree/assets/154798552/27d46e3c-ab8f-44e7-82bf-6d3ec6f5f82d)

모든 input modality에 대해 식 5,6이 성립되는 상황에서는 다음에 주어지는

![식 7](https://github.com/machineHan/paper-review-tree/assets/154798552/0b8a078f-e587-489c-9ac6-d54bd3137df0)
![식 8](https://github.com/machineHan/paper-review-tree/assets/154798552/32724d1b-7001-4c97-a907-da16aef88325)

식 7,8이 성립되어 식 4번이 성립된다. 즉 dynamic fusion이 더 좋다.  

즉, 식 5,6번을 만족하는 unimodal classifier weight라면 Dynamic fusion의 상황에서 더 좋은 성능을 발휘한다. 그리하여 식 5,6번에 성립하는 fusion weight를 찾는 것이 관건이다.  


<br>

## Method

이 섹션에서 dynamic multimodal fusion 과 uncertainty estimation의 관계에 대한 논리적인 분석을 하고, Quality-aware Multimodal fusion(QMF)를 제안한다.  

<br>

### Coincidence with Uncertainly Estimation

dynamic fusion에서 식6.번을 만족하는 weight를 찾는 것과 uncertainty estimator의 등장배경이 본질적으로 비슷하다. 그리하여 식 5,6번에 부합하는 weight를 uncertainty를 통해 구하겠다  

- 추정 1 : m modality에 대한 uncertainty estimator하나가 주어졌을 때, 측정된 uncertainty는 Modal-specific loss와 양의 상관관계가 있다. => uncertainty가 증가하면 loss도 커진다  

이 추정으로 인해 일반적인 static fusion method보다 dynamic fusion method가 좋다는 것을 탐색할 기회를 준다.  


<br>

#### uncertainty-aware weighting

Uncertainty-aware fusion 함수는 구해진 uncertainty의 선형, 음의 함수이다.  
식은 다음과 같다.

![식 9](https://github.com/machineHan/paper-review-tree/assets/154798552/0f0bd857-728b-40ea-b1ac-cfa9326c72ac)

α ,β는 하이퍼 파라미터로, 이를 조정하며 식 5,6에 맞는 weight를 찾을 수 있다.  

추정 1을 통해 어떤 static fusion weight 든지 식5,6번을 만족하는 β가 무조건 존재한다. 이를 만족하는 weight를 찾고 이를 통한 decision을 만든다.   

<br>


#### Enhance Correlation by Additional

Robust dynamic fusion의 핵심과제(식 5,6번에 부합하는 weight 찾기)를 uncertainty를 통해 찾을 것이다. 그리하여 uncertainty learning을 진행하게 된다.   

이 논문은 Uncertainty learning에서 널리 사용되는 energy score라는 것도 채용한다. Energy score는 주어진 데이터의 Helmholtz free energy와 이것의 밀집 사이의 격차를 줄이는 역할을 한다.  => 균일한 분포로 만든다  

Multimodal 상황에서, density function은 energy function을 통해 만들 수 있다.  

![식 12](https://github.com/machineHan/paper-review-tree/assets/154798552/3fa69234-0353-454e-a496-6c7f09d1290e)

이 식에서 energy 함수는 energy함수에 대해 선형 함수이다.  

다음은 density 함수를 구할때 사용하는 Energy 함수이다.  

![식 13](https://github.com/machineHan/paper-review-tree/assets/154798552/fb84cf4c-d246-4b48-9a39-e7fabc4e08e6)

균일하게 분포된 예측은 높은 질의 uncertainty를 얻을 수 있게한다.   

하지만 이를 통해 얻어진 Uncertainty는 추정1을 만족시키기 어렵다는 것을 실험적으로 발견했다. 그리하여 기존의 방식에 correlation을 강화하는 방식의 sampling-based Regularization를 추가하기로 했다.   

Respective loss와 uncertainty간의 상관관계를 향상하기 위한 간단하고 직관적인 방식은 훈련 중에 sample-wise loss를  supervision info로써 활용하는 것이다.   

최근 연구인 bayesian learning과 uncertainty estimation을 통해, 과거의 훈련 궤적을 활용하여 fusiom weigth를 regularize하는 방식을 채택한다.  

특정 모달리티 i에 대한 데이터 포인트가 (xi,yi)이 주어 졌을 때, 훈련 평균 로스는 다음과 같이 구해진다.  

![식 14](https://github.com/machineHan/paper-review-tree/assets/154798552/08453159-dfdf-4311-8d0e-360a0886cc35)

각 epoch마다 생겼던 Loss에 대한 평균치를 나타낸다. 왜 과거의 훈련 궤적을 사용한다고 하냐면, 식에서 에포크마다의 loss를 가지고 평균을 내기 때문이다.

어느 논문에서, 분류하기 쉬운 셈플들로 학습을 한 것이 분류하기 어려운 셈플들로 훈련한 것 보다 더 쉽게 배워진다는 것을 보였다. 다음과 같은 관계를 따르는 훈련에 의한 dynamic fusion model을 정규화하는 것이 바람직하다.  

![식 15](https://github.com/machineHan/paper-review-tree/assets/154798552/067f3e21-def2-4d6e-a963-686ba70da5ce)

마지막 15번째 식을 끝으로 우리는 regularization에 대한 식을 완성할 수 있고, QMF method에 적용될 전체적인 로스를 구할 수 있다.


![식 16](https://github.com/machineHan/paper-review-tree/assets/154798552/691051e4-2db0-49eb-9da0-63535184405e)
![식 17](https://github.com/machineHan/paper-review-tree/assets/154798552/1e6b2d26-3c8b-4fa7-ac52-903b8be914fa)
![식 18](https://github.com/machineHan/paper-review-tree/assets/154798552/71ce2b54-ef63-483d-8479-057014075259)

<br>

QMF 순서에 대해 간략하게 서술하겠다.
1. 입력에 대한 각 unimodality classifier에 결과를 만듬
2. 1번의 과정에서 각 modality에 대한 training average loss를 구함
3. Uncertainty-aware fusion weight 방식을 사용하여, 각 modal에 대한 weight를 구한다.
4. 1,3번 과정에서 나온 결과를 통해 최종 decision을 출력한다.
5. Multimodal decision, unimodal decision, regularation을 통해 total loss를 구한 후, 이를 가지고 각 unimodal predictor를 update를 진행한다.


<br>


## Summary ans impressions

여기는 논문에서 느껴지는 논리적 흐름의 순서를 나열하겠다.

1. 기존의 ensemble-like late fusion에 대한 논리적인 분석을 하자, multimodal 상황에서 generalization bound를 통해 분석

2. 식 5,6번을 만족할 때 dynamic late fusion이 static보다 좋다. 식 5,6번을 만족하는 weight를 찾는 것이 관건이다.

3. 특정 assumption을 사용하면, 6번식을 만족시키는 것 (weight와 loss간의 correlation을 만족시키는 것)이 uncertainty와 loss간의 correlation을 만족시키는 것과 결이 비슷하다. 그리고 더 구하기 쉽다.

4. uncertainty를 이용한 식 5,6번을 만족하는 weight를 찾았다.

5. Uncertainty learning에서 사용하는 Energy score를 사용하자. 그러면 좋은 질의 uncertainty를 얻을 수 있다. 하지만 이를 사용하면 3번 과정에서의 assumption이 어긋나는 경우 발생한다.

6. 그래서 weight와 loss간의 correlation을 높여주는 regularization을 추가 = 3번 과정의 assumption이 보존

7. 구해진 각기다른 Loss를 가지고 unimodal predictor를 업데이트한다.

전반적인 fusion method에 대한 논리적 분석을 다루지 않고 late fusion, ensemble-like fusion에 대한 분석만을 다루고 있다.  

근사적인 방식의 중요성도 보인다. 대부분의 논문이 어려운 난관을 타개하기 위해 비슷하지만 다르고 좀 더 쉬운 방식을 택한다. 이 논문에서도 weight를 찾는 상황에서 Assumption를 설정해 uncertainty를 가지고 구했다.

읽으면서 수학적인 지식이 많이 필요함을 느꼈다, 특히 정보이론, 베이즈정리에 대한 고찰이 더욱 필요하다.  이해했다고 말하기 어려운 논문이었다.  


