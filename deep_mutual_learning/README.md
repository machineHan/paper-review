# Deep-mutual-learning

## Before

먼저 이를 접하기 앞서 model distillation에 대해 설명 
<br/>
보통 우리는 큰 모델을 가지고 최고의 performance를 내기 위해 학습을 한다. 큰 모델이기에 학습 시간도 걸리고 많은 양의 자원이 필요하다. 하지만 이 모델을 직접 사용자들에게 배포하는 것은 적합하지 않을 수 있다. 시간이 오래걸리기도 하고, 사용자들의 디바이스가 강력한 하드웨어가 아닐 수도 있기 때문이다.
<br/>

그래서 좀 간략화된 작은 모델 + 약간의 성능 저하 를 뿌릴껀데, 작은 모델(이하 학생모델)을 큰 모델(이하 선생모델)을 이용해  어떻게 좋은 generalization을 학습할까 =>  Knowledge distillation 
<br/>

즉 모델 배포 과정에서 큰 모델이 적합하지 않아, 작은 모델을 뿌릴껀데 최대한의 성능을 낼 수 있도록 큰 모델이 학습한 generalization을 작은 모델에게 tansfer하는 것을 Knowledge distillation 라고 한다
<br/>

시초의 knowledge distillation은 선생 모델의 아웃풋을 가지고 만든 2차 정보인 softened output을 가지고 예측한 것과 그 이외의 것에 대한 정보를 포함한 정보를 가지고 학생을 학습시킨다.
<br/>
> Ex) 개로 예측한 선생의 output layer의 값을 가지고 softened하여 개이외의 타겟값에 대한 정보를 좀더 크게 해석한다. 여기서 개 이외의 정보도 학생을 가르치는데 사용되는 2차 정보가 된다.




## Abstract

Model distillation은 잘 사용되는데 우리는 여기서 파생된 다른 증류법을 사용하겠다.
<br/>
기존의 model distillation은 하나의 큰모델 즉 선생과 하나의 작은모델 학생을 가르키는 단방향 과정이었다.
이 논문에서는 많은 학생들이 서로 협력하여 다른 모델들의 성능향상을 도운다.
<br/>
이것은 선생이라는 큰 모델이 필요 없다는 큰 장점이 있고, 그럼에도 더 높은 성능을 낸다

<br/>

## Introduction

딥러닝은 좋은데 너무 크고, 너무 오래걸린다. 이 단점이 작은 메모리일때, 빠른 결과가 필요할 때 문제가 된다.
<br/>

간결하고 정확한 모델을 만들기 위해 많은 노력이 있었다.
<br/>

초기의 증류기반 모델 압축은 작은 모델에게 큰 모델과 같은 representation 을 학습 시킬려고함. 당연히 모델이 작아지니 동일한 표현을 만들기 위한 적합한 파라미터로 가는게 어렵다. (자원을 적게 쓰고 똑같은 성능을 찾는것은 힘들다)
<br/>

작은 모델인 학생이 큰 모델인 선생을 흉내내는 방법으로 하는 것이 위보다 더 좋은 성능으로 내고, 단순히 작은 모델을 학습하는 것 이상의 정보를 획득한다.
그리고 흉내내는 것이 직접 작은 모델이 학습하는거 보다 최적화가 더 잘 된다. 심지어 작은 모델이 선생을 뛰어넘거나 비슷한 경우도 있다.
<br/>

이 논문은 하나의 pretrained teacher가 untrained student를 학습시키는 단방향 전송이 아닌, 
“미리 학습된 선생없이” 학습이 안된 학생끼리의 양방향 전송만으로 학습하는 방식으로 진행하겠다.
<br/>

2개의 로스를 사용 : supervised learning loss, and a mimicry loss
<br/>

> supervised learning loss : 그 학생이 라벨값과 얼마나 다른가

> Mimicry loss : 다른 학생들과 얼마나 유사한가

<br/>
한 학생을 다른 모든 학생의 선생이 됨

<br/>

그리고 독립적인 하나의 큰 네트워크를 하는 것보다 많은 네트워크의 상호 학습이 더 좋은 결과를 낸다.
(그래서 학생들의 협업으로 학습하는 아이디어를 가지고 왔다)
<br/>

왜 학습이 안된 여러 학생들을 가지고 하는 것이 좋은 결과를 내느냐
<br/>
Conventional supervised learning loss를 가지고 학습을 하기 때문. 이것은 모든 학생이 같은 지점을 향해 학습을 하고 있어 임의의 값(쓰래기 값)으로 꺾는 상황을 방지함. 모든 학생이 다양한 방식으로 같은 지점을 향하기 때문에 서로의 다각도 시선(장단점)을 서로 공유하는 방식이다.
다각도의 시선으로 다양한 가능성을 서로 공유하기에 DML이 좋은 결과를 낼 수 있었다.
<br/>

각 학생들의 가중치는 다 다를거임 : 학습을 하며 한 학생이 이상한 곳에 가중치를 높게 두고 있음을 확인했다 > 다른 학생들도 이후에 그곳에 빠져들 수 있다 > 다른 학생들에게 posterior entropy 를 증가 시켜 빠져들기 전에사전에 막자
<br/>
이러면 robust, flatter한 최소값에 수렴하게 도와준다
<br/>


우리 논문에서 다음과 같은 결과도 얻울 수 있다.
<br/>
1. 협력하는 네트워크가 많을수록 성능이 좋아진다(모델이 작아서 하나의 GPU에 많은 모델 학습가능)
2. 다른 구조사이에서도 적용 가능하다 ex) 큰 네트워크, 작은 네트워크, mix architecture network
3. 우리는 배포하기 위해서 작은 네트워크끼리 협업하고, 여기서 골라 배포하지만, 큰 네트워크 사이에서 협업하는 방식도 기존의 혼자 학습하는 것보다 좋다.

우리는 배포를 위해 하나의 작은 효율적인 하나의 네트워크를 얻을라 하지만 전체 학생그룹을 앙상블 모델로 사용하는 것도 좋은 성능을 낸다.
<br/>
<br/>


## Formulation

DML 은 “지도학습 로스” + “동료들의 추정가능성에 맞게 하도록 하는 KLD-based mimicry loss” 를 사용하여 훈련한다.
<br/>

Supervised learning loss 는 단순히 cross entropy 를 사용하여 구한다
<br/>
Mimicry loss 는 다른 네트워크의 prediction을 가져와서 이를 가지고 KL distance를 측정한다. 이것이 KLD-based mimicry loss이다.
<br/>
훈련하는 순서는 다음과 같이 설명이 가능하다.

1. 모든 학생이 같은 데이터를 가지고 학습
2. 자신의 prediction을 가지고 supervised learning loss 계산 , 자신의 prediction과 다른 학생의 prediction을 가지고 mimicry loss 측정 (조건부 확률처럼 구함  ex - network1은 “Pn | P1”의 확률을 가지고 mimicry loss를 계산, n은 network1 이외의 다른 학생들)
3. 이 둘을 sumation한 값을 가지고 학생 모델업데이트, Network 1 loss using for update  = L(supervised learning) + KL distance between student and other students

즉 다른 students와 해당 학생의 prediction 값이 유사해질 수록 KL distance는 줄어들어 total loss가 줄어듬



<br/>
<br/>

## 2.2 optimisation


이 훈련 전략은 미니배치 모델 업데이트 스텝마다 이뤄진다. 각 반복마다,  우리는 두 모델의 예측값을 내고, 이를 가지고 두 네트워크 파라미터를 업데이트 한다. 각 학생들의 학습은 수렴될 때까지 행해진다.
<br/><br/>

## 2.3 Extension to Larger Student Chorus

2.1~2에서 두개의 학생을 가지고만 예시를 들었지만 이제는 많은 학생들에 대한 설명을 하겠다.
<br/>

각 학생들의 로스에 약간의 수정이 있다. 로스에 포함되는 KL distance 관련 부분이 k-1 개의  자신을 제외한 모든 학생들의 KL distance의 평균이 된다. 이 외는 특별한 사항이 없다.
<br/>

그리고 실험에서 2개의 네트워크를 사용하는 것보다 k개의 네트워크를 사용하는 것이 더 좋은 결과를 내는 것을 발견했다.  model averaging step에서 점점 정확도가 높아지며, posterior entropy가 낮아지는 양상을 보이기 때문이다. 
<br/>

하지만 high posterioir entropy를 만들어서 robust solution을 만들어야 하는 이 논문의 목표와 모순되는 결과이다.
<br/><br/>



## summary

이제 distillation할 때 학생을 선생을 통해 학습하는 것이 아닌, 각각의 학생들와 유사하게 행동하도록 흉내를 낸다. 그래서 큰 선생 모델이 필요 없다는 것이 아주 좋은 장점이다. <br/>

학생이 많으면 너무 의견이 동일해지는 시간이 오래 걸려 엔트로피가 낮아 지는데 시간이 오래 걸린다.<br/>

학생들의 의견을 수렴하면서, label값으로 천천히 움직임.


학생이 많다 > posterior probability가 적다 (내 의견에 동조하지 않는 학생들이 많다, 근데 그거까지 반영한다) > posterior entropy가 크다
 

왜 posterior probability와 mimicry loss가 관련이 있냐 : 모델n의 발생이 일어날때, 동일하게 모델m에서 동일한 일이 일어 났다 = 잘 따라 했다 = mimicry loss가 적다 = 조건부 확률, 가설이 실행됐을 떄, 어떤 일이 일어날 확률
