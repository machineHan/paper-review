# Deep-gradient-compresion

## Abstract	

Large scale distributed learning 에서, 파라미터가 엄청 많은 것은 항상 얘기 되던 문제이다. 그리고 그에 따른 통신 비용문제가 같이 따라 붙는다. 대역폭 문제를 해결하기 위해, 각 워커들의 통신 비용을 줄이는데 통신 데이터를 압축하는 방식을 따르게 하자.
<br/>

특히 이 통신 비용 문제는 모바일 디바이스에서 크다 : because of high latency, lower throughput and poor connection
<br/>


Gradient exchange에서(통신)  99%의 데이터가 redundant하다 > 통신 비용을 많이 줄일 수 있었다.
<br/>


Key method : momentum correction, local gradient clipping, momentum factor masking, and warm-up training
<br/>



## 1. Introduction

모델의 크기를 늘리면서, data parallelism을 활용하면 훈련시간을 줄일 수 있다.
하지만 gradient exchange(통신)에 필요한 bandwidth가 훈련의 bottleneck이 되어 이를 해결할려한다.
<br/>

bandwidth의 문제로 인해 통신에 한계가 있어 계산속도가 아무리 빨라져도 의미가 크게 없다.
<br/>


특히 모바일 기기에서 이런 문제가 두드러진다.
<br/>


DGC는 통신할 gradient를 압축하여 문제를 해결한다. 압축시 생길 수 있는 성능저하를 위의 4가지 방법을 사용한다.
<br/>

<br/>


## 2. Related Work

기존도 Communication cost를 줄이기 위한 수많은 접근이 있었다. ASGD 등등.
그중 몇가지를 살펴보자
<br/>


Gradient quantization(양자화 = 정수화)
<br/>

> ① 모델의 사이즈 축소, ② 모델의 연산량 감소, ③ 효율적인 하드웨어 사용 <br/>
> : 통신을 하기위한 숫자를 정수화 하여, 통신 비용 감소 ( float : 8bit > short 2bit)
실수형을 정수로 바꾸기 위해 실수형의 범위를 알아야함

<br/>

Gradient Sparsification <br/>
: 통신을 할때, 모든 gradient를 통신하지 말고, 기준에 적합하는 gradient만을 보냄
<br/>

조건을 충족하는 데이터셋을 만드는 것을 압축 한다고 한다.
> Ex) threshold 이상, 일정 %의 양수 and 음수 gradient만, gradient dropping(절대값 기준으로 결정, 추가 batch Norm레이어 필요), local gradient activity을 기반한 압축률 자동조정 등등

<br/>

두 방법 모두, 통신 데이터 자체를 줄임 ( ex 100% data > 10% data )

<br/>


## 3.Deep gradient compression

### 3.1  gradient Sparsification

우리는 중요한 gradient만을 압축해 통신할 것이다. 일단 중요한 것이 전송할 만큼 커지면 즉시 전송한다. 하지만 결국엔 모든 gradient정보가 다 전달 되는 것은 마찬가지다. ( 현제 loss의 기울기와 크게 엇나는 중요한 gradient만을 업데이트 )
<br/>

중요도가 낮은 gradient를 한번에 축적시켜 업데이트 하는 것은 배치 사이즈를 늘리는 것과 같은 효과를 볼 수 있다.
<br/>


주어진 local gradient accumlation에서 learning rate를 스케일 하면 미니배치를 늘리는 것과 동일한 효과를 본다.
<br/><br/>


### 3.2 improving the local gradient accumulation

위의 gradient sparsification만 사용하여 훈련시, local accumulation에서 문제가 있다. 또한 sparsity가 클수록 정확도와 수렴에 심각한 타격을 준다(빠르긴하지만). 이를 보좌하기 위한 2가지 테크닉을 부여한다.
<br/>


#### Momentum correction
<br/>


모멘텀을 추가한 SGD는 기존의 바닐라 SGD를 대채하여 자리를 잡게되었다. 하지만 이를 sparse update에 적용하게 되면 기존의 식과 다르게 변하게 된다. 
<br/>


momemtum을 dense에 대한 것이기 때문에 sparse update에 적용하게 되면 기존 식이 변형되며 여기서 문제가 발생하게 된다.
<br/>


기존의 식은 interval이 돌면 인터벌 만큼의 모멘텀이 축적되는 부분이 존재하는데 변형된 식은 이가 사라지게 된다. 손실된 모멘텀의 축적이 수렴에 부작용을 일으킨다. sparsity가 커지면 업데이트 인터벌이 더 커지고 이는 수렴에 어려움을 일으킨다.
<br/>

그래서 real gradient가 아닌 velocity를 축적한다. 이렇게 치환하게 되면 바닐라 SGD를 하는 것과 비슷하게 할 수 있다
<br/>


#### Local Gradient Clipping
<br/>


Gradient의 L2 norm값이 임계치를 넘을 때마다 rescale 
<br/>

보통 그레디언트 집계후에 이뤄지는 스텝
<br/>


하지만 우리는 sparse update여서(accumulate gradient over iteration independently) 집계 전에 이뤄져야한다
그리고, 여기서 사용되는 임계치는 모델의 sparse update에 사용되는 threshold에 N의 -1/2승 하여 사용한다.
<br/><br/>




### 3.3 Overcoming the Staleness Effect

Sparse update 자체가 지연된 데이터를 가지고 나중에 업데이트를 진행한다. ASGD에서도 발생한 Staleness문제가 나타난다. Sparse update를 sparsity 99.9%에서 실행하면 600~1000 iteration마다 업데이트가 되니 상당히 느리게 업데이트 된다는 것을 인지할 수 있다.
<br/>


#### Momentum Factor Masking
Implicit momentum : 비동기적 흐름에 의해 생기는 staleness을 칭함
<br/>


축적 그레디언트와 모멘텀 인자에 마스크를 씌움. 

이 마스크로 인해 오래된, 이상한 방향의 모멘텀이 현제 그레디언트를 옮기지 않게끔 합니다
<br/>



#### Warm-up Training


Large minibatch SGD에서 봤듯이, 초기 훈련상태의 모델은 파라미터가 공격적으로, 극단적이게 움직이는 경우가 많다. 여기에 스케일을 때리는 것은 무모하다.
<br/>

Sparsyfying gradient 하면 움직임이 제한되고 이는 모델이 급속도로 변하는 초기 구간이 길어지는 결과를 낸다.
<br/>

급변하는 모델의 그레디언트는 선택되기 전에 축적되므로 다음 업데이트에도 악영향을 미친다.
<br/>


Warm-up을 하면 덜 극단적으로 모델이 변하는 것 뿐만 아니라, 이후 업데이트에도 선영향을 미친다. Delayed update 때문
<br/>



결과는 좋다. 속도, 정확도 모두 올라가는 task도 있고, 정확도는 일부 떨어지지만 compression ratio가 굉장히 많이 올라가는 경우도 있다
<br/><br/>

## summary 

gradient를 배치마다 보내는 것이 아니고, 일정수준 모아서 보내는 것으로 업데이트 횟수를 줄인다. 이로 인해 훈련 속도가 빨라진다. <br/>

분명 전체적인 업데이트 횟수는 줄어든다. 하지만 축적을 하는 과정에서 현재 축적되고 있는 gradient가 좋은지 않좋은지 확인하는 추가적인 computational cost가 추가적으로 필요하다. 

