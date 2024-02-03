# Training Neural Networks with Local Error Signals


## Abstract

Global loss function 이게 우리가 항상 사용하던 손실함수이다. 출력 레이어의 그레디언트를 반환에 이를 통해 BP로 히든 레이어의 가중치를 조정한다.  

우리는 layer-wise training 방법을 소개한다. layer별로 로스를 계산하는 것  
> Single-layer subnetwork and two different supervised loss function을 이용해 히든 레이어에 대한 local error signal 를 생성함.

> two different supervised loss function : Sim loss, pred loss

그리고 논문에서 biologically plausible 이라는 단어를 정말 많이 쓰는데, 일반적인 사람은 저렇게 안한다 라는 식으로 생각하면 될듯 > 인간이 자연스럽게 하는 방식이 좀더 효율적이다  

<br>

## Introduction

히든 레이어는 항상 feedforward, BP가 완전히 완료되야 업데이트가 다 됨. 즉 data pass도중에 update 불가 => parallel하게 진행하지 못함.
병렬로 업데이트 못하면 자동적으로 memory reuse못함  

layer-wise training of the hidden layers 를 사용해 backward-locking problem(memory reuse, parallel update)을 해결하겠다.  

local(hidden) error 는 global error에 독립적이되고(로컬에서 생성된 local error signal을 통해 로컬 레이어가 업데이트), gradient도 BP되지 않음. Hidden layer update 가 forward/back pass 중일 때도 가능 = parallel update  

이 방식에서 hidden layer를 업데이트 할때마다 필요가 없어지는 메모리가 존재. 이는 메모리 공간을 줄임  
각 레이어가 필요할때, local error를 구하고 이를 버림. 아주 좋다  
<br>


## Related Work
<br>

### Local Loss Function

Global loss가 아닌 global에 independ한 local loss로 모델을 pre-train하고, 이는 좋은 성능을 가진다.  

이 논문은local similarity matching loss와 결합된 분류기가  global backprop 가 대응될 수 있다는 것을 증명하는 것이다.  


synthetic(합성) gradient(=>approximation) 을 이용하는 방법도 있다.
합성 그레디언트를 L2 loss를 가지고 훈련, 훈련할 수록 real과 비슷해짐.  

<br>

### Similarity Measures in Neuroscience
뇌과학에서 유사성 분석  

우리가 사용할 sim loss를 최대한 biologically plausible version으로 만들겠다. 그에 필요한 신경과학에서의 유사도 분석방법 
<br>

### Similarity Measures in Machine Learning
머신러닝에서 유사성분석  

위의 신경과학과 최대한 유사하게 구현하겠다.  

지금 우리는 matrix 내에서 유사성을 분석하고 이를 측정해 훈련에 사용할 것인데,
이게 어떤식으로 행해지는지, 그리고 어떤 효과가 있는지를 설명  

Similarity Measures 은 unsupervised clustering and feature learning 에서 전에 나온 방법에 관계가 있다. 
우리는 거기서 나온 식과 비슷한 방법으로 supervised clustering에 대해서 할 것이다.  

위에서는 clustering을 잘하는 방식대로 했다면, 우리는 target label이 있으므로 supervised clustering loss를 구한다.  
<br>

## Method

우리는 일반적인 CNN, FF  network를 사용할 것이다. 가중치 업데이트를 Global BP 대신 local learning signal을 사용할 것이다.  

각 layer는 각자만의 2개의 single layer subnetwork를 가지고 있다(cross entropy loss, similarity matching loss). 여기서 각자의 activation을 가지고 local error signal(= local gradient)를 구하게 된다.  
이로써 data pass 도중 모든 레이어를 parallel하게 update 가능하다.  

이제 사용할 2가지 로스에 대해 설명하려고 한다. 두 로스는 이미 존재하던 것으로 먼저 그 둘에 대한 설명을 한 뒤 논문에서는 어떻게 변형해 사용하는지를 설명하겠다.
<br>

### Similarity Matching Loss (이하 sim loss)

Sim loss는 미니배치 매트릭스 와 target label matrix 사이의 L2 distance로 구함.  
그림을 보면 hidden layer output activation 정보를 가지고 loss를 구함. 그렇게 되면 one-hot label matrix와 hidden activation을 가지고 sim loss를 구하게됨  

> Lsim = ||S(NeuralNet(H)) - S(one-hot encoded label)||^2

S : cosine similarity matrix(or correlation matrix) 그냥 유사성을 구하는 함수로 생각하자  


NeuralNet : 들어온 H가 거친 layer의 타입과 동일한 타입의 layer.  
Ex) H = activation of CNN, then NueralNet is CNN with standard setting.  


<br>


### Prediction Loss

Pred loss는 로컬 분류기(각 레이어 마다 나오는 activation을 가지고 분류)와 one-hot encoded label target 값을 가지고 만들어짐.  

Lpred =  CrossEntropy(Y,W(transpos)*H)
> Y = one-hot label matrix, W = weight matrix, H = hidden layer activation

<br>

### Backdrop Free Version


Sim loss function의 식에서 NeuralNet()을 안쓰고, feature map 에 바로 표준편차 식을 적용시키고 이에 대해 유사도를 검사한다면, hidden에서 BP의 필요성이 사라짐.  
NeuralNet(.)  =>  feature map의 표준편차 연산  

loss에 사용되는 원핫 인코딩 타겟 백터는 random transformation of the same target vector로 대채된다.  

Backdrop free vision의 similarity Matching Loss을 앞으로 sim-bps loss 라 칭함  

Sim loss 등장부터 현재 논문에서 사용할 수 있게 변형하는 과정을 보면  
1. Unsupervised clustering and feature learning 에서 사용되는 sim loss (최초로 제안된 sim loss의 형태)
2. 논문에서 사용할 supervised clustering loss (논문 마춤 변형1)
3. bpf version (변형2)
이렇게 됨


마찬 가지로 Fred loss도 비슷한 과정을 통해 backprop free version을 만들 수 있다.  
이를 pred-bpf loss라 칭함.  

이러면 두 로스 모두 global target label(늘 사용하던 것) 대신 random transformation of the same target vertor를 사용하는데, 이 transformation 만드는 작업이 중요해진다.  
<br>

### Combine Loss

Lpredsim = (1 − β)Lpred + βLsim
Lpredsim−bpf = (1 − β)Lpred−bpf + βLsim−bpf

베타는 하이퍼 파라미터
<br>

## summary


layer단에서 업데이를 하는 layer-wise update방식을 제안한다. 모든 layer에서 자체 loss를 2가지 single layer subnetwork를 마련한다. 이 2개의 서브네트워크에서 구해진 값을 가지고 local error signal을 자체적으로 생산하고 이를 가지고 layer층을 각자 업데이트하게 된다. 이러면 컴퓨터 구조론에서 배운 operation의 pipeline같은 dataset parallel이 실행이 가능하다.  
