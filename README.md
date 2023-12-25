# paper-reveiw-Improving-multimodal-dataset-with-Image-Captioning

Improving multimodal datasets with image captioning

요약

 실험 순서 
1. Datacamp dataset scale 선택
2. 캡션 생성모델 선정(BLIP, BLIP2 , OpenCLIP-CoCa) (standard 캡션 생성 모델에 대한 metrics에 대한 재고)
3. 생성된 캡션 + raw 캡션을 가지고 여러가지 필터링, 데이터 셋 구축
4. 완성된 데이터 셋을 가지고 메인 모델 training 
5. 4번에서 훈련한 메인 모델을 가지고 여러가지 데이터 셋(ImageNet, 38 task average)에 대한 평가


캡션 생성 모델의 성능을 평가하는 수단 여러개를 비교. BLUE4, CIDEr, cosine simolarity … 생성된 캡션들을 가지고 메인모델 훈련후, 메인모델에서 퍼포먼스를 측정하는 방식으로 성능 측정.

논문에서 등장한 benchmark는 크게 2종류로 pretrained CLIP model을 평가할 것과, caption mixing + filtering할 때 사용하는 것 이렇게 2개로 나눠짐. 이를 구분해서 생각하자

Standard image captioning benchmark가 전자에 속한다.

이 논문에서 등장하는 여러가지 아키텍처를 명확히 구분해야함

Main model : CLIP
Caption model : BLIP1, 2, OpenCLIP-CoCa
Pretrain Benchmark : datacamp
image captioning benchmark : cosine similarity, BLUE4, CIDEr, ImageNet accuracy.. etc



Abstract 

인터넷에서 터프하게 크롤한 데이터는 noise가 많다. 이를 제거하는 method 역시 존재하지만 이는 data diversity를 크게 홰손한다. 우리는 이미지 캡션의 퀄리티, 특히 noise 관점에서 집중한다. 

생성 캡션의 성능을 평가하던 기존의 standard benchmark가 별로라는 것도 발견했다
ex) BLUE benchmark에서 가장 높은 점수를 받은 캡션이 가장 좋은 성능을 내지 못함.



Introduction

인터넷에서 크롤한 이미지 텍스트 데이터셋은 noise가 너무 많다. 이를 전처리하는 method가 많지만 대부분이 휴리스틱(증명이 없는 상식에 기반한 방식)에 기반한다. 캡션 전처리 기술 대부분의 방법이 어떤 상황(캡션 퀄리티가 좋아도, 별로여도)이든 대부분의 데이터를 버린다.

우리는 이 버려지는 데이터에 focus한다. 버려지는 데이터 셋의 일부를 생성 캡션을 이용해 재사용하겠다. 전처리가 최소화된 Datacamp benchmark를 사용하겠다. 이로서 raw data와 유사한 환경에서 실험이 가능하다. 

생성 캡션, 합성 캡성을 좀 혼용하여 사용하는데 둘이 같은 뜻으로 사용했다. 이를 명심하고 읽도록

최근의 캡션 생성 모델은 확실히 좋다. 생성된 캡션으로 CLIP을 훈련하니 좋은 성능을 낸다. 기존에 사용하던 코사인 유사도 상위 n%를 이용한 raw data 훈련보다, 합성 캡션을 이용해서 훈련하는 방식이 더 낫다.

합성 캡션(생성된 캡션)의 이점을 이해하기 위해, 캡션 noise, diversity를 측정해야한다. 그리고 두 특성이 모델 성능에 얼마나 영향을 미치는지 알아야한다.
기존의 fittering방식은 noise는 줄이지만 diversity에 손상이 크다. 합성 캡션을 사용하는 것으로 이 손해를 매꿀 수 있다. 그 이유는 기존의 방식 + @(원래 버려지던 데이터들) 기 때문에 기존에 비해 다양성이 떨어질 수 없다

좋은 downstream 성능을 내기위한 캡션모델 선택은 쉽지 않다. 이미지 캡션 밴치마크의 성능이 좋은 모델이 생성한 캡션이 꼭 CLIP 훈련에 좋은 것이 아니기 때문이다. 

Fine-tuning, Optimization in caption model는 CIDEr에선 좋은 점수를 받지만, 실제 multimodal training(메인 모델)에서는 이와 비례하지 않다. 캡션모델을 downstream에 전문화하면 안된다.
즉, 캡션 생성모델은 가져온 그대로 사용해야 좋은 성능을 낼 수 있다.

이 논문의 핵심은 다음과 같다
-합성 캡션 집단은 노이즈도 적고 다양성도 적다. Raw 데이터 집단은 노이즈도 높고 다양성도 높다. 이를 섞어서 둘의 이득을 취하겠다.
-사용하는 데이터의 양에 따라 필터링 성능이 달라진다. 즉 특정 스케일마다 최상인 필터링 방식이 있다는 것이다.
-standard image caption benchmark에서의 성능이 실성능과 비례하지 않다.
-캡션 생성 모델을 fine-tuned하는 것은 좋지않다.

Impact of model specialization on captions generated for multimodal training

Standard image captioning benchmark에서 높은 점수를 딴 생성 캡션이 정말로 모델 훈련시 더 좋을까?

기존엔 reference-based metrix이 많이 쓰였다. CIDEr, BLEU4 등등. 당연히 이전에 작성된 논문중에 이를 사용하여 성능을 측정한 것이 많다. Reference-based metrix를 사용했기에 좋은 성능지표를 받기 위해 captioning model은 fine-tuning하였다.

우리는 BLIP, OpenCLIP-CoCa with and without fine-tuning 이렇게 4가지 종류의 캡션 생성모델로 생성한 합성 캡션으로 메인 모델인 CLIP을 훈련시킬 것이다. 결과를 말하자면 fine-tuning 모델은 메인모델의 검색능력은 올리지만 이미지넷의 분류능력은 떨어뜨린다. 우리는 fine-tuning이 생성된 캡션의 다양성을 떨어뜨려 생긴 결론이라고 생각한다. 증명은 안한듯

fine-tuning하지 않은 모델은 reference-based metrix에서 낮은 점수를 받는 경향이 있다. 이전의 논문들은 이 reference-based metrix을 기저에 두고 생각하기에 fine-tuning하지 않는다면 쓸모가 없을 거라고 생각을 했을 것이다.

Reference-based metrix은 사람이 생성한 reference caption에 의존해 결과를 내는 반면 reference-free metrix은 생성된 캡션 + 이미지간에 관계에 대해 결과는 낸다. 우리는 이 reference-free metrix중 하나인 CLIP-S에 대해 계산할 것이다. 실제로 실험에서 CLIP-S,cosine similarity와 같은 reference-free metrix은 CIDEr score과 같은 reference-based metrix에 다른 움직이며 좋은 결과를 보여준다. 더 zero-shot performance를 잘 대변한다.

캡션 생성모델인 BLIP2가 굉장히 강력하므로 따로 언급이 없다면 이를 사용하여 캡션을 생성할 것이다

Filtering raw and synthetic captions

이제 실험에서 사용할 필터링 방식에 대해 설명하겠다.
1. No filtering: 모든 이미지 + 생성캡션 사용
2. CLIP score filtering : 코사인 유사도를 따져 상위 n%만 학습
3. CLIP score filtering with IamgeNet1k clustering : ImageNet1k의 이미지 중 클러스터 중심이 가장 가까운 이미지만을 선택. 그리고 이 집합과 top n%의 집합을 가지고 교집합으로 데이터를 선택, 훈련이 방식이 raw caption을 가지고 훈련하던 방식중 가장 좋았던 방법임.
4. Combining raw and synthetic captions : 말 그래도

예시 ) raw(top 30) + BLIP2(70) : raw caption-image의 코사인 유사도중 top 30% 사용 + 걸러진 나머지 모두를 generated caption로 훈련

raw(top 30) + BLIP2(70, filtered)  : raw caption-image의 코사인 유사도중 top 30% 사용 + 걸러진 나머지를 generated caption를 사용하여 코사인 유사도 측정, 임계치 70 이하의 데이터셋을 버리고 나머지를 다시 훈련풀로 복구

BLIP2(top 70) + raw(30, filtered) : generated caption-image의 코사인 유사도중 top 70% 사용 + 걸러진 나머지를 raw caption로 코사인 유사도 측정, 임계치 30 이하를 거르고 나머지를 다시 훈련 풀로 복구

실험 결과를 보면 mix caption을 사용할때, 포함되는 raw caption이 커지면 커질 수록 단순히 합성 캡션만을 사용했을 때보다, 성능이 낮아진다. 즉, 합성 캡션을 모두 사용하는 것 보단 필터링을 사용하여 하면 좋은 성능을 낼 수 있다.

What makes synthetic captions effective?

실험을 통해 본 결과 raw만 쓴다면 정확도는 낮다, 하지만 결과로 나오는 unique trigram의 수가 굉장히 많다.
그에 반면 BLIP2 합성 캡션만을 사용하여 훈련을 할 경우 정확도는 눈에 띄게 오르지만 unique trigram의 수가 많이 줄어든다.
이 둘을 mix + filtering 한다면 정확도와 diversity 두 마리의 토끼를 모두 잡을 수 있다.

——————————————————-
N-gram model : n - 1개의 단어를 사용해서 나올 다음 단어를 예측
Ex) trigram : sample sentence “An adorable little boy is spreading ??”  : 만약 spreading 의 다음 단어를 예측하고 싶다면 ‘boy is' 만을 입력, ‘An adorable little' 는 같이 입력되지 않는다.
그리고 논문에서 unique trigram의 갯수를 표기하는데 이는 단어 예측 시 넣어지는 n-1 단어의 조합 수를 뜻하고, 이는 diversity와 밀접한 연관이 있다.
——————————————————-

느낌점

multimodal learning에 익숙하지 않은 채로 논문을 읽었다. 그래서 정말 오랜시간이 걸려 이해를 했다. 특히 논문에서 등장하는 아키텍처에 대한 구분이 너무 어려웠다. 애초에 처음으로 benchmark, dataset 에 큰 관심을 두고 읽은 논문이라 첫 리뷰시 이해가 되는 내용이 너무 없었다. 하지만 여러 차례 다시 읽어보니 이전에 읽었던 내용과 다르게 이해됐다. 
확실이 이 논문대로 코드로 구현을 한다면면 들어온 데이터셋의 어떻게 흘러가고 어떤 식으로 변환되어 어떤 출력이 나오는지를 머리 속에 그리며 상상하니 큰 도움이 되었다.
