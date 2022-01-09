# 玉山人工智慧挑戰賽2021冬季賽-信用卡消費類別推薦程式碼分享

## 1. 競賽說明

請參閱[T-Brain官方網站-玉山人工智慧挑戰賽2021冬季賽](https://tbrain.trendmicro.com.tw/Competitions/Details/18)

## 2. 競賽成果
* 本次競賽為獨自參加，隊伍名稱：Macaca Cyclopis
* Public Leaderboard: 排名28/859，Top排名3.26%，分數：0.714550
* Private Leaderboard: 排名22/859，Top排名3.26%，分數：0.713800

## 3. 程式說明

程式碼已整理在Colab：

<a href="https://colab.research.google.com/github/SuYenTing/-esun_2021_winter_ai_competition/blob/main/%E7%8E%89%E5%B1%B1%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E5%85%AC%E9%96%8B%E6%8C%91%E6%88%B0%E8%B3%BD2021%E5%86%AC%E5%AD%A3%E8%B3%BD%E7%A8%8B%E5%BC%8F%E7%A2%BC.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

此程式碼是在Colab Pro環境下執行，由於本次競賽的資料集非常大(約3,000多萬筆資料)，若只單純用Colab跑會遇到記憶體不足的問題。我已有先減少特徵及訓練樣本數方便整理資料及模型訓練能夠順利，實際上在競賽時有用到更多特徵及訓練樣本，但模型訓練與預測的流程框架並沒有改變，相關的差異有註解在程式碼內提供參考。最後是在Azure上開一台虛擬主機，規格為一顆K80 GPU和56G記憶體才能順利訓練完整資料。

## 4. 參賽心得

以下分享我認為在這次競賽中，值得提供參考的3個做法：

### 4.1 資料量龐大到記憶體無法負荷

本次競賽是我第一次遇到資料大到無法用pandas直接整理資料的狀況，在Colab Pro的High RAM(25GB)記憶體環境下，用panda時讀取資料整理時記憶體會無法負荷。最後在學弟冠廷的協助下，利用[Dask套件](https://dask.org/)分散式讀取資料可解決此問題。這次競賽在前期花費許多時間研究Dask套件要如何使用，個人認為Dask是一個可以值得投入研究的套件，其api幾乎皆與pandas相同，整合得相當好。

### 4.2 Naive Prediction

在提交預測結果時，有發現部分客戶並不會在每一期都會消費，所以有可能發生該客戶沒有特徵資料的狀況讓我們能夠預測。例如說我們以dt=13到24期的樣本來預測dt=25期，但有些客戶並沒有在dt=13到24期消費，所以完全沒有資料。為了解決這個狀況，我直接整理每個客戶在dt=1到24期各類別消費金額，由高到低取前3名作為naive預測。當預測樣本缺少該客戶的資料，則直接以naive預測的排名補上。另外我也有實測直接提交naive預測結果，分數可達0.67，顯見naive直接預測效果也是蠻好的。

### 4.3 評分指標一致性

本次競賽我主要使用XGBoost的Learning to Rank模型，XGBoost有提供ndcg評分指標，但和主辦單位有些許差異。

此次競賽主要是以NDCG@3作為評分指標，我們由官網提供的DCG定義來看：

![DCG_{c}=\sum_{i=1}^{i=3}\frac{V_{i,c}}{log_{2}(1+i)}](https://latex.codecogs.com/svg.latex?\Large&space;DCG_{c}=\sum_{i=1}^{i=3}\frac{V_{i,c}}{log_{2}(1+i)}) 

分子的![V_{i,c}](https://latex.codecogs.com/svg.latex?\Large&space;V_{i,c})為客戶在該類別消費的金額。

這邊要注意的是XGBoost的DCG定義為：

![DCG_{c}=\sum_{i=1}^{i=3}\frac{2^{rel_{i}}-1}{log_{2}(1+i)}](https://latex.codecogs.com/svg.latex?\Large&space;DCG_{c}=\sum_{i=1}^{i=3}\frac{2^{rel_{i}}-1}{log_{2}(1+i)}) 

其中![rel_{i}](https://latex.codecogs.com/svg.latex?\Large&space;rel_{i})為XGBoost Learing to Rank的預測目標。

由上可以發現在分子的部分定義是不一樣的，所以為能夠讓XGBoost的ndcg能夠符合競賽目標，所以此處我們會對預測目標做調整：

![V_{i,c}=2^{rel_{i}}-1](https://latex.codecogs.com/svg.latex?\Large&space;V_{i,c}=2^{rel_{i}}-1) 

移項處理:

![rel_{i}=log_{2}(V_{i,c}+1)](https://latex.codecogs.com/svg.latex?\Large&space;rel_{i}=log_{2}(V_{i,c}+1)) 

透過對上式調整，即可讓競賽與XGBoost模型評分指標能夠一致，準確度也可以再往上提升。


這次算是我第一次真正處理Learing to Rank問題，並且train出一個預測能力好的模型。其實兩年前有看過相關L2R的文獻應用在股票推薦上，那時候也有用R來實作XGBoost模型，可是效果並沒有很好，也許是那時研究還沒很透徹，所以在資料的整理及相關設定可能有問題。透過這次競賽成功的經驗，讓我對透過XGBoost使用Learing to Rank的做法更加熟悉。其實本來是還想要用深度學習相關的做法來預測看看，但因為這次在資料上處理的問題花費的時間太久，所以最後先暫緩不測試，下次有機會再來試試看!
