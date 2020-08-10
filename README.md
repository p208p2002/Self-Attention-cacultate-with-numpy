# Self Attention cacultate with numpy

### Attention 公式
![file](https://github.com/p208p2002/Self-Attention-cacultate-with-numpy/blob/master/attention.png?raw=true)

公式中的(Q)uerys, (K)eys, (V)alues可以視為一種representations,他們各自對應一組權重，模型的目的就是去學習權重

而√dk則是scaling factor, Q或K的維度

所以更詳細的表示:
```
Q = Q * Q_Weight
K = K * K_Weight
V = V * V_Weight
```

在Self-Attention中 Q=K=V, 僅對應的權重不同

### Self-Attention Score
#### 輸入
inputs 可以視為一句話經過encode的結果，每一個input則為其單詞
```python
input_1 = np.array([1, 0, 1, 0], dtype='float32')
input_2 = np.array([0, 2, 0, 2], dtype='float32')
input_3 = np.array([1, 1, 1, 1], dtype='float32')
inputs = np.vstack([input_1,input_2,input_3])
'''
inputs
[[1. 0. 1. 0.]
 [0. 2. 0. 2.]
 [1. 1. 1. 1.]]
'''
```
#### 權重
```python
wk = np.array([[0, 0, 1],
             [1, 1, 0],
             [0, 1, 0],
             [1, 1, 0]], dtype='float32')
wq = np.array([[1, 0, 1],
             [1, 0, 0],
             [0, 0, 1],
             [0, 1, 1]], dtype='float32')
wv = np.array([[0, 2, 0],
             [0, 3, 0],
             [1, 0, 3],
             [1, 1, 0]], dtype='float32')
```

#### Calculate QKV representations
```python
# Q=K=V=inputs
query_representations = inputs.dot(wq)
key_representations = inputs.dot(wk)
value_representations = inputs.dot(wv)
query_representations_dim = np.array([float(query_representations.shape[0])**0.5],dtype='float32')
'''
[[1. 0. 2.]
 [2. 2. 2.]
 [2. 1. 3.]] 

[[0. 1. 1.]
 [4. 4. 0.]
 [2. 3. 1.]] 

[[1. 2. 3.]
 [2. 8. 0.]
 [2. 6. 3.]] 

[1.7320508]
'''
```
#### Calculate attention scores
```python
softmax(np.divide(query_representations.dot(key_representations.transpose()),query_representations_dim),axis=1)\
    .dot(value_representations)
'''
array([[1.8638741 , 6.3193707 , 1.7041886 ],
       [1.9991105 , 7.8141265 , 0.27347228],
       [1.9925548 , 7.479635  , 0.73587704]], dtype=float32)
'''
```

#### Jupyter Notebook Sample Code
[Self-Attention-Caculate.ipynb](https://github.com/p208p2002/Self-Attention-cacultate-with-numpy/blob/master/Self-Attention-Caculate.ipynb)


### Refs
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [illustrated-self-attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
- [Attention-PyTorch](https://github.com/EvilPsyCHo/Attention-PyTorch)
