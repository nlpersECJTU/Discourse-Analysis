Paper List for Discourse

## Discourse Relation Recognition
- A Regularization Approach for Incorporating Event Knowledge and Coreference Relations into Neural Discourse Parsing (EMNLP 2019). [Paper](https://www.aclweb.org/anthology/D19-1295).[source code (2018 Dai and Huang)](https://github.com/ZeyuDai/paragraph-level_implicit_discourse_relation_classification)
<br>简要说明：本文在神经网络模型中结合外部知识和共指关系缓解数据稀疏问题，并进一步提升隐式篇章关系识别任务的性能。由于外部知识和共指关系并不总能被应用于特定的文本中，于是作者提出一种正则化的方法，即在上下文语境中紧密结合外部知识和共指关系信息生成词向量语义表示，同时，在总的代价函数中增加一个正则化参数平衡上下文信息与外部知识及共指关系之间的注意力分布。实验结果证明本文方法是有效的。<br>
- Acquiring Annotated Data with Cross-lingual Explicitation for Implicit Discourse Relation Classification (NAACL 2019). [Paper](http://aclweb.org/anthology/W19-2703)
<br>简要说明：本文对英文训练数据先进行预处理，然后翻译成三种语言（French，German，Czech），以显式化源数据的隐式篇章关系，接着通过PDTB式篇章分析器在三种语言中选择要翻译回英文的样本数据，最后使用定义好的规则选择样本数据并加入训练数据集，以缓解训练数据集的稀疏问题并提升隐式篇章关系模型的识别性能。 <br>
- Tree Framework With BERT Word Embedding for the Recognition of Chinese Implicit Discourse Relations (IEEE Access 2020). [Paper](https://ieeexplore.ieee.org/document/9178269/)
<br>简要说明：本文使用基于BERT预训练模型的树结构篇章关系框架在中文语料HIT-CDTB上提升中文隐式篇章关系识别的性能。<br>

- Using a Penalty-based Loss Re-estimation Method to Improve Implicit Discourse Relation Classification (Coling 2020). [Paper](https://www.aclweb.org/anthology/2020.coling-main.132.pdf)
<br>简要说明：由于attention机制学习重要特征表示时会同时关注不重要词的信息，因此在提升重要词的特征表示信息学习时，本文在attention机制中使用惩罚损失函数加强相关语义信息的学习，提升隐式篇章关系识别效果。 <br>

- Interactively-Propagative Attention Learning for Implicit Discourse Relation Recognition (Coling 2020). [Paper](https://www.aclweb.org/anthology/2020.coling-main.282.pdf)
<br>简要说明：本文发现self-attention与interactive attention两种attention机制之间可以共享重要特征信息，于是在两个attention机制之间以交互形式建立一种传递注意力信息学习的模型，提升隐式篇章关系识别性能。<br>

- Linguistic Properties Matter for Implicit Discourse Relation Recognition: Combining Semantic Interaction, Topic Continuity and Attribution (AAAI 2018). [Paper]()

- Deep Enhanced Representation for Implicit Discourse Relation Recognition (Coling 2018). [Paper](https://aclweb.org/anthology/papers/C/C18/C18-1048/), [Code](https://github.com/hxbai/Deep_Enhanced_Repr_for_IDRR)
- A Knowledge-Augmented Neural Network Model for Implicit Discourse Relation Classification (Coling 2018). [Paper](https://aclweb.org/anthology/papers/C/C18/C18-1049/)
- Implicit Discourse Relation Recognition using Neural Tensor Network with Interactive Attention and Sparse Learning (Coling 2018). [Paper]()
- Using Active Learning to Expand Training Data for Implicit Discourse Relation Recognition (EMNLP 2018). [Paper]()
- Improving Implicit Discourse Relation Classification by Modeling Inter-dependencies of Discourse Units in a Paragraph (NAACL 2018). [Paper](https://aclweb.org/anthology/papers/N/N18/N18-1013/), [Code](https://github.com/ZeyuDai/paragraph-level_implicit_discourse_relation_classification)
- Adversarial Connective-exploiting Networks for Implicit Discourse Relation Classification (ACL 2017). [Paper](https://aclweb.org/anthology/papers/P/P17/P17-1093/), [Code](https://github.com/qkaren/Adversarial-Network-for-Discourse-ACL2017)
- Improving Implicit Discourse Relation Recognition with Discourse-specific Word Embeddings (ACL 2017). [Paper](https://aclweb.org/anthology/papers/P/P17/P17-2042/)
- Multi-task Attention-based Neural Networks for Implicit Discourse Relationship Representation and Identification (EMNLP 2017)]. [Paper]()
- SWIM: A Simple Word Interaction Model for Implicit Discourse Relation Recognition (IJCAI 2017). [Paper]()
- Implicit Discourse Relation Classification via Multi-Task Neural Networks (AAAI 2016). [Paper]()
- Bilingually-constrained Synthetic Data for Implicit Discourse Relation Recognition (EMNLP 2016). [Paper]()
- Shallow Convolutional Neural Network for Implicit Discourse Relation Recognition (EMNLP 2015). [Paper]()
## Tree Structure
- A Unified Linear-Time Framework for Sentence-Level Discourse Parsing (ACL 2019). [Paper]()
- Transition-based Neural RST Parsing with Implicit Syntax Features (Coling 2018). [Paper](https://aclweb.org/anthology/papers/C/C18/C18-1047/)
- A Two-Stage Parsing Method for Text-Level Discourse Analysis (ACL 2017). [Paper](http://aclweb.org/anthology/P17-2029)
## EDU Segment 
- SegBot: A Generic Neural Text Segmentation Model with Pointer Network (IJCAI 2018). [Paper]()
## Nuclearity in Chinese Discourse
- 基于门控记忆网络的汉语篇章主次关系识别方法 (中文信息学报 2019). [Paper]()
- Employing Text Matching Network to Recognise Nuclearity in Chinese Discourse (Coling 2018). [Paper](http://www.aclweb.org/anthology/C18-1044)
- 自然语言处理中的篇章主次关系研究 (计算机学报 2017). [Paper]()
## Data
To be continued ...
