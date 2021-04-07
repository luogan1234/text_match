# text_match
全球人工智能技术创新大赛（赛道三: 小布助手对话短文本语义匹配）

https://tianchi.aliyun.com/competition/entrance/531851/introduction

基于BERT finetune和logistic regression的ensemble方法

第一阶段：基于bert-base-chinese的预训练参数，固定除了bert.embeddings.word_embeddings之外的部分，用训练+测试集做mask位置的预测，专门学习token embedding。

第二阶段：在上一步的基础上，用多个不同的seed生成的随机划分，[CLS]预测句子二分类是否相关，mask位置继续做预测，进行finetune。

第三阶段：提取[CLS]的隐状态向量，拼接上tf-idf、bm25等各种向量，用LR预测结果，然后ensemble。