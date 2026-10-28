# 问题一预测结果

## 评估方式

- 数据：250款啤酒，输入为1.csv的231个数值特征，预测4.csv的50个专家感官指标；未使用7.csv。
- 划分：seed=0，按啤酒风格分层，175款训练、75款留出评价。
- 调参：175款训练数据内部做五折交叉验证，各模型使用相同折划分；先恢复原论文的训练样本顺序，再使用不打乱的五折。
- 搜索规模：GBR完整穷举648组、RF完整穷举2304组；TabPFN搜索18组，多任务网络搜索48组。按五折平均R²选参数，再用全部175款训练、评价75款留出样本。
- 指标：每个感官指标分别计算，再对50个指标取平均；交叉验证R²再对五折取平均。R²越大越好，MAE、RMSE越小越好。评分沿用4.csv处理后的尺度。

## 结果

| 模型 | 五折平均R² | 留出集R² | 留出集MAE | 留出集RMSE |
| --- | ---: | ---: | ---: | ---: |
| 梯度提升回归（GBR） | 0.1878 | 0.2157 | 0.3415 | 0.4875 |
| 随机森林（RF） | 0.1861 | 0.2175 | 0.3497 | 0.4895 |
| TabPFN 3.5 | 0.2055 | 0.2363 | 0.3374 | 0.4762 |
| 多任务神经网络 | 0.1599 | 0.1699 | 0.3636 | 0.5084 |

本轮TabPFN的五折平均R²和留出集R²最高。

## 最优参数

| 模型 | 参数 |
| --- | --- |
| GBR | learning_rate=0.02，loss=absolute_error，max_depth=6，n_estimators=500，subsample=0.6 |
| RF | max_depth=8，max_features=122，min_impurity_decrease=0，n_estimators=200 |
| TabPFN | n_estimators=4，softmax_temperature=1.0，average_before_softmax=True |
| 多任务网络 | hidden=(64,16)，dropout=0.3，learning_rate=0.003，weight_decay=0.1，input_clip=5；3个成员集成 |
