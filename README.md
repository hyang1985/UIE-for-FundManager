# 知识图谱解析基金经理简历
## 实现步骤
### 1.1对doccano标注的数据集进行序列化处理
python doccano.py     --doccano_file ./data/doccano_ext.jsonl     --task_type ext     --save_dir ./data     --splits 0.8 0.2 0

注：doccano_ext.jsonl 由doccano的Sequence Labeling任务实现

### 1.2对模型进行单卡微调
python finetune.py     --train_path ./data/train.txt     --dev_path ./data/dev.txt     --save_dir ./checkpoint     --learning_rate 1e-5     --batch_size 16     --max_seq_len 64     --num_epochs 30     --model uie-base     --seed 1000     --logging_steps 10     --valid_steps 100     --device gpu

注：需要使用gpu进行调优，调优后的最佳模型存放在checkpoint目录中，可供后续调用

### 1.3 模型验证语句
包含所有schema
python evaluate.py     --model_path ./checkpoint/model_best     --test_path ./data/dev.txt     --batch_size 16     --max_seq_len 64
结果：
[2022-07-30 23:04:43,570] [    INFO] - -----------------------------
[2022-07-30 23:04:43,571] [    INFO] - Class Name: all_classes
[2022-07-30 23:04:43,571] [    INFO] - Evaluation Precision: 0.61905 | Recall: 0.74286 | F1: 0.67532
分开展示各个schema
python evaluate.py --model_path ./checkpoint/model_best --test_path ./data/dev.txt --debug
结果：
[2022-07-30 23:06:58,963] [    INFO] - -----------------------------
[2022-07-30 23:06:58,964] [    INFO] - Class Name: 大学
[2022-07-30 23:06:58,965] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-07-30 23:06:58,985] [    INFO] - -----------------------------
[2022-07-30 23:06:58,986] [    INFO] - Class Name: 专业
[2022-07-30 23:06:58,986] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-07-30 23:06:59,014] [    INFO] - -----------------------------
[2022-07-30 23:06:59,015] [    INFO] - Class Name: 学历
[2022-07-30 23:06:59,015] [    INFO] - Evaluation Precision: 0.50000 | Recall: 0.50000 | F1: 0.50000
[2022-07-30 23:06:59,043] [    INFO] - -----------------------------
[2022-07-30 23:06:59,044] [    INFO] - Class Name: 公司
[2022-07-30 23:06:59,044] [    INFO] - Evaluation Precision: 0.90909 | Recall: 1.00000 | F1: 0.95238
[2022-07-30 23:06:59,074] [    INFO] - -----------------------------
[2022-07-30 23:06:59,075] [    INFO] - Class Name: 职位
[2022-07-30 23:06:59,076] [    INFO] - Evaluation Precision: 0.90000 | Recall: 0.64286 | F1: 0.75000
[2022-07-30 23:06:59,106] [    INFO] - -----------------------------
[2022-07-30 23:06:59,107] [    INFO] - Class Name: 部门
[2022-07-30 23:06:59,107] [    INFO] - Evaluation Precision: 1.00000 | Recall: 0.66667 | F1: 0.80000
[2022-07-30 23:06:59,135] [    INFO] - -----------------------------
[2022-07-30 23:06:59,136] [    INFO] - Class Name: 基金名称
[2022-07-30 23:06:59,136] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000