# ONNX

## 1. run pth model to dump necessary data preparing for onnx model generation
```shell
tools/dist_test_pth.sh projects/configs/StreamPETR/stream_petr_r50_704_bs2_seq_428q_nui_60e_pth.py ckpts/iter_105480.pth 1 --eval=bbox
```
此脚本用于生成模型转换需要的中间变量，以.bin的形式保存，数据size写在文件名中。同时还会保存输出all_bbox_preds、all_cls_scores用于验证onnx模型的输出精度。

保存的变量有:
**1.外部输入：用于模型输入**
* 1.1 img_6x3x256x704.bin: 6张经过normalized的2d图像。
* 1.2 img2lidar_1x4224x4x4.bin: 相机内外参矩阵乘积取逆,用于2d embedding生成。
* 1.3 intrinsic_1x4224x2.bin：相机内参。
* 1.4 timestamp_1.bin: 时间戳, **float64**,由于onnx不支持float64的sin和cos计算，暂时以float64的形式送入，在模型内部强转float32.
* 1.5 ego_pose_1x4x4.bin: 当前车辆姿态矩阵。
* 1.6 ego_pose_inv_1x4x4.bin:姿态矩阵的逆。
* 1.7 prev_exists_1.bin:是否存在前一帧。

**2.上一帧结果输出&memory：用于对齐精度**
* 2.1 out_mem_embedding_1x512x256_pth.bin: self.memory_embedding
* 2.2 out_mem_timestamp_1x512x1_pth.bin: self.memory_timestamp, 保留的时间戳delta **float64**,同timestamp_1.bin
* 2.3 out_mem_egopose_1x512x4x4_pth.bin: self.memory_egopose, 累积的当前车辆姿态
* 2.4 out_mem_ref_point_1x512x3_pth.bin: self.memory_reference_point, 参考中心点
* 2.5 out_mem_velo_1x512x2_pth.bin: self.memory_velo, 速度
* 2.6 out_outs_dec_1x428x256_pth.bin: transformer输出的最后一层高维特征
* 2.7 post_mem_embedding_1x640x256_pth.bin: 经过post update的mem_embedding
* 2.8 post_mem_timestamp_1x512x1_pth.bin
* 2.9 post_mem_egopose_1x512x4x4_pth.bin
* 2.10 post_mem_ref_point_1x640x3_pth.bin
* 2.11 post_mem_velo_1x640x2_pth.bin


**3.模型内部固化参数**
—— 模型训练完成后,参数固定，不随输入变化，即可省去中间处理算子，直接固化结果。
* 3.1 coords_1x4224x64x4x1.bin
* 3.2 query_pos_1x300x256.bin
* 3.3 tgt_1x300x256.bin

## 2.run onnx generation
```shell
tools/dist_test_onnx.sh projects/configs/StreamPETR/stream_petr_r50_704_bs2_seq_428q_nui_60e_onnx.py ckpts/iter_105480.pth 1 --eval=bbox
```
原始streampetr forward函数的输入是一个大字典，包含各种有用或无用信息,部分数据格式也不支持onnx转换，不利于onnx模型生成。这里使用**PetrWrapper**对model进行重新封装，并定义好实际需要的输入。
```shell
class PetrWrapper(torch.nn.Module):
    def __init__(self, org_model):
        super(PetrWrapper, self).__init__()
        self.org_model = org_model
        self.data = None

    def set_data(self, data):
        self.data = data

    def forward(self, img=None, img2lidar=None, intrinsic=None, timestamp=None, ego_pose=None, ego_pose_inv=None,
                prev_exists=None, mem_embedding=None, mem_timestamp=None, mem_egopose=None, mem_ref_point=None, mem_velo=None):
        with torch.no_grad():
            outs = self.org_model(return_loss=False, rescale=True, **self.data)
        # for k in outs.keys():
        #     print(k)
        #     if outs[k] is None:
        #         continue
        #     print(outs[k])
        return outs

```
虽然PetrWrapper的forward函数看上去传入的参数没有发挥实际作用，实际参与推理的是**data**，但这些入参是来自data内数据的引用。
在执行torch.onnx.export(...)时，onnx转换的特性使得前向推理数据流过的算子被保存，且根据forward的输入参数确定模型的输入。而data中参与数据流的其他参数则直接作为模型参数被固化下来。
因此coords、query_pos、query_pos_delta、tgt等参数，由于与输入无关，训练完成后即固定，那么可以直接省略之间处理的算子，当这些参数保存于data，但没有出现在forward输入列表中，即会保存为模型的内部参数。



