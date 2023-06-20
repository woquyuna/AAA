import torch
import numpy as np
import os

# config
OUTPUT_PATH = "/mnt/data/userdata/hjj/streampetr_data/"

DEVICE = 'cuda:0'
first_frame = True

B, N = 1, 6
BN = B * N
H, W = 16, 44
D = 64
LEN = BN * H * W

def img2lidar_tensor_generator(data):
    # B, N = 1, 6
    # BN = B * N
    # H, W = 16, 44
    # D = 64
    # LEN = BN * H * W
    img2lidars = data['lidar2img'][0].data[0][0].to(DEVICE).inverse()
    img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H * W, D, 1, 1).view(B, LEN, D, 4, 4)
    img2lidars = torch.tensor(img2lidars, dtype=torch.float32)
    return img2lidars

def intrinsics_tensor_generator(data):
    # B, N = 1, 6
    # BN = B * N
    # H, W = 16, 44
    # D = 64
    # LEN = BN * H * W
    intrinsic = torch.stack([data['intrinsics'][0].data[0][0][..., 0, 0], data['intrinsics'][0].data[0][0][..., 1, 1]], dim=-1)
    intrinsic = torch.abs(intrinsic) / 1e3
    intrinsic = intrinsic.repeat(1, H * W, 1).view(B, -1, 2)
    intrinsic = torch.tensor(intrinsic, dtype=torch.float32).to(DEVICE)
    return intrinsic

def mem_tensor_generator(first_frame, model_cfg):
    B = 1
    # TODO:get from config?
    memory_len = 512
    num_propagated = 128
    embed_dims = 256
    mems = []
    if first_frame:
         mems.append(torch.zeros(B, memory_len + num_propagated, embed_dims).to(DEVICE))             # embedding
         mems.append(torch.zeros(B, memory_len + num_propagated, 1).to(torch.float64).to(DEVICE))    # timestamp TODO:float64 - float32
         mems.append(torch.zeros(B, memory_len + num_propagated, 4, 4).to(DEVICE))                   # egopose
         mems.append(torch.zeros(B, memory_len + num_propagated, 3).to(DEVICE))                      # reference_point
         mems.append(torch.zeros(B, memory_len + num_propagated, 2).to(DEVICE))                      # velo
    else:
        # load from last run
        pass
    # mems[0].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/in_mem_embedding_1x512x256.bin')
    # mems[1].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/in_mem_timestamp_1x512x1.bin')
    # mems[2].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/in_mem_egopose_1x512x4x4.bin')
    # mems[3].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/in_mem_ref_point_1x512x3.bin')
    # mems[4].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/in_mem_velo_1x512x2.bin')
    return mems

class nuscenceData:
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self.prev_scene_token = None

    # generate continuous frame from valset first frame
    def generate_input_data_from_pth(self, model_cfg=None, num_frame=3):
        for i, data in enumerate(self._data_loader):
            if i >= num_frame:
                break
            print("*" * 25, "dump frame {} input".format(i), "*" * 25)

            img_metas = data['img_metas'][0].data[0][0]
            data['img_metas'][0] = []
            data['img_metas'][0].append(img_metas)

            dir_path = OUTPUT_PATH + '/' + data['img_metas'][0][0]['scene_token'] + '_frame{}'.format(str(i).zfill(2))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            # input from dataset
            img_metas = [img_metas]
            if img_metas[0]['scene_token'] != self.prev_scene_token:
                self.prev_scene_token = img_metas[0]['scene_token']
                data['prev_exists'] = [torch.zeros(1, dtype=torch.float32).to(DEVICE)]
            else:
                data['prev_exists'] = [torch.ones(1, dtype=torch.float32).to(DEVICE)]
            data['prev_exists'][0].cpu().detach().numpy().tofile(dir_path + '/prev_exists_1.bin')
            img = data['img'][0].data[0]
            B, N, C, H, W = img.size()
            img = img.to(DEVICE)
            data['img'][0] = img.reshape(B * N, C, H, W)
            print("img:", data['img'][0].shape)
            data['img'][0].cpu().detach().numpy().tofile(dir_path + '/img_6x3x256x704.bin')

            data['img2lidar'] = [img2lidar_tensor_generator(data)]
            print("img2lidar:", data['img2lidar'][0].shape)
            data['img2lidar'][0].cpu().detach().numpy().tofile(dir_path + '/img2lidar_1x4224x64x4x4.bin')

            data['intrinsic'] = [intrinsics_tensor_generator(data)]
            print("intrinsic:", data['intrinsic'][0].shape)
            data['intrinsic'][0].cpu().detach().numpy().tofile(dir_path + '/intrinsic_1x4224x2.bin')

            data['timestamp'] = [data['timestamp'][0].data[0][0].unsqueeze(0).to(DEVICE)]     #float64
            print("timestamp:", data['timestamp'][0].shape)
            data['timestamp'][0].cpu().detach().numpy().tofile(dir_path + '/timestamp_1.bin')

            data['ego_pose'] = [data['ego_pose'][0].data[0][0].unsqueeze(0).to(DEVICE)]       #for post update memory
            print("ego_pose:", data['ego_pose'][0].shape)
            data['ego_pose'][0].cpu().detach().numpy().tofile(dir_path + '/ego_pose_1x4x4.bin')

            data['ego_pose_inv'] = [data['ego_pose_inv'][0].data[0][0].unsqueeze(0).to(DEVICE)]
            print("ego_pose_inv:", data['ego_pose_inv'][0].shape)
            data['ego_pose_inv'][0].cpu().detach().numpy().tofile(dir_path + '/ego_pose_inv_1x4x4.bin')
        return data

    def generate_constant_memory_from_pth(self, model_cfg=None, num_frame=3):
        datas = []
        for i, data in enumerate(self._data_loader):
            if i >= num_frame:
                break

            img_metas = data['img_metas'][0].data[0][0]
            data['img_metas'][0] = []
            data['img_metas'][0].append(img_metas)

            # constant path
            c_path = OUTPUT_PATH + '/constant'
            if not os.path.exists(c_path):
                os.mkdir(c_path)
            data['c_path'] = [c_path]
            # memory path
            dir_path = OUTPUT_PATH + '/' + data['img_metas'][0][0]['scene_token'] + '_frame{}'.format(str(i).zfill(2))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            data['dir_path'] = [dir_path]

            img = data['img'][0].data[0]
            B, N, C, H, W = img.size()
            img = img.to(DEVICE)
            data['img'][0] = img.reshape(B * N, C, H, W)
            data['timestamp'] = [data['timestamp'][0].data[0][0].unsqueeze(0).to(DEVICE)]
            data['ego_pose'] = [data['ego_pose'][0].data[0][0].unsqueeze(0).to(DEVICE)]
            data['ego_pose_inv'] = [data['ego_pose_inv'][0].data[0][0].unsqueeze(0).to(DEVICE)]
            datas.append(data)

        return datas

    def load_data(self, model_cfg):
        for data in self._data_loader:
            img = data['img'][0].data[0]
            B, N, C, H, W = img.size()
            # input from dataset
            img = img.to(DEVICE)
            data['img'][0] = img.reshape(B*N, C, H, W)
            # data['img'][0].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/img_6x3x256x704.bin')
            data['img2lidar'] = [img2lidar_tensor_generator(data)]
            data['intrinsic'] = [intrinsics_tensor_generator(data)]
            # data['intrinsic'][0].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/intrinsic_1x4224x2.bin')
            data['timestamp'] = [data['timestamp'][0].data[0][0].unsqueeze(0).to(torch.float64).to(DEVICE)] #TODO: input is float64, but would convert to float32 in model
            # data['timestamp'][0].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/timestamp_1.bin')
            data['ego_pose'] = [data['ego_pose'][0].data[0][0].unsqueeze(0).to(DEVICE)]
            data['ego_pose_inv'] = [data['ego_pose_inv'][0].data[0][0].unsqueeze(0).to(DEVICE)]
            # data['ego_pose_inv'][0].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/ego_pose_inv_1x4x4.bin')
            data['prev_exists'] = [torch.zeros(1, dtype=torch.float32).to(DEVICE)]
            # data['prev_exists'][0].cpu().detach().numpy().tofile('/mnt/data/userdata/hjj/streampetr_data/prev_exists_1.bin')
            # input from last run
            mems = mem_tensor_generator(first_frame, model_cfg)
            data['mem_embedding'] = [mems[0]]
            data['mem_timestamp'] = [mems[1]]   #TODO:float64 - float32
            data['mem_egopose'] = [mems[2]]
            data['mem_ref_point'] = [mems[3]]
            data['mem_velo'] = [mems[4]]
            # constant params
            data['coords'] = [torch.tensor(
                np.fromfile(OUTPUT_PATH + "/constant/coords_1x4224x64x4x1.bin", dtype=np.float32),
                dtype=torch.float32)]
            data['coords'][0] = data['coords'][0].to(DEVICE).reshape(1, 4224, 64, 4, 1)
            data['query_pos'] = [torch.tensor(
                np.fromfile(OUTPUT_PATH + "/constant/query_pos_1x300x256.bin", dtype=np.float32),
                dtype=torch.float32)]
            data['query_pos'][0] = data['query_pos'][0].to(DEVICE).reshape(1, 300, 256)
            # data['query_pos_delta'] = [torch.tensor(
            #     np.fromfile(OUTPUT_PATH + "/constant/query_pos_delta_1x300x256.bin", dtype=np.float32),
            #     dtype=torch.float32)]
            # data['query_pos_delta'][0] = data['query_pos_delta'][0].to(DEVICE).reshape(1, 300, 256)
            # data['query_pos'][0] += data['query_pos_delta'][0]
            # print("query_pos + query_pos_delta:")
            # print(data['query_pos'][0])
            # print(data['query_pos_delta'][0])
            # print(data['query_pos'][0] + data['query_pos_delta'][0])
            data['tgt'] = [torch.tensor(
                np.fromfile(OUTPUT_PATH + "/constant/tgt_1x300x256.bin", dtype=np.float32),
                dtype=torch.float32)]
            data['tgt'][0] = data['tgt'][0].to(DEVICE).reshape(1, 300, 256)

            img_metas = data['img_metas'][0].data[0][0]
            data['img_metas'][0] = []
            data['img_metas'][0].append(img_metas)

            # save input
            return data, data['img'][0], data['img2lidar'][0], data['intrinsic'][0], \
                   data['timestamp'][0], data['ego_pose'][0], data['ego_pose_inv'][0], \
                   data['prev_exists'][0], \
                   data['mem_embedding'][0], data['mem_timestamp'][0], data['mem_egopose'][0], \
                   data['mem_ref_point'][0], data['mem_velo'][0]


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
        return outs

def get_onnx_model(model,
                   img, img2lidar, intrinsic, timestamp, ego_pose, ego_pose_inv,
                   prev_exists, mem_embedding, mem_timestamp, mem_egopose, mem_ref_point, mem_velo,
                   out_path, device):
    model.eval()
    model = model.to(device)
    img_tensor = img.to(device)
    img2lidar = img2lidar.to(device)
    intrinsic = intrinsic.to(device)
    timestamp = timestamp.to(device)
    ego_pose = ego_pose.to(device)
    ego_pose_inv = ego_pose_inv.to(device)
    prev_exists = prev_exists.to(device)
    mem_embedding = mem_embedding.to(device)
    mem_timestamp = mem_timestamp.to(device)
    mem_egopose = mem_egopose.to(device)
    mem_ref_point = mem_ref_point.to(device)
    mem_velo = mem_velo.to(device)

    torch.onnx.export(
        model,
        tuple([img_tensor, img2lidar, intrinsic, timestamp, ego_pose, ego_pose_inv,
               prev_exists, mem_embedding, mem_timestamp, mem_egopose, mem_ref_point, mem_velo]),
        out_path,
        export_params=True,
        opset_version=11,
        input_names=["img", "img2lidar", "intrinsic", "timestamp", "ego_pose", "ego_pose_inv",
                     "prev_exists", "in_mem_embedding", "in_mem_timestamp", "in_mem_egopose",
                     "in_mem_ref_point", "in_mem_velo"],
        output_names=["all_cls_scores", "all_bbox_preds",
                      "mem_embedding", "mem_timestamp", "mem_egopose", "mem_ref_point", "mem_velo",
                      "outs_dec"],
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        verbose=True
    )
    print("export onnx success:", out_path)
