import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
####add 
from omegaconf import OmegaConf
from diffsynth.models.wan_video_dit import WanModel
from torch.utils.tensorboard import SummaryWriter


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        while True:
            try:
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path)
                if self.is_i2v:
                    video, first_frame = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"text": text, "video": video, "path": path}
                break
            except:
                data_id += 1
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            pth_path = path + ".tensors.pth"
            if not os.path.exists(pth_path):
                # prompt
                prompt_emb = self.pipe.encode_prompt(text)
                # video
                video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_frame" in batch:
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
                torch.save(data, pth_path)
            else:
                print(f"File {pth_path} already exists, skipping.")

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch):
        metadata = json.load(open(dataset_path, 'r'))
        self.path = [m["video_path"] for m in metadata]
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch


    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)


    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses
    
    def ray_condition(self, K, c2w, device, flip_flag=None, H=480, W=832):
        # c2w: B, V, 4, 4
        # K: B, V, 4

        B, V = K.shape[:2]
        def custom_meshgrid(*args):
            from packaging import version as pver
            # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
            if pver.parse(torch.__version__) < pver.parse('1.10'):
                return torch.meshgrid(*args)
            else:
                return torch.meshgrid(*args, indexing='ij')

        j, i = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        )
        i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
        j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

        n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
        if n_flip > 0:
            j_flip, i_flip = custom_meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
                torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
            )
            i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            i[:, flip_flag, ...] = i_flip
            j[:, flip_flag, ...] = j_flip

        fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

        zs = torch.ones_like(i)                 # [B, V, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
        rays_o = c2w[..., :3, 3]                                        # B, V, 3
        rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
        # c2w @ dirctions
        rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
        # plucker = plucker.permute(0, 1, 4, 2, 3)
        return plucker
    
    def __getitem__(self, index):
        # Return: 
        # data['latents']: torch.Size([16, 21*2, 60, 104])
        # data['camera']: torch.Size([21, 3, 4])
        # data['prompt_emb']["context"][0]: torch.Size([512, 4096])
        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
                path_tgt = self.path[data_id]
                ### get focal length and aperture
                pattern = r'f(\d+)_aperture(\d+\.\d+)'
                match = re.search(pattern, path_tgt)
                focal_length = float(match.group(1))
                d = 23.76 / 1280
                fx=focal_length/d
                fy=focal_length/d
                cx=832/2
                cy=480/2
                #################################
                
                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")

                # load the condition latent
                match = re.search(r'cam(\d+)', path_tgt)
                tgt_idx = int(match.group(1))
                cond_idx = random.randint(1, 10)
                while cond_idx == tgt_idx:
                    cond_idx = random.randint(1, 10)
                path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                data['latents'] = torch.cat((data_tgt['latents'],data_cond['latents']),dim=1)
                data['prompt_emb'] = data_tgt['prompt_emb']
                data['image_emb'] = {}

                # load the target trajectory
                base_path = path_tgt.rsplit('/', 2)[0]
                tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                with open(tgt_camera_path, 'r') as file:
                    cam_data = json.load(file)
                multiview_c2ws = []
                cam_idx = list(range(81))[::4]
                for view_idx in [cond_idx, tgt_idx]:
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]
                        c2w[:3, 1] *= -1.
                        c2w[:3, 3] /= 100
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
                relative_poses = self.get_relative_pose(tgt_cam_params)
                pose_embedding = torch.as_tensor(relative_poses)[None] # [1, n_frame, 4, 4]

                # relative_poses = []
                # for i in range(len(tgt_cam_params)):
                #     relative_pose = self.get_relative_pose([cond_cam_params[0], tgt_cam_params[i]])
                #     relative_poses.append(torch.as_tensor(relative_pose)[1])
                # pose_embedding = torch.stack(relative_poses, dim=0)  # 21x4x4
                n_frames = pose_embedding.shape[1]
                intrinsics = np.asarray([[fx,fy,cx,cy]])
                intrinsics = torch.as_tensor(intrinsics)
                intrinsics = intrinsics.repeat(1,n_frames,1) # [1, n_frame, 4]
                # pose_embedding = pose_embedding.unsqueeze(0) # [1, n_frame, 4, 4]
                flip_flag = torch.zeros(n_frames, dtype=torch.bool, device=c2w.device)
                plucker_embedding = self.ray_condition(intrinsics, c2w, device='cpu',
                                                                  flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous() # V, 6, H, W
                data['camera'] = plucker_embedding.to(torch.bfloat16)
                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.path))
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        args,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None
    ):
        super().__init__()
        

        # load var & text_encoder
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([
            args.text_encoder_path,
            args.vae_path,
        ])
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)

        # init dit model
        print(f"        This model is initialized with extra kwargs: {args.dit_kwargs}")
        self.pipe.dit = WanModel(**args.dit_kwargs)
        
        # load pretrained model
        state_dict = load_state_dict(args.dit_path)
        state_dict_to_load = {}
        for key, value in state_dict.items():
            state_dict_to_load[key] = value
        dit_missing, dit_unexpected = self.pipe.dit.load_state_dict(state_dict_to_load, assign=True, strict=False)
        print(f'Dit Missing: {len(dit_missing)}, Dit Unexpected: {len(dit_unexpected)}')
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()

        ##### copy dit state_dict (wo cam_encoder & projector) to controlnet block
        model = self.pipe.denoising_model()
        for i in range(args.dit_kwargs.num_control_layers):
            missing, unexpected = model.control_blocks[i].load_state_dict(model.blocks[i].state_dict(), strict=False)
            print(f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')
            ### only missing projector & cam_encoder
        #######################################################

        for param in model.parameters():
            param.requires_grad_(False)
        
        for name, param in model.named_parameters():
            if name in dit_missing:
                param.requires_grad_(True)
            # if "control_blocks" in name:
            #     param.requires_grad_(True)
        
        trainable_params = 0
        train_params_names = []
        frozen_params_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                train_params_names.append(name)
            else:
                frozen_params_names.append(name)
        print(f"Total number of trainable parameters: {trainable_params}")
        print(f"Trainable parameters: {train_params_names}")
        print(f"Frozen parameters: {frozen_params_names}")
        assert len([name for name in train_params_names if "control_blocks" in name]) == len (train_params_names)
        assert len([name for name in frozen_params_names if "control_blocks" in name]) == 0

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
    
    def on_train_start(self):
        if self.global_rank == 0:
            os.makedirs(self.trainer.log_dir, exist_ok=True)
            OmegaConf.save(args, os.path.join(self.trainer.log_dir, 'config.yaml'))
            self.writer = SummaryWriter(os.path.join(self.trainer.log_dir, 'logs'))
        
    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        
        cam_emb = batch["camera"].to(self.device)
        cam_emb = cam_emb[None] # [1, V, 6, H, W]

        # Loss
        self.pipe.device = self.device
        tgt_latent_len = latents.shape[2] // 2
        source_latents = latents[:, :, tgt_latent_len:, ...]
        target_latents = latents[:, :, :tgt_latent_len, ...]

        # ================== timestep ==================
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        # ================== model input & condition ==================
        noise = torch.randn_like(target_latents)
        noisy_target_latents = self.pipe.scheduler.add_noise(target_latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(target_latents, noise, timestep).to(device=self.pipe.device)
        # model_input = torch.cat([noisy_target_latents, source_latents, render_latents], dim=2).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        model_input = noisy_target_latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        controlnet_input = noisy_target_latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        # ================== Forward & Compute loss ==================
        tgt_latent_len = training_target.shape[2]
        noise_pred = self.pipe.denoising_model()(
            model_input, controlnet_input, timestep=timestep, cam_emb=cam_emb, **prompt_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        if self.global_rank == 0:
            self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))



def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        '--config',
        type=str,
        default='train_recammaster.yaml',
        help='config file',
    )
    args = parser.parse_args()
    args = OmegaConf.load(args.config)
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        args=args,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )

    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
