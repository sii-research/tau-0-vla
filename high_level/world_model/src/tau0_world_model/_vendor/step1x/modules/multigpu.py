import os
import functools
import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist
from xfuser.core.distributed import (
    init_distributed_environment,
    get_classifier_free_guidance_world_size, 
    get_classifier_free_guidance_rank,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_cfg_group
)

if os.getenv("TORCHELASTIC_RUN_ID") is not None:
    dist.init_process_group("nccl")
    init_distributed_environment(
        rank=dist.get_rank(), 
        world_size=dist.get_world_size()
    )

def parallel_transformer(pipe):
    transformer = pipe.dit
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        llm_embedding: Tensor,
        t_vec: Tensor,
        mask: Tensor,
    ):  
        txt, y = self.connector(
            llm_embedding, t_vec, mask
        )   # 
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # ---------------------------------------------------------------------
        if (
            isinstance(timesteps, torch.Tensor)
            and timesteps.ndim != 0
            and timesteps.shape[0] == img.shape[0]
        ):
            timesteps = torch.chunk(
                timesteps, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]

            y = torch.chunk(
                y, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]
        # ---------------------------------------------------------------------
        
        img = self.img_in(img) 
        vec = self.time_in(self.timestep_embedding(timesteps, 256)) 

        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt) 


        # ---------------------------------------------------------------------
        # img cfg_usp
        img = torch.chunk(
            img, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        img = torch.chunk(
            img, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        # txt cfg_usp
        txt = torch.chunk(
            txt, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        txt = torch.chunk(
            txt, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        # pe cfg_usp
        txt_ids = torch.chunk(
            txt_ids, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        txt_ids = torch.chunk(
            txt_ids, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        img_ids = torch.chunk(
            img_ids, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        img_ids = torch.chunk(
            img_ids, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        # ---------------------------------------------------------------------

        ids = torch.cat((txt_ids, img_ids), dim=1)  
        pe = self.pe_embedder(ids) 

        if not self.blocks_to_swap:
            for block in self.double_blocks:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            img = torch.cat((txt, img), 1) 
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe)
        else:
            for block_idx, block in enumerate(self.double_blocks):
                self.offloader_double.wait_for_block(block_idx)
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                self.offloader_double.submit_move_blocks(self.double_blocks, block_idx)

            img = torch.cat((txt, img), 1)

            for block_idx, block in enumerate(self.single_blocks):
                self.offloader_single.wait_for_block(block_idx)
                img = block(img, vec=vec, pe=pe)
                self.offloader_single.submit_move_blocks(self.single_blocks, block_idx)
        img = img[:, txt.shape[1] :, ...]

        if self.training and self.cpu_offload_checkpointing:
            img = img.to(self.device)
            vec = vec.to(self.device)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels) 

        # ---------------------------------------------------------------------
        img = get_sp_group().all_gather(img, dim=-2)
        img = get_cfg_group().all_gather(img, dim=0)
        # ---------------------------------------------------------------------
        
        return img

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward


def teacache_transformer(pipe):

    transformer = pipe.dit
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        llm_embedding: Tensor,
        t_vec: Tensor,
        mask: Tensor,
        idx = None,
    ): 
        txt, y = self.connector(
            llm_embedding, t_vec, mask
        )   
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)  
        vec = self.time_in(self.timestep_embedding(timesteps, 256)) 

        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)  
        ids = torch.cat((txt_ids, img_ids), dim=1)  
        pe = self.pe_embedder(ids) 

        # ---------------------- teacache ------------------------
        inp = img.clone()
        # vec_ = vec.clone()
        # img_mod1_, _ = self.double_blocks[0].img_mod(vec_)
        modulated_inp = self.double_blocks[0].img_norm1(inp)
        # modulated_inp = (1 + img_mod1_.scale) * modulated_inp + img_mod1_.shift

        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            # coefficients = [13.47372344, -35.72299611, 16.9750467, -1.43670151, 0.07216511]
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp 
        self.cnt += 1 
        if self.cnt == self.num_steps:
            self.cnt = 0           
        # ---------------------- teacache ------------------------

        # ---------------------- teacache ------------------------
        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            if not self.blocks_to_swap:
                for block in self.double_blocks:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

                img = torch.cat((txt, img), 1)
                for block in self.single_blocks:
                    img = block(img, vec=vec, pe=pe)
            else:
                for block_idx, block in enumerate(self.double_blocks):
                    self.offloader_double.wait_for_block(block_idx)
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                    self.offloader_double.submit_move_blocks(self.double_blocks, block_idx)

                img = torch.cat((txt, img), 1)

                for block_idx, block in enumerate(self.single_blocks):
                    self.offloader_single.wait_for_block(block_idx)
                    img = block(img, vec=vec, pe=pe)
                    self.offloader_single.submit_move_blocks(self.single_blocks, block_idx)
            
            img = img[:, txt.shape[1] :, ...]
            self.previous_residual = img - ori_img
        # ---------------------- teacache ------------------------
        if self.training and self.cpu_offload_checkpointing:
            img = img.to(self.device)
            vec = vec.to(self.device)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward


def parallel_teacache_transformer(pipe):

    transformer = pipe.dit
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        llm_embedding: Tensor,
        t_vec: Tensor,
        mask: Tensor,
        idx = None,
    ): 
        txt, y = self.connector(
            llm_embedding, t_vec, mask
        )  
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # ------------------------------ xDiT ---------------------------------------
        if (
            isinstance(timesteps, torch.Tensor)
            and timesteps.ndim != 0
            and timesteps.shape[0] == img.shape[0]
        ):
            timesteps = torch.chunk(
                timesteps, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]

            y = torch.chunk(
                y, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]
        # ------------------------------ xDiT ---------------------------------------
        

        img = self.img_in(img)
        vec = self.time_in(self.timestep_embedding(timesteps, 256))

        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)  

        # ------------------------------ xDiT ---------------------------------------
        # img cfg_usp
        img = torch.chunk(
            img, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        img = torch.chunk(
            img, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        # txt cfg_usp
        txt = torch.chunk(
            txt, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        txt = torch.chunk(
            txt, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        # pe cfg_usp
        txt_ids = torch.chunk(
            txt_ids, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        txt_ids = torch.chunk(
            txt_ids, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        img_ids = torch.chunk(
            img_ids, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        img_ids = torch.chunk(
            img_ids, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        # ------------------------------ xDiT ---------------------------------------

        ids = torch.cat((txt_ids, img_ids), dim=1)  
        pe = self.pe_embedder(ids) 

        # ---------------------- teacache ------------------------
        device = img.device
        if dist.is_initialized():
            tensor_cnt = torch.tensor(self.cnt, device=device)
            tensor_accum = torch.tensor(self.accumulated_rel_l1_distance, device=device)
            dist.broadcast(tensor_cnt, src=0)
            dist.broadcast(tensor_accum, src=0)
            self.cnt = tensor_cnt.item()
            self.accumulated_rel_l1_distance = tensor_accum.item()
        
        inp = img.clone()
        # vec_ = vec.clone()
        # img_mod1_, _ = self.double_blocks[0].img_mod(vec_)
        modulated_inp = self.double_blocks[0].img_norm1(inp)
        # modulated_inp = (1 + img_mod1_.scale) * modulated_inp + img_mod1_.shift
        modulated_inp = get_sp_group().all_gather(modulated_inp, dim=-2)

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            # coefficients = [13.47372344, -35.72299611, 16.9750467, -1.43670151, 0.07216511]
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            
            if dist.is_initialized():
                tensor_accum = torch.tensor(self.accumulated_rel_l1_distance, device=device)
                dist.broadcast(tensor_accum, src=0)
                self.accumulated_rel_l1_distance = tensor_accum.item()
            
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        should_calc_tensor = torch.tensor(should_calc, device=img.device)
        if dist.is_initialized():
            dist.broadcast(should_calc_tensor, src=0)
        should_calc = should_calc_tensor.item()

        self.previous_modulated_input = modulated_inp 
        self.cnt += 1 
        if self.cnt == self.num_steps:
            self.cnt = 0           

        if dist.is_initialized():
            dist.barrier()
        # ---------------------- teacache ------------------------

        # ---------------------- teacache ------------------------
        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            if not self.blocks_to_swap:
                for block in self.double_blocks:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

                img = torch.cat((txt, img), 1)
                for block in self.single_blocks:
                    img = block(img, vec=vec, pe=pe)
            else:
                for block_idx, block in enumerate(self.double_blocks):
                    self.offloader_double.wait_for_block(block_idx)
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                    self.offloader_double.submit_move_blocks(self.double_blocks, block_idx)

                img = torch.cat((txt, img), 1)

                for block_idx, block in enumerate(self.single_blocks):
                    self.offloader_single.wait_for_block(block_idx)
                    img = block(img, vec=vec, pe=pe)
                    self.offloader_single.submit_move_blocks(self.single_blocks, block_idx)
            
            img = img[:, txt.shape[1] :, ...]
            self.previous_residual = img - ori_img
        # ---------------------- teacache ------------------------
        if self.training and self.cpu_offload_checkpointing:
            img = img.to(self.device)
            vec = vec.to(self.device)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        # ---------------------------- xDiT -----------------------------------------
        img = get_sp_group().all_gather(img, dim=-2)
        img = get_cfg_group().all_gather(img, dim=0)
        # ---------------------------- xDiT -----------------------------------------

        return img

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
