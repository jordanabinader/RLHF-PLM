from __future__ import annotations

import argparse
import copy
import gc
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.optim import AdamW

from dataset import loading_dataset
from mlp import MLP
from reward import reward_amp_cls
from utils import clean_sequences, load_esm, load_pretrained_progen_model

# Import wandb (optional)
try:
    import wandb
except ImportError:
    wandb = None

# Add personalization imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from personalization.unified_property_fn import create_unified_property_function
from personalization.personas import get_persona, list_personas
from personalization.hybrid_reward import create_hybrid_reward_fn
from personalization.user_conditioned_policy import UserConditionedPolicyWrapper

def setup_distributed(rank, world_size, port="12355"):
    print(f"[Rank {rank}] Setting up distributed with world_size={world_size}, port={port}", flush=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    print(f"[Rank {rank}] Initializing process group...", flush=True)
    # Add timeout to prevent hanging
    import datetime
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60)
    )
    print(f"[Rank {rank}] Process group initialized", flush=True)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


@dataclass
class TrainingConfig:
    tracker_project_name: str = "ampgen_grpo"
    exp_name: str = "grpo_mix_ddp"
    steps: int = 100
    batch_size: int = 16 * 8
    epochs: int = 30
    beta: float = 0.1
    kl_clip: bool = True
    kl_clip_value: float = 0.0
    ranking_loss_weight: float = 0.1
    entropy_weight: float = 0.01
    advantage_normalization: bool = True
    group_size: int = 4
    num_candidates: int = 8
    lr: float = 2e-5
    save_every: int = 50
    max_new_tokens: int = 48
    max_sequence_length: int = 50
    top_p: float = 1.0
    top_k: int = 20
    temperature: float = 0.9
    world_size: int = torch.cuda.device_count()
    port: str = "12359"
    device: str = "cuda"
    prompt: str = "<|bos|>"
    prompt_file: Path | None = None
    tokenizer_path: Path | None = None
    base_model_path: Path | None = None
    lora_checkpoint: Path | None = None
    classifier_checkpoint: Path | None = None
    output_dir: Path = Path("grpo_runs")
    esm_mode: str = "8M"
    reward_margin_threshold: float = 0.01
    use_wandb: bool = True
    wandb_entity: str | None = None
    seed: int = 913
    # Personalization fields
    use_personalization: bool = False
    persona_name: str = "BalancedDesigner"
    persona_cycle_mode: str = "random"  # "random", "round_robin", or "single"
    toxicity_checkpoint: Path | None = None
    stability_checkpoint: Path | None = None
    reward_penalty: float = -10.0
    min_charge: float = 0.0


CFG = TrainingConfig()


def parse_args() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Run GRPO training for AMP design.")
    parser.add_argument("--base-model-path", type=Path, required=True, help="Path to the base ProGen2 checkpoint.")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Path to the tokenizer directory.")
    parser.add_argument("--lora-checkpoint", type=Path, help="Optional LoRA checkpoint.")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Classifier checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir, help="Directory to store checkpoints.")
    parser.add_argument("--steps", type=int, default=defaults.steps, help="Number of steps.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Global batch size.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=defaults.beta, help="KL penalty weight.")
    parser.add_argument("--group-size", type=int, default=defaults.group_size, help="Samples per GRPO group.")
    parser.add_argument("--num-candidates", type=int, default=defaults.num_candidates, help="Candidates per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens, help="Max tokens to generate.")
    parser.add_argument("--max-sequence-length", type=int, default=defaults.max_sequence_length, help="Prompt trim length.")
    parser.add_argument("--temperature", type=float, default=defaults.temperature, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=defaults.top_p, help="Top-p sampling value.")
    parser.add_argument("--top-k", type=int, default=defaults.top_k, help="Top-k sampling value.")
    parser.add_argument("--world-size", type=int, help="Number of GPUs to use.")
    parser.add_argument("--device", type=str, default=defaults.device, help="Device template (e.g., cuda).")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    parser.add_argument("--prompt", type=str, default=defaults.prompt, help="Default generation prompt.")
    parser.add_argument("--prompt-file", type=Path, help="Optional file with prompts.")
    parser.add_argument("--esm-mode", type=str, default=defaults.esm_mode, choices=["8M", "650M"], help="ESM model size.")
    parser.add_argument("--reward-margin-threshold", type=float, default=defaults.reward_margin_threshold, help="Minimum reward margin to keep a group.")
    parser.add_argument("--save-every", type=int, default=defaults.save_every, help="Save checkpoint every N steps.")
    parser.add_argument("--tracker-project-name", type=str, default=defaults.tracker_project_name, help="wandb project name.")
    parser.add_argument("--exp-name", type=str, default=defaults.exp_name, help="wandb experiment name.")
    parser.add_argument("--wandb-entity", type=str, help="Optional wandb entity.")
    parser.add_argument("--port", type=str, default=defaults.port, help="Distributed master port.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    # Personalization arguments
    parser.add_argument("--use-personalization", action="store_true", help="Enable user-conditioned personalization")
    parser.add_argument("--persona-name", type=str, default=defaults.persona_name, help="Persona to use")
    parser.add_argument("--persona-cycle-mode", type=str, choices=["single", "random", "round_robin"], default=defaults.persona_cycle_mode, help="How to cycle through personas")
    parser.add_argument("--toxicity-checkpoint", type=Path, help="Path to toxicity head checkpoint")
    parser.add_argument("--stability-checkpoint", type=Path, help="Path to stability head checkpoint")
    parser.add_argument("--reward-penalty", type=float, default=defaults.reward_penalty, help="Penalty for invalid sequences")
    parser.add_argument("--min-charge", type=float, default=defaults.min_charge, help="Minimum net charge for validity")
    args = parser.parse_args()

    namespace = vars(args)
    for field in fields(CFG):
        if field.name == "use_wandb":
            continue
        if field.name == "world_size" and namespace.get(field.name) is None:
            continue
        if field.name in namespace:
            setattr(CFG, field.name, namespace[field.name])
    CFG.world_size = CFG.world_size or torch.cuda.device_count()
    CFG.use_wandb = not args.no_wandb


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

class DistributedUltraLowMemoryGRPOTrainer:
    def __init__(self, policy, tokenizer, device, rank, world_size, use_user_conditioning=False):
        print(f"[Trainer __init__] Rank {rank} starting initialization...", flush=True)
        self.dev = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.tok = tokenizer
        
        # Wrap policy with user conditioning if enabled
        if use_user_conditioning:
            if self.is_main_process:
                print("Wrapping policy with user conditioning...", flush=True)
            policy = UserConditionedPolicyWrapper(policy)
            if self.is_main_process:
                print("Policy wrapped successfully", flush=True)
        
        if self.is_main_process:
            print(f"Moving policy to device {device}...", flush=True)
        self.policy = policy.to(device)
        if self.is_main_process:
            print(f"Wrapping with DDP...", flush=True)
        self.policy = DDP(self.policy, device_ids=[rank], find_unused_parameters=False)
        if self.is_main_process:
            print(f"DDP wrapper applied", flush=True)
            print(f"Creating reference model...", flush=True)
        self.ref_model = copy.deepcopy(self.policy.module if not use_user_conditioning else self.policy.module.base_policy).to(self.dev).half().eval()
        if self.is_main_process:
            print(f"Reference model created", flush=True)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        total_trainable_params = sum(p.numel() for p in trainable_params)
        if self.is_main_process:
            print(f"Trainable parameter count: {total_trainable_params:,}")
        self.opt = AdamW(
            trainable_params, 
            lr=CFG.lr, 
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        self.step = 0

    def _get_cache_key(self, query, response):
        query_info = f"{len(query)}_{query[0].item() if len(query) > 0 else 0}"
        response_info = f"{len(response)}_{response[0].item() if len(response) > 0 else 0}"
        return f"{query_info}_{response_info}"

    def _logits(self, out):
        return out[0] if isinstance(out, tuple) else out.logits

    def truncate_sequences(self, sequences):
        return [seq[-CFG.max_sequence_length:] if len(seq) > CFG.max_sequence_length else seq 
                for seq in sequences]

    def get_ref_logprobs_no_copy(self, query, response):
        with torch.no_grad():
            input_ids = torch.cat([query, response]).unsqueeze(0)
            out = self.ref_model(input_ids.cpu(), use_cache=False)
            logits = out[0] if isinstance(out, tuple) else out.logits
            ref_logits = logits[0, len(query):]
            logprobs = F.log_softmax(ref_logits, dim=-1)
            tokens = response[:ref_logits.size(0)].unsqueeze(1).cpu()
            token_logprobs = logprobs.gather(1, tokens).squeeze(1)
            return token_logprobs.to(self.dev)

    def policy_logprobs_efficient(self, query, response):
        query = query.clone().detach().to(self.dev)
        response = response.clone().detach().to(self.dev)
        input_ids = torch.cat([query, response]).unsqueeze(0)
        logits = F.log_softmax(self._logits(self.policy(input_ids, use_cache=False))[0][len(query):], dim=-1)
        length = min(len(response), logits.size(0))
        tokens = response[:length].unsqueeze(1)
        return logits[:length].gather(1, tokens).squeeze(1).sum()

    def grpo_loss_memory_minimal(self, group_data):
        query = group_data["query"].to(self.dev)
        candidates = [cand.to(self.dev) for cand in group_data["candidates"]]
        rewards    = torch.tensor(group_data["rewards"], device=self.dev, dtype=torch.float32)

        if len(candidates) < 2:
            return torch.tensor(0.0, device=self.dev, requires_grad=True)

        seqs = [torch.cat([query, cand], dim=0) for cand in candidates]
        lengths = [s.size(0) for s in seqs]

        padded = pad_sequence(seqs, batch_first=True,
                            padding_value=self.tok.eos_token_id).to(self.dev)
        policy_out = self.policy(padded, use_cache=False)
        logits     = policy_out.logits if not isinstance(policy_out, tuple) else policy_out[0]

        policy_logprobs = []
        for i, cand in enumerate(candidates):
            resp_len = cand.size(0)
            start    = lengths[i] - resp_len
            slice_logits = logits[i, start:start+resp_len]
            lp = F.log_softmax(slice_logits, dim=-1)
            token_lp = lp.gather(1, cand.unsqueeze(1)).squeeze(1)
            policy_logprobs.append(token_lp.sum())
        policy_logprobs = torch.stack(policy_logprobs)

        with torch.no_grad(), autocast(device_type="cuda"):
            ref_out    = self.ref_model(padded, use_cache=False)
            ref_logits = ref_out.logits if not isinstance(ref_out, tuple) else ref_out[0]

        ref_logprobs = []
        for i, cand in enumerate(candidates):
            resp_len = cand.size(0)
            start    = lengths[i] - resp_len
            slice_logits = ref_logits[i, start:start+resp_len]
            lp = F.log_softmax(slice_logits, dim=-1)
            token_lp = lp.gather(1, cand.unsqueeze(1)).squeeze(1)
            ref_logprobs.append(token_lp.sum())
        ref_logprobs = torch.stack(ref_logprobs)

        rewards = rewards[: policy_logprobs.size(0)]
        advantages = rewards - rewards.mean()
        if CFG.advantage_normalization and advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)
        slice_logits_list = [logits[:, t, :] for t in range(logits.size(1))]
        token_lens = [r for _, r in lengths]
        per_token_pl = policy_logprobs / torch.tensor(token_lens, device=self.dev)
        per_token_rl = ref_logprobs    / torch.tensor(token_lens, device=self.dev)

        adv = rewards - rewards.mean()
        if CFG.advantage_normalization and adv.std()>1e-8:
            adv = adv / (adv.std()+1e-8)
        policy_loss = -(adv * policy_logprobs).mean()

        kl_div = (per_token_rl - per_token_pl).mean()
        if CFG.kl_clip:
            kl_div = torch.clamp(kl_div, min=CFG.kl_clip_value)
        kl_penalty = CFG.beta * kl_div

        ranking_loss = torch.tensor(0.0, device=self.dev)
        cnt = 0; rl_sum = torch.tensor(0.0, device=self.dev)
        n = policy_logprobs.size(0)
        for i in range(n):
            for j in range(i+1, n):
                diff = policy_logprobs[i] - policy_logprobs[j]
                if rewards[i] > rewards[j]:
                    rl_sum += F.relu(-diff); cnt += 1
                elif rewards[j] > rewards[i]:
                    rl_sum += F.relu(diff); cnt += 1
        if cnt>0:
            ranking_loss = (rl_sum / cnt) * CFG.ranking_loss_weight

        ent_losses = []
        for (ql, rl), logits_slice in zip(lengths, slice_logits_list):
            lp = F.log_softmax(logits_slice, dim=-1)
            p  = lp.exp()
            ent = -(p * lp).sum(dim=-1).mean()
            ent_losses.append(ent)
        seq_entropy = torch.stack(ent_losses).mean()
        entropy_loss = -CFG.entropy_weight * seq_entropy

        total_loss = policy_loss + kl_penalty + ranking_loss + entropy_loss

        return total_loss

    def all_gather_object_list(self, obj_list):
        gathered_lists = [None] * self.world_size
        dist.all_gather_object(gathered_lists, obj_list)
        return gathered_lists

    def step_batch_immediate_update(self, groups):
        if not groups:
            return 0.0
        all_groups = self.all_gather_object_list(groups)
        combined = [g for gl in all_groups for g in (gl or [])]
        total = len(combined)
        per = total // self.world_size
        start = self.rank * per
        end   = total if self.rank == self.world_size-1 else start + per
        local_groups = combined[start:end]
        if not local_groups:
            return 0.0

        flat_seqs, flat_rewards, lengths, group_sizes = [], [], [], []
        for g in local_groups:
            q = g["query"].to(self.dev)
            qlen = q.size(0)
            group_sizes.append(len(g["candidates"]))
            for i, cand_cpu in enumerate(g["candidates"]):
                cand = cand_cpu.to(self.dev)
                rlen = cand.size(0)
                flat_seqs.append(torch.cat([q, cand], dim=0))
                flat_rewards.append(g["rewards"][i])
                lengths.append((qlen, rlen))

        padded = pad_sequence(flat_seqs, batch_first=True,
                            padding_value=self.tok.eos_token_id).to(self.dev)
        policy_out = self.policy(padded, use_cache=False)
        logits     = policy_out.logits if not isinstance(policy_out, tuple) else policy_out[0]

        with torch.no_grad(), autocast(device_type="cuda"):
            ref_out    = self.ref_model(padded, use_cache=False)
        ref_logits  = ref_out.logits if not isinstance(ref_out, tuple) else ref_out[0]

        policy_lps, ref_lps = [], []
        entropies = []
        for (qlen, rlen), seq, plogits, rlogits in zip(lengths, flat_seqs, logits, ref_logits):
            sl = plogits[qlen:qlen+rlen]
            rl = rlogits[qlen:qlen+rlen]
            resp = seq[qlen:qlen+rlen].unsqueeze(1)

            logp = F.log_softmax(sl, dim=-1)
            p    = torch.exp(logp)
            token_ent = -(p * logp).sum(dim=-1).mean()

            policy_lps.append(logp.gather(1, resp).squeeze(1).sum())
            ref_lps   .append(F.log_softmax(rl, dim=-1).gather(1, resp).squeeze(1).sum())
            entropies .append(token_ent)

        policy_lps = torch.stack(policy_lps)
        ref_lps    = torch.stack(ref_lps)
        entropies  = torch.stack(entropies)
        rewards    = torch.tensor(flat_rewards, device=self.dev, dtype=torch.float32)

        losses = []
        p_losses, kl_losses, r_losses, e_losses = [], [], [], []
        idx = 0
        for sz in group_sizes:
            pl = policy_lps[idx:idx+sz]
            rl = ref_lps   [idx:idx+sz]
            rw = rewards   [idx:idx+sz]
            ent = entropies[idx:idx+sz]
            idx += sz

            adv = rw - rw.mean()
            if CFG.advantage_normalization and adv.std()>1e-8:
                adv = adv / (adv.std()+1e-8)
            policy_loss = -(adv * pl).mean()

            kl_loss = CFG.beta * (pl - rl).mean()

            rank_loss = torch.tensor(0.0, device=self.dev)
            if sz>1:
                sum_rl, cnt = torch.tensor(0., device=self.dev), 0
                for i in range(sz):
                    for j in range(i+1, sz):
                        if rw[i]>rw[j]:
                            sum_rl += F.relu(-(pl[i]-pl[j])); cnt+=1
                        elif rw[j]>rw[i]:
                            sum_rl += F.relu(-(pl[j]-pl[i])); cnt+=1
                if cnt>0:
                    rank_loss = sum_rl / cnt

            entropy_loss = -CFG.entropy_weight * ent.mean()

            group_loss = policy_loss + kl_loss + CFG.ranking_loss_weight*rank_loss + entropy_loss

            p_losses.append(policy_loss); kl_losses.append(kl_loss)
            r_losses.append(rank_loss);   e_losses.append(entropy_loss)
            losses .append(group_loss)

        m_p = torch.stack(p_losses).mean().item()
        m_k = torch.stack(kl_losses).mean().item()
        m_r = torch.stack(r_losses).mean().item()
        m_e = torch.stack(e_losses).mean().item()
        batch_loss = torch.stack(losses).mean()

        if self.is_main_process:
            print(
                f"[Step {self.step+1}] policy={m_p:.4f}, kl={m_k:.4f}, "
                f"rank={m_r:.4f}, ent={m_e:.4f}, total={batch_loss.item():.4f}"
            )
            if CFG.use_wandb:
                wandb.log(
                    {
                        "policy_loss": m_p,
                        "kl_penalty": m_k,
                        "ranking_loss": m_r,
                        "entropy_loss": m_e,
                        "total_loss": batch_loss.item(),
                        "step": self.step + 1,
                    }
                )

        self.opt.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.opt.step()
        self.step += 1

        return batch_loss.item()

    def create_grpo_groups(self, queries, candidates_list, reward_fn):
        groups = []
        for q_idx, (query, candidates) in enumerate(zip(queries, candidates_list)):
            valid_candidates = []
            sequences = []
            query_cpu = query.clone().detach().cpu()
            for cand_idx, cand in enumerate(candidates):
                if cand.numel() == 0:
                    continue
                try:
                    cand_cpu = cand.clone().detach().cpu()
                    
                    sequence = self.tok.decode(torch.cat([query_cpu, cand_cpu]), skip_special_tokens=True)
                    
                    if len(sequence.strip()) > 5:  
                        sequences.append(sequence)
                        valid_candidates.append(cand_cpu)
                            
                except Exception as e:
                    if self.is_main_process:
                        print(f"Decode failure for candidate {cand_idx}: {e}")
                    continue
            
            if len(valid_candidates) < 2:
                if self.is_main_process:
                    print(f"Only {len(valid_candidates)} valid candidates found, skipping query {q_idx}")
                continue

            try:
                rewards, _ = reward_fn(clean_sequences(sequences))
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.squeeze().detach().cpu().numpy()
                else:
                    rewards = np.array(rewards).squeeze()
                
                if rewards.ndim != 1:
                    rewards = rewards.flatten()
                    
            except Exception as e:
                if self.is_main_process:
                    print(f"Reward calculation failure for query {q_idx}: {e}")
                continue
            
            if len(rewards) != len(valid_candidates):
                if self.is_main_process:
                    print(f"Unmatched reward amount: {len(rewards)} vs {len(valid_candidates)}")
                continue

            if len(valid_candidates) > CFG.group_size:
                idx = np.argsort(rewards)[::-1]
                half = CFG.group_size // 2
                head = list(idx[:half])
                mid_start = len(idx) // 3
                mid_len = CFG.group_size - half
                mid = list(idx[mid_start:mid_start + mid_len])
                tail_need = CFG.group_size - len(head) - len(mid)
                tail = list(idx[-tail_need:]) if tail_need > 0 else []
                selected = (head + mid + tail)[:CFG.group_size]
            else:
                selected = range(len(valid_candidates))
            group_candidates = [valid_candidates[i] for i in selected]
            group_rewards = [float(rewards[i]) for i in selected]
            
            reward_std = np.std(group_rewards)
            if reward_std < 0.001:
                if self.is_main_process:
                    print(f"Skip group {q_idx}: rewards too similar (std={reward_std:.6f})")
                continue
            group_candidates = [cand.cpu() for cand in group_candidates]
            group = {
                "query": query_cpu,
                "candidates": group_candidates,
                "rewards": group_rewards,
                "reward_std": reward_std
            }
            groups.append(group)
        return groups

    def generate_candidates_ultra_memory_efficient(self, prompts, user_context=None):
        if self.is_main_process:
            print(f"[generate] Starting generation for {len(prompts)} prompts", flush=True)
        bad_ids = [self.tok.encode(w) for w in ["B", "O", "U", "X", "Z"]]
        gen_cfg = dict(
            max_new_tokens=CFG.max_new_tokens, 
            top_p=CFG.top_p, 
            top_k=CFG.top_k,
            temperature=CFG.temperature, 
            do_sample=True,
            pad_token_id=self.tok.eos_token_id, 
            bad_words_ids=bad_ids,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict_in_generate=False
        )
        
        # Add user context if provided and policy supports it
        if user_context is not None and hasattr(self.policy.module, 'generate'):
            gen_cfg['user_context'] = user_context
        
        all_candidates = []
        
        self.policy.eval()
        torch.set_grad_enabled(False)
        
        try:
            for prompt_idx, prompt in enumerate(prompts):
                if self.is_main_process and prompt_idx % 5 == 0:
                    print(f"[generate] Prompt {prompt_idx+1}/{len(prompts)}", flush=True)
                candidates = []
                for cand_idx in range(CFG.num_candidates):
                    try:
                        prompt_input = prompt.clone().detach().unsqueeze(0).to(self.dev, non_blocking=True)
                        # Check if policy has user-conditioned generate method
                        if hasattr(self.policy.module, 'generate') and user_context is not None:
                            output = self.policy.module.generate(prompt_input, **gen_cfg)
                        else:
                            # Fall back to base policy generate (remove user_context from gen_cfg)
                            gen_cfg_no_user = {k: v for k, v in gen_cfg.items() if k != 'user_context'}
                            if hasattr(self.policy.module, 'base_policy'):
                                output = self.policy.module.base_policy.generate(prompt_input, **gen_cfg_no_user)
                            else:
                                output = self.policy.module.generate(prompt_input, **gen_cfg_no_user)
                        candidates.append(output[0][len(prompt):].detach().cpu())
                        del prompt_input, output
                    except torch.cuda.OutOfMemoryError:
                        if self.is_main_process:
                            print(f"Skip. OOM during candidate {cand_idx} generation.")
                        continue
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Skip. Error on generating candidate {cand_idx}: {e}")
                        continue
                all_candidates.append(candidates)
        finally:
            torch.set_grad_enabled(True)
            self.policy.train()
        return all_candidates

def load_progen_memory_efficient():
    print(f"[load_progen] Checking paths...", flush=True)
    if CFG.base_model_path is None or CFG.tokenizer_path is None:
        raise ValueError("Both base_model_path and tokenizer_path must be provided.")
    print(f"[load_progen] Loading from {CFG.base_model_path}", flush=True)
    tokenizer, model = load_pretrained_progen_model(
        base_model_path=str(CFG.base_model_path),
        tokenizer_path=str(CFG.tokenizer_path),
        lora_checkpoint=str(CFG.lora_checkpoint) if CFG.lora_checkpoint else None,
        inference_mode=False,
    )
    print(f"[load_progen] Done loading model", flush=True)
    return tokenizer, model

def create_distributed_dataloader(cfg, rank, world_size, tokenizer):
    dataset = loading_dataset(
        cfg.steps,
        cfg.batch_size,
        tokenizer=tokenizer,
        prompt=cfg.prompt,
        prompt_file=cfg.prompt_file,
        return_dataset=True,
    )

    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    per_rank_batch = max(1, cfg.batch_size // world_size)
    dataloader = DataLoader(
        dataset,
        batch_size=per_rank_batch,
        sampler=sampler,
        pin_memory=True,
        num_workers=0,
        persistent_workers=False
    )

    return dataloader

def train_worker(rank, world_size, cfg):
    global CFG  # Declare CFG as global to avoid local scope issues
    CFG = cfg
    try:
        print(f"[Rank {rank}] Starting worker...", flush=True)
        setup_distributed(rank, world_size, cfg.port)
        print(f"[Rank {rank}] Distributed setup complete", flush=True)
        set_seed(cfg.seed, rank)
        
        device = torch.device(f"cuda:{rank}")
        
        if rank == 0 and cfg.use_wandb:
            if wandb is None:
                print("Warning: wandb not installed, skipping wandb logging", flush=True)
            else:
                try:
                    print("Initializing wandb...", flush=True)
                    wandb.init(
                        project=cfg.tracker_project_name,
                        name=cfg.exp_name,
                        config=asdict(cfg),
                        entity=cfg.wandb_entity,
                    )
                    print("Wandb initialized successfully!", flush=True)
                except Exception as e:
                    print(f"Warning: Could not initialize wandb: {e}", flush=True)

        if rank == 0:
            print("Loading model...", flush=True)
            print(f"Model path: {CFG.base_model_path}", flush=True)
            print(f"Tokenizer path: {CFG.tokenizer_path}", flush=True)

        tok, model = load_progen_memory_efficient()
        
        if rank == 0:
            print("Model loaded successfully!", flush=True)
            print("Creating trainer...", flush=True)

        trainer = DistributedUltraLowMemoryGRPOTrainer(
            model, tok, device, rank, world_size, 
            use_user_conditioning=cfg.use_personalization
        )
        
        if rank == 0:
            print("Trainer created successfully!", flush=True)

        # Setup reward function
        if cfg.use_personalization:
            if rank == 0:
                print("Loading property function for personalization...", flush=True)
            
            # Load property function
            property_fn = create_unified_property_function(
                activity_checkpoint=cfg.classifier_checkpoint,
                toxicity_checkpoint=cfg.toxicity_checkpoint,
                stability_checkpoint=cfg.stability_checkpoint,
                esm_model_size=cfg.esm_mode,
                device=device,
            )
            
            if rank == 0:
                print("Property function loaded successfully!", flush=True)
            
            # Setup persona cycling
            if cfg.persona_cycle_mode == "single":
                personas_to_use = [get_persona(cfg.persona_name)]
                if rank == 0:
                    print(f"Using single persona: {cfg.persona_name}", flush=True)
            else:
                if rank == 0:
                    print("Loading personas...", flush=True)
                personas_to_use = [get_persona(name) for name in list_personas()]
                if rank == 0:
                    print(f"Cycling through {len(personas_to_use)} personas ({cfg.persona_cycle_mode} mode)", flush=True)
            
            current_persona_idx = 0
            
            def get_next_persona():
                nonlocal current_persona_idx
                if cfg.persona_cycle_mode == "random":
                    return random.choice(personas_to_use)
                else:  # round_robin or single
                    persona = personas_to_use[current_persona_idx]
                    if cfg.persona_cycle_mode == "round_robin":
                        current_persona_idx = (current_persona_idx + 1) % len(personas_to_use)
                    return persona
            
            # Create reward function generator
            def make_reward_fn(persona):
                return create_hybrid_reward_fn(
                    property_function=property_fn,
                    persona=persona,
                    penalty=cfg.reward_penalty,
                    min_charge=cfg.min_charge,
                    device=device,
                )
            
            # Initialize with first persona
            current_persona = get_next_persona()
            reward_fn = make_reward_fn(current_persona)
            
        else:
            # Original classifier-based reward
            if rank == 0:
                print("Loading ESM and classifier...", flush=True)
            
            batch_converter, esm_model, alphabet = load_esm(cfg.esm_mode, device=device)
            if rank == 0:
                print("ESM loaded", flush=True)
            classifier = MLP(input_dim=320, hidden_dim=128).to(device)
            if cfg.classifier_checkpoint is None:
                raise ValueError("Classifier checkpoint must be provided.")
            classifier.load_state_dict(torch.load(cfg.classifier_checkpoint, map_location="cpu", weights_only=False))
            classifier.eval()

            def reward_fn(seqs):
                return reward_amp_cls(
                    seqs,
                    esm_model=esm_model,
                    batch_converter=batch_converter,
                    alphabet=alphabet,
                    classifier=classifier,
                    device=device,
                )
            
            personas_to_use = None
            get_next_persona = None
            make_reward_fn = None
        
        if rank == 0:
            print("Creating dataloader...", flush=True)
        dataloader = create_distributed_dataloader(cfg, rank, world_size, tok)
        if rank == 0:
            print(f"Dataloader created. Starting training loop...", flush=True)
        for epoch in range(cfg.epochs):
            if rank == 0:
                print(f"\n{'='*50}", flush=True)
                print(f"Epoch {epoch+1}/{cfg.epochs}", flush=True)
                print(f"{'='*50}", flush=True)
            dataloader.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(dataloader):
                if rank == 0:
                    print(f"\n--- Batch {batch_idx+1} ---", flush=True)
                
                # Select persona for this batch if using personalization
                if cfg.use_personalization:
                    if rank == 0:
                        print(f"Selecting persona...", flush=True)
                    current_persona = get_next_persona()
                    reward_fn = make_reward_fn(current_persona)
                    user_context = current_persona.weights.to(device)
                    
                    if rank == 0:
                        print(f"Using persona: {current_persona.name}", flush=True)
                else:
                    user_context = None
                
                try:
                    if rank == 0:
                        print(f"Processing batch data...", flush=True)
                    if isinstance(batch, dict):
                        if "input_ids" in batch:
                            prompts = batch["input_ids"]
                        else:
                            prompts = next(iter(batch.values()))
                    else:
                        prompts = batch
                    if prompts.dim() > 2:
                        prompts = prompts.squeeze(-1)
                    prompts = prompts.to(device)
                    prompts = [prompts[i] for i in range(prompts.size(0))]
                    prompts = trainer.truncate_sequences(prompts)
                    
                    if rank == 0:
                        print(f"Generating {CFG.num_candidates} candidates for {len(prompts)} prompts...", flush=True)
                    
                    # Generate with user context
                    candidates_list = trainer.generate_candidates_ultra_memory_efficient(
                        prompts, 
                        user_context=user_context
                    )
                    
                    if rank == 0:
                        print(f"Generation complete. Creating GRPO groups...", flush=True)
                    
                    groups = trainer.create_grpo_groups(prompts, candidates_list, reward_fn)
                    if not groups:
                        if rank == 0:
                            print("Skip. No valid GRPO groups")
                        continue

                    loss = trainer.step_batch_immediate_update(groups)
                    if rank == 0 and trainer.step % cfg.save_every == 0 and trainer.step > 0:
                        checkpoint_path = cfg.output_dir / f"checkpoint_step_{trainer.step}"
                        ensure_dir(checkpoint_path)
                        try:
                            # Save the policy (handles user-conditioned wrapper if present)
                            if cfg.use_personalization and hasattr(trainer.policy.module, 'save_pretrained'):
                                trainer.policy.module.save_pretrained(checkpoint_path, safe_serialization=True)
                            elif hasattr(trainer.policy.module, 'base_policy'):
                                trainer.policy.module.base_policy.save_pretrained(checkpoint_path, safe_serialization=True)
                            else:
                                trainer.policy.module.save_pretrained(checkpoint_path, safe_serialization=True)
                            print(f"Saving checkpoint: {checkpoint_path}")
                        except Exception as e:
                            print(f"Error saving checkpoint: {e}")
                    
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print("Skip. OOM during batch processing.")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    if rank == 0:
                        print(f"Error on batch processing: {e}")
                    continue
        
        if rank == 0:
            try:
                final_path = cfg.output_dir / "final_model"
                ensure_dir(final_path)
                # Save the policy (handles user-conditioned wrapper if present)
                if cfg.use_personalization and hasattr(trainer.policy.module, 'save_pretrained'):
                    trainer.policy.module.save_pretrained(final_path, safe_serialization=True)
                elif hasattr(trainer.policy.module, 'base_policy'):
                    trainer.policy.module.base_policy.save_pretrained(final_path, safe_serialization=True)
                else:
                    trainer.policy.module.save_pretrained(final_path, safe_serialization=True)
                print(f"Save final checkpoint: {final_path}")
            except Exception as e:
                print(f"Failure on model saving: {e}")
    
    except Exception as e:
        print(f"Error in train_worker rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if rank == 0 and cfg.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
        
        try:
            cleanup_distributed()
        except:
            pass

def main():
    parse_args()

    if CFG.world_size < 1:
        CFG.world_size = 1

    print(f"Starting distributed training on {CFG.world_size} GPU(s)", flush=True)
    ensure_dir(CFG.output_dir)

    try:
        print("Setting multiprocessing start method to 'spawn'...", flush=True)
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Start method already set", flush=True)
        pass

    try:
        print(f"Spawning {CFG.world_size} worker process(es)...", flush=True)
        mp.spawn(
            train_worker,
            args=(CFG.world_size, CFG),
            nprocs=CFG.world_size,
            join=True,
        )
        print("All workers completed", flush=True)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error in distributed training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Training completed.")

if __name__ == "__main__":
    main()
