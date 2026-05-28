import torch
from torch.utils.data import DataLoader, IterableDataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from utils.helpers import DiversityFilter
from utils.top_k_tracker import TopKTracker
from agfn.backtraj import ReverseFineTune
from gflownet.envs.mol_building_env import build_frozen_seed_graph
import pickle
import numpy as np

class FTSampling_Iterator(IterableDataset):
    def __init__(self, ft_trainer, wrapped_model, wrapped_model_prior, dev, num_online, beta, ft_conditionals_dict= None ):
        self.num_online = num_online
        self.sub_batch_size =  ft_trainer.hps['training_batch_size']
        self.batch_size = ft_trainer.hps['sampling_batch_size']
        if ft_trainer.hps.offline_data:
            self.num_offline = num_online
            self.widx2smiles = ft_trainer.widx2smiles
            self.small_task_data = True if len(self.widx2smiles[0])<self.sub_batch_size else False
        self.cond_info_task = ft_trainer.cond_info_task
        self.wrapped_model = wrapped_model
        self.wrapped_model_prior = wrapped_model_prior
        self.dev = dev
        self.beta = beta
        self.gfn_samples_path = getattr(ft_trainer, 'gfn_samples_path', None)
        self.hps = ft_trainer.hps
        self.graph_sampler = ft_trainer.graph_sampler
        self.ft_conditionals_dict = ft_conditionals_dict
        self.ctx = ft_trainer.ctx
        self.algo = ft_trainer.algo
        self.reward = ft_trainer.reward
        self.offline_data = ft_trainer.hps.offline_data
        self.task_model = ft_trainer.task_model
        self.tasks = [
            'Caco2',
            'LD50',
            'Lipophilicity',
            'Solubility',
            'Solubility',
            'BindingRate',
            'MicroClearance',
            'HepatocyteClearance'
        ]
        if (self.hps['task'] in self.tasks):
            self.Y_scaler = ft_trainer.Y_scaler
        self.reverse = ReverseFineTune(ft_trainer.env, self.ctx,self.hps, ft_trainer.rng)
        if self.hps.get('initial_scaffold', None):
            # Frozen-core / fixed-seed growth: start every trajectory from the literal seed molecule
            # with its core held immutable (unlike seed_smiles below, which uses the Murcko scaffold
            # and leaves the seed editable). See build_frozen_seed_graph for the index conventions.
            self.seed_graph = build_frozen_seed_graph(
                self.ctx, self.hps['initial_scaffold'],
                allowed_growth_atoms=self.hps.get('allowed_growth_atoms', None),
                frozen_atoms=self.hps.get('frozen_atoms', None),
            )
            print(f"[frozen-core] seeding from initial_scaffold "
                  f"({self.seed_graph.graph['frozen_seed_size']} core atoms, "
                  f"{len(self.seed_graph.graph['frozen_growth_sites'])} growth site(s)).")
            if 'seed_smiles' in self.hps:
                print("[frozen-core] note: seed_smiles/seed_scaffold are ignored because "
                      "initial_scaffold is set.")
            if self.offline_data:
                # Offline trajectories are reconstructed from dataset molecules that do not contain
                # the seed core, so they cannot satisfy the frozen-core constraint. Run online-only.
                print("[frozen-core] forcing ONLINE-ONLY mode (offline dataset molecules "
                      "cannot preserve the frozen core).")
                self.offline_data = False
        elif 'seed_smiles' in self.hps:
            if self.hps['seed_scaffold']:
                print('trajsamp.py Optimizing ', self.hps['seed_scaffold'])
                scaffold = Chem.MolFromSmiles(self.hps['seed_scaffold'])
            else:
                print('trajsamp.py Optimizing ', self.hps['seed_smiles'])
                mol = Chem.MolFromSmiles(self.hps['seed_smiles'])
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            self.seed_graph = self.ctx.mol_to_graph(scaffold)
        else:
            self.seed_graph = None
        self.div_fil = DiversityFilter()
        
        # Then assign the dictionary
        self.task_model_reward_funcs = {
            'Caco2': self._reward_caco2,
            'LD50': self._reward_ld50,
            'Lipophilicity': self._reward_lipophilicity,
            'Solubility': self._reward_solubility,
            'BindingRate': self._reward_binding_rate,
            'MicroClearance': self._reward_micro_clearance,
            'HepatocyteClearance': self._reward_hepatocyte_clearance
        }

        self.top_k_tracker = TopKTracker(ks=(10, 100))
        # Resolved lazily in __iter__, not here: build_train_loader runs before train()
        # appends the per-run subdir to hps['log_dir'], so capturing it now would point the
        # hall of fame at the shared base dir (leaking molecules across runs). The worker that
        # actually fills the heap is forked only after the subdir is set, so __iter__ sees it.
        self.top_k_dir = None
        self.iter_counter = 0
        # canonical SMILES seen across the whole run, backing the cumulative-novelty metric
        self.seen_smiles = set()

    def _reward_caco2(self, x):
        return self.reward.caco2(self.Y_scaler, self.task_model, x)

    def _reward_ld50(self, x):
        return self.reward.toxicity(self.Y_scaler, self.task_model, x)

    def _reward_lipophilicity(self, x):
        return self.reward.lipophilicity(self.Y_scaler, self.task_model, x)

    def _reward_solubility(self, x):
        return self.reward.solubility(self.Y_scaler, self.task_model, x)

    def _reward_binding_rate(self, x):
        return self.reward.binding_rate(self.Y_scaler, self.task_model, x)

    def _reward_micro_clearance(self, x):
        return self.reward.micro_clearance(self.Y_scaler, self.task_model, x)

    def _reward_hepatocyte_clearance(self, x):
        return self.reward.hepatocyte_clearance(self.Y_scaler, self.task_model, x)

    def smiles_2_offln_trajs(self, smiles_batch, wrapped_model,dev):
         mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_batch if smiles is not None]
         lg_rewards, flat_rewards, total_reward, zinc_flat_offln_rew = self.reward.molecular_rewards(mols)
         cond_info_bck = self.cond_info_task.compute_cond_info_backward(flat_rewards)
         self.cond_info_bck_encoding = self.cond_info_task.thermometer_encoding(cond_info_bck)
         data = self.reverse.reverse_trajectory(smiles_batch,self.cond_info_bck_encoding, wrapped_model, dev)
         flipped_data = self.reverse.flip_trajectory(data)
         return data, flipped_data, (lg_rewards, flat_rewards, total_reward, zinc_flat_offln_rew)

    def _batch_diversity_metrics(self, trajs):
        """Validity / uniqueness / novelty as **percentages (0-100)** over the full sampled batch.

        Computed once per sampled batch (not per training sub-batch) so the numbers are meaningful:
          - valid%  = trajectories whose graph yields a valid molecule, over the whole batch
          - unique% = distinct molecules (canonical SMILES) among the valid ones
          - novel%  = valid molecules never generated before this point in the run
        Updates the running ``seen_smiles`` set that backs the cumulative-novelty metric.
        """
        valid_smiles = [
            Chem.MolToSmiles(self.ctx.graph_to_mol(trajs[i]['traj'][-1][0]))
            for i in range(len(trajs)) if trajs[i]["is_valid"]
        ]
        n_total = max(len(trajs), 1)
        n_valid = max(len(valid_smiles), 1)
        uniq = set(valid_smiles)
        novel = uniq - self.seen_smiles
        self.seen_smiles |= uniq
        valid_pct = 100.0 * len(valid_smiles) / n_total
        unique_pct = 100.0 * len(uniq) / n_valid
        novel_pct = 100.0 * len(novel) / n_valid
        return valid_pct, unique_pct, novel_pct

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        if self.offline_data:
            worker_smiles = self.widx2smiles[self._wid]
            for k in range(self.hps['num_iter']):
                self.smiles_idx = 0
                while self.smiles_idx <len(worker_smiles):
                    try:
                        cond_info = self.cond_info_task.compute_cond_info_forward(self.num_online)
                        cond_info_encoding = self.cond_info_task.thermometer_encoding(cond_info)
                        smiles_batch = worker_smiles[self.smiles_idx:self.smiles_idx+self.batch_size]  # TODO: Account for some smiles left out at the end of the worker_smiles list
                        self.smiles_idx += self.batch_size//2  # divided by 2 because smiles_batch is batch_size/2; the remainder batch_size/2 is for online trajs
                        with torch.no_grad(): 
                            online_trajs = self.graph_sampler.sample_from_model(self.wrapped_model,self.num_online,
                                                                                cond_info_encoding.to(self.dev),
                                                                                self.dev, random_stop_action_prob= self.hps['random_stop_prob'],
                                                                                random_action_prob = self.hps['random_action_prob'],
                                                                                seed_graph = self.seed_graph )
                            tmp_offln_model = self.wrapped_model_prior if self.hps.type=='rtb' else self.wrapped_model
                            _, offline_trajs, offln_rew_tup = self.smiles_2_offln_trajs(smiles_batch, tmp_offln_model, self.dev)

                        # Full-batch validity/uniqueness/novelty (%) over all online trajectories,
                        # attached to every sub-batch yielded below.
                        batch_valid_pct, batch_unique_pct, batch_novel_pct = self._batch_diversity_metrics(online_trajs)

                        r = (len(offline_trajs)+len(online_trajs)) // self.sub_batch_size
                        for j in range(0,min(len(offline_trajs), len(online_trajs)),self.sub_batch_size):
                            sub_online_trajs = online_trajs[j : j + self.sub_batch_size]
                            avg_batch_len, avg_fwd_logprob, avg_bck_logprob = 0, 0,0
                            for i in range(len(sub_online_trajs)):
                                avg_batch_len += len(sub_online_trajs[i]['bck_a'])
                                avg_fwd_logprob += sub_online_trajs[i]['fwd_logprob'][0]
                                avg_bck_logprob += sub_online_trajs[i]['bck_logprob'][0]
                            valid_idcs = torch.tensor([i + 0 for i in range(len(sub_online_trajs)) if (sub_online_trajs[i + 0]["is_valid"]) & ("fwd_logprob" in sub_online_trajs[i+0])]).long()
                            if len(valid_idcs)==0:
                                with open(self.hps["log_dir"] + f"/invalid_mols{j}.txt", "w") as f:
                                    content = f"{str(online_trajs)}"
                                    f.write(content)
                            online_mols = [self.ctx.graph_to_mol(sub_online_trajs[i]['traj'][-1][0]) for i in valid_idcs]
                            
                            if self.small_task_data:
                                offline_mols = [Chem.MolFromSmiles(s) for s in smiles_batch] # when offline data is small
                                online_rew_tup = self.reward.molecular_rewards(online_mols)
                                online_rew = torch.Tensor(online_rew_tup[2]).unsqueeze(dim=1)
                                sub_offline_trajs = offline_trajs # when offline data is small
                                offline_rew = offln_rew_tup[2] #when offline data is small 
                            else:
                                offline_mols = [Chem.MolFromSmiles(s) for s in smiles_batch[j : j + self.sub_batch_size]] # when offline data is large
                                online_rew_tup = self.reward.molecular_rewards(online_mols)
                                online_rew = torch.Tensor(online_rew_tup[2]).unsqueeze(dim=1)
                                sub_offline_trajs = offline_trajs[j : j + self.sub_batch_size] # when offline data is large
                                offline_rew = offln_rew_tup[2][j : j + self.sub_batch_size] # when offline data is large
                            
                            offline_rew = torch.Tensor(offline_rew).unsqueeze(dim=1)
                            if self.hps['task'] in self.task_model_reward_funcs:
                                normalized_task_rew_online, true_task_score_online = self.reward.task_reward(self.hps.task, self.task_model, online_mols)
                                normalized_task_rew_offline, true_task_score_offline = self.reward.task_reward(self.hps.task, self.task_model,
                                                                                offline_mols)
                            else:
                                normalized_task_rew_online, true_task_score_online = self.reward.task_reward(self.hps.task, self.task_model, online_mols)
                                normalized_task_rew_offline, true_task_score_offline = self.reward.task_reward(self.hps.task, self.task_model, offline_mols)
                            if self.hps.task_rewards_only:
                                flat_rewards_online, flat_rewards_offline =  torch.Tensor(normalized_task_rew_online), torch.Tensor(normalized_task_rew_offline)
                            else:
                                flat_rewards_online, flat_rewards_offline = online_rew*normalized_task_rew_online, offline_rew*normalized_task_rew_offline
                            
                            smiles_list = [Chem.MolToSmiles(mol) for mol in online_mols]
                            if self.hps['diversity_filter']:
                                self.div_fil.update(smiles_list)
                                #penalize rewards for frequently generated scaffolds
                                flat_rewards = self.div_fil.penalize_reward(smiles_list,flat_rewards_online)
                                
                            #Ensure reward is set to 0 for invalid trajectories (molecules)
                            pred_reward_online = torch.zeros(len(sub_online_trajs), flat_rewards_online.shape[1])
                            pred_reward_online[valid_idcs - 0] = flat_rewards_online.float()
                            pred_reward = torch.cat((pred_reward_online, flat_rewards_offline))
                            beta_vector = self.hps['beta_exp']*torch.ones(pred_reward.shape[0])
                            log_rewards = self.beta_to_logreward(beta_vector, pred_reward)
                            cond_info = torch.cat((cond_info_encoding[j : j + self.sub_batch_size],
                                                self.cond_info_bck_encoding))
                            gfn_batch = self.algo.construct_batch(sub_online_trajs+sub_offline_trajs, cond_info, log_rewards)
                            gfn_batch.num_offline = self.num_offline // r
                            gfn_batch.num_online = self.num_online // r
                            gfn_batch.flat_rewards = pred_reward.detach().cpu()
                            gfn_batch.online_flat_rewards = pred_reward_online
                            gfn_batch.offline_flat_rewards = offline_rew
                            gfn_batch.num_sub_online_trajs = len(sub_online_trajs)
                            gfn_batch.valid_percent = batch_valid_pct
                            gfn_batch.unique_percent = batch_unique_pct
                            gfn_batch.novel_percent = batch_novel_pct
                            gfn_batch.avg_batch_len = (avg_batch_len)/len(sub_online_trajs)
                            gfn_batch.avg_fwd_logprob = avg_fwd_logprob/len(sub_online_trajs)
                            gfn_batch.avg_bck_logprob = avg_bck_logprob/len(sub_online_trajs)
                            gfn_batch.offln_zinc_rad = np.average(offln_rew_tup[3])#[j : j + self.sub_batch_size])
                            gfn_batch.onln_zinc_rad = np.average(online_rew_tup[3])
                            gfn_batch.offline_avg_tpsa =  np.average(offln_rew_tup[1][0][0])
                            gfn_batch.online_avg_tpsa =  np.average(online_rew_tup[1][0][0])
                            gfn_batch.offln_num_rings = np.average(offln_rew_tup[1][1][0])
                            gfn_batch.online_num_rings = np.average(online_rew_tup[1][1][0])
                            gfn_batch.offline_avg_SAS = np.average(offln_rew_tup[1][2][0])
                            gfn_batch.online_avg_SAS =  np.average(online_rew_tup[1][2][0])
                            gfn_batch.offline_qed = np.average(offln_rew_tup[1][3][0])
                            gfn_batch.online_qed = np.average(online_rew_tup[1][3][0])
                            gfn_batch.avg_task_reward_online = np.average(normalized_task_rew_online) 
                            gfn_batch.avg_task_reward_offline = np.average(normalized_task_rew_offline)  
                            if self.hps.task is not None:
                                gfn_batch.avg_task_score_online = np.average(true_task_score_online)
                                gfn_batch.avg_task_score_offline = np.average(true_task_score_offline)
                            else:
                                gfn_batch.avg_task_reward_online, gfn_batch.avg_task_reward_offline, gfn_batch.avg_task_score_online, gfn_batch.avg_task_score_offline = 0,0,0,0

                            yield gfn_batch, online_mols

                    except Exception as e:
                        raise e
                        print(e)
                        continue
        else:
            try:
                # hps['log_dir'] is now the finalized per-run subdir (train() set it before
                # this worker was forked); persist the hall of fame there and resume from it.
                self.top_k_dir = self.hps['log_dir']
                self.top_k_tracker.load(self.top_k_dir)
                for _ in range(self.hps['num_iter']):
                    try:
                        cond_info = self.cond_info_task.compute_cond_info_forward(self.num_online)
                        cond_info_encoding = self.cond_info_task.thermometer_encoding(cond_info)
                        with torch.no_grad():
                            online_trajs_sampled = self.graph_sampler.sample_from_model(self.wrapped_model,self.num_online,
                                                                                    cond_info_encoding.to(self.dev),
                                                                                    self.dev, random_stop_action_prob= self.hps['random_stop_prob'],
                                                                                    random_action_prob = self.hps['random_action_prob'],
                                                                                    seed_graph= self.seed_graph)

                        # Full-batch validity/uniqueness/novelty (%), computed once over all sampled
                        # trajectories and attached to every sub-batch yielded below.
                        batch_valid_pct, batch_unique_pct, batch_novel_pct = self._batch_diversity_metrics(online_trajs_sampled)

                        for j in range(0, self.num_online, self.sub_batch_size):
                            online_trajs = online_trajs_sampled[j:j+self.sub_batch_size]
                            avg_batch_len, avg_fwd_logprob, avg_bck_logprob = 0, 0, 0
                            for i in range(len(online_trajs)):
                                avg_batch_len += len(online_trajs[i]['bck_a'])
                                avg_fwd_logprob += online_trajs[i]['fwd_logprob'][0]
                                avg_bck_logprob += online_trajs[i]['bck_logprob'][0]

                            valid_idcs = torch.tensor([i + 0 for i in range(len(online_trajs)) if (online_trajs[i + 0]["is_valid"])]).long()
                            if len(valid_idcs)==0:
                                with open(self.hps["log_dir"] + f"/invalid_mols{j}.txt", "w") as f:
                                    content = f"{str(online_trajs)}"
                                    f.write(content)
                        
                            mols = [self.ctx.graph_to_mol(online_trajs[i]['traj'][-1][0]) for i in valid_idcs]
                            if self.hps.get('write_mols_to_disk', False):
                                with open(self.gfn_samples_path+'/sampled_mols.pkl', 'wb') as f:
                                    pickle.dump(mols, f)
                            rew_tup = self.reward.molecular_rewards(mols)
                            rew = torch.Tensor(rew_tup[2]).unsqueeze(dim=1)
                            if self.hps['task'] in self.task_model_reward_funcs:
                                normalized_task_rew, true_task_score = self.reward.task_reward(self.hps.task, self.task_model, mols)
                            else:
                                normalized_task_rew, true_task_score = self.reward.task_reward(self.hps.task, mols)
                            if self.hps['task_rewards_only']:
                                flat_rewards =  torch.Tensor(normalized_task_rew) #  #torch.mul(rew,flat_rewards_task) #flat_rewards_qed
                            else:
                                flat_rewards = rew*normalized_task_rew

                            #Drug likeliness score
                            smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
                            if self.hps['diversity_filter']:
                                self.div_fil.update(smiles_list)
                                #penalize rewards for frequently generated scaffolds
                                flat_rewards = self.div_fil.penalize_reward(smiles_list,flat_rewards)

                            rewards_for_topk = flat_rewards.detach().cpu().numpy().reshape(-1).tolist()
                            affinities_for_topk = np.asarray(true_task_score).reshape(-1).tolist()
                            self.top_k_tracker.add_batch(
                                smiles_list, rewards_for_topk,
                                affinities=affinities_for_topk,
                                iteration=self.iter_counter,
                            )
                            # Persist on the top-K interval, decoupled from model checkpointing
                            # (`checkpoint_every`): the hall of fame is tiny, so it can be flushed far
                            # more often than the multi-hundred-MB model state. A final flush in
                            # __iter__'s finally captures whatever accumulated since the last interval.
                            if self.iter_counter % self.hps.get('top_k_save_every', 100) == 0:
                                self._flush_top_k()
                            self.iter_counter += 1

                            #Ensure reward is set to 0 for invalid trajectories (molecules)
                            pred_reward = torch.zeros((len(online_trajs), flat_rewards.shape[1]))
                            pred_reward[valid_idcs - 0] = flat_rewards.float()
                        
                            beta_vector = self.hps['beta_exp']*torch.ones(pred_reward.shape[0])
                            log_rewards = self.beta_to_logreward(beta_vector, pred_reward)

                            gfn_batch = self.algo.construct_batch(online_trajs, cond_info_encoding, log_rewards)#pred_reward.squeeze(dim=1))
                            
                            gfn_batch.num_online = len(online_trajs)
                            gfn_batch.num_offline = 0
                            gfn_batch.flat_rewards = pred_reward.detach().cpu()
                            gfn_batch.valid_percent = batch_valid_pct
                            gfn_batch.unique_percent = batch_unique_pct
                            gfn_batch.novel_percent = batch_novel_pct
                            # Running hall-of-fame averages (after this batch was added above):
                            # mean reward/affinity of the current top-10/top-100. Logged each
                            # iteration so the notebook can plot the monotonic improvement curves.
                            topk_avgs = self.top_k_tracker.averages()
                            gfn_batch.top10_reward = topk_avgs["top10_reward"]
                            gfn_batch.top100_reward = topk_avgs["top100_reward"]
                            gfn_batch.top10_affinity = topk_avgs["top10_affinity"]
                            gfn_batch.top100_affinity = topk_avgs["top100_affinity"]
                            gfn_batch.avg_batch_len = (avg_batch_len)/len(online_trajs)
                            gfn_batch.avg_fwd_logprob = avg_fwd_logprob/len(online_trajs)
                            gfn_batch.avg_bck_logprob = avg_bck_logprob/len(online_trajs)
                        
                            gfn_batch.avg_qed = np.average(rew_tup[1][3][0])
                            gfn_batch.avg_tpsa = np.average(rew_tup[1][0][0])
                            gfn_batch.avg_num_rings =  np.average(rew_tup[1][1][0])
                            gfn_batch.avg_sas =  np.average(rew_tup[1][2][0])
                            if (self.hps.get("objective",None)=='property_targeting'):
                                if (self.hps.subtype=='new_props'):
                                    gfn_batch.avg_new_prop = np.average(rew_tup[1][4][0])
                        
                            gfn_batch.avg_zinc_rad = np.average(rew_tup[3])
                            gfn_batch.avg_task_reward = np.average(normalized_task_rew) #torch.mean(flat_rewards_task).item()
                            if self.hps.task is not None:
                                gfn_batch.avg_task_score = np.average(true_task_score)
                            else:
                                gfn_batch.avg_task_reward, gfn_batch.avg_task_score = 0,0
                                
                            yield gfn_batch, mols#, avg_batch_fwd_traj_len
                    except Exception as e:
                        print(e)
                        raise(e)
                        continue
            finally:
                self._flush_top_k_safe()
        
    def _flush_top_k(self):
        """Persist the running top-K heaps to disk.

        Always writes the two flat CSVs; when a Uni-Dock backend is present it also exports
        the 3D docked poses (with scores) as SDFs for visual inspection — poses come from the
        backend's pose cache, so this never re-docks. The heaps are bounded at ``max_k``, so
        this is cheap enough to call on every interval and again as a final flush.
        """
        self.top_k_tracker.save(self.top_k_dir)
        if hasattr(self.reward, "unidock"):
            snap = self.top_k_tracker.snapshot()
            k = self.top_k_tracker.max_k
            self.reward.unidock.write_hall_of_fame_sdf(
                snap["top_by_affinity"][k], f"{self.top_k_dir}/top100_by_affinity.sdf")
            self.reward.unidock.write_hall_of_fame_sdf(
                snap["top_by_reward"][k], f"{self.top_k_dir}/top100_by_reward.sdf")

    def _flush_top_k_safe(self):
        """Best-effort final flush, called from __iter__'s finally so it runs on normal end,
        on an exception, or on the GeneratorExit raised when the DataLoader tears the worker
        down. Guarded so a flush failure can never mask the real teardown reason or crash the
        worker during shutdown.
        """
        try:
            self._flush_top_k()
        except Exception as e:
            print(f"[top-k] final flush skipped: {e}")

    def beta_to_logreward(self,beta_vector, pred_reward):
        scalar_logreward = pred_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            beta_vector.shape
        ), f"dangerous shape misatch: {scalar_logreward.shape} vs {beta_vector.shape}"
        return scalar_logreward * beta_vector   # log(r(x)**beta) = beta*log(r(x))

            
def build_train_loader(hps,finetuner):
    wrapped_model, dev = finetuner._wrap_for_mp(finetuner.gfn_trainer.model)
    if hasattr(finetuner.gfn_trainer, "model_prior") and finetuner.gfn_trainer.model_prior is not None:
        wrapped_model_prior,_ = finetuner._wrap_for_mp(finetuner.gfn_trainer.model_prior)
    else:
        wrapped_model_prior = None
    iterator = FTSampling_Iterator(finetuner,wrapped_model,wrapped_model_prior, dev, hps['sampling_batch_size'], hps['beta_exp'])
    train_loader = DataLoader(iterator,batch_size =None, num_workers = hps['num_workers'])
    return train_loader