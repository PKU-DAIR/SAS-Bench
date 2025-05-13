import torch
import torch.nn as nn
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from functools import partial


class CriticModel(nn.Module):
    def __init__(self, model_from_pretrained, resume_path=None, layers_keep=1) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.config.num_layers = layers_keep
        model = AutoModel.from_pretrained(
            model_from_pretrained, trust_remote_code=True, config=self.config).to(torch.bfloat16)
        model = model.transformer
        # solve RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
        self.model = model
        self.output_linear = nn.Linear(
            self.config.hidden_size, 1, device=self.model.device, dtype=self.model.dtype)
        if resume_path is not None:
            self.output_linear.load_state_dict(torch.load(resume_path))

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        values = torch.tanh(self.output_linear(output.last_hidden_state))
        return values.transpose(0, 1).squeeze(-1)


class RewardModel(nn.Module):
    def __init__(self, model_from_pretrained) -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained)
        model = AutoModel.from_pretrained(model_from_pretrained)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def jaccard(s1, s2):
        assert len(s1)+len(s2) > 0
        s1 = set(s1)
        s2 = set(s2)
        s_or = s1 | s2
        s_and = s1 & s2
        jaccard_distance = len(s_and)/len(s_or)
        return jaccard_distance

    def forward(self, gen_texts=["I am the generated content from LLM."],
                gold_answers=['Generation from LLM.', "I am the output content from LLM."],
                bad_answers=['I am human.', 'I am the content from human writing.'],
                weight_for_cos_and_jaccard=[0.5, 0.5]):
        examples = gold_answers + bad_answers
        example_num = len(examples)
        assert len(gen_texts) > 0 and example_num > 0
        reward_direction = torch.ones(example_num, device=self.model.device)
        reward_direction[len(gold_answers):] = -1
        sentences = gen_texts + examples
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, return_tensors='pt')
        ids = self.tokenizer.batch_encode_plus(
            sentences, add_special_tokens=False)["input_ids"]
        # temporary truncate position_ids
        batch_size, max_seq_len = encoded_input["input_ids"].shape
        if max_seq_len > self.model.config.max_position_embeddings:
            encoded_input["position_ids"] = torch.arange(
                max_seq_len).expand((1, -1)).repeat(batch_size, 1)
            encoded_input["position_ids"] = encoded_input["position_ids"] / \
                max_seq_len*self.model.config.max_position_embeddings
            encoded_input["position_ids"] = encoded_input["position_ids"].floor(
            ).long()
        # Compute token embeddings
        with torch.no_grad():
            encoded_input = encoded_input.to(self.model.device)
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])
        gen_text_vecs = sentence_embeddings[:len(gen_texts)]
        answers_vecs = sentence_embeddings[len(gen_texts):]
        reward_ = []
        for i in range(gen_text_vecs.shape[0]):
            gen_text_vecs_ = gen_text_vecs[i:i+1]
            # 用一下广播计算cos
            coses = torch.cosine_similarity(
                gen_text_vecs_, answers_vecs, dim=1)
            # 余弦截断
            coses[(coses < 0)] = 0
            # 计算 jaccard距离
            jaccard_s1 = partial(RewardModel.jaccard, ids[i])
            jaccards = torch.tensor(np.vectorize(jaccard_s1)(np.array(
                ids[-len(examples):], dtype=object)), dtype=coses.dtype, device=coses.device)
            similarity = weight_for_cos_and_jaccard[0] * \
                coses + weight_for_cos_and_jaccard[1]*jaccards
            value, index = similarity.max(dim=-1)
            reward_.append(value*reward_direction[index])
        reward = torch.stack(reward_)
        return reward

class PPO(nn.Module):
    def __init__(self, tokenizer, qa_logs=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_logs = qa_logs
        # get weight matrix
        self.decay_up_matrix_T = self.get_decay_up_matrix_T()
    
    def get_log_prob(self, generated_outputs, input_ids, gen_method = "greedy_search"):
        # beam_search generate 给出来的scores就是log_prob了，所以直接gather获取即可
        gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:] 
        # let's stack the logits generated at each step to a tensor
        # 要小心greedy search 拿到的是score，需要再log_softmax
        # 而beam_search 拿到的已经是log_softmax了
        scores = torch.stack(generated_outputs.scores, dim=1)
        # if scores.max() >0 :
        #     gen_method = "greedy_search"
        if gen_method == "beam_search":
            log_prob_stacked = scores
        else:
            log_prob_stacked = torch.stack(generated_outputs.scores, dim=1).log_softmax(dim=-1)
        # now we need to collect the log_prob of the generated token # we need to add a dummy dim in the end to make gather work 
        log_prob = torch.gather(log_prob_stacked, 2, gen_sequences[:, :, None]).squeeze(-1)
        return log_prob
    
    def get_log_probs_with_input_ids(self, actor_model, states, gen_max_len):
        input_ids = states
        output = actor_model(input_ids)  #将已经生成的序列放进去计算，再次计算得到目标action也就是后续字符的概率或者log_prob值
        logits = output.logits[:, -(gen_max_len+1):-1].log_softmax(dim=-1) # 比先softmax再log好,复杂度减小，并且解决些nan问题
        new_log_probs = logits.gather(dim=-1, index=input_ids[:, -gen_max_len:].unsqueeze(-1)).squeeze(-1)
        return new_log_probs, output.logits
    
    def process_response(self, output):
        content = ""
        for response in output.split("<|assistant|>"):
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                content = content.replace("[[训练时间]]", "2023年")
            else:
                content = {"name": metadata.strip(), "content": content}
        return content
    
    def generate_with_rlhf(self, actor_model, input_ids, query, num_beams=1, num_return_sequences=1, max_new_tokens=8):
        '''
        `params:`
            - input_ids: [batch_size, seq_len]
            - query: list, the user query content
            - num_beams: int, 3, 2 # set bigger if you have bigger compute memory
            - num_return_sequences: int, 3, 2 # set bigger if you have bigger compute memory
            - max_new_tokens: int, the max token that LLM can generate for new content
        
        `return:`
            - sequences:  the generated ids of sequences
            - log_probs:  the log_probs of the generated ids
            - gen_texts:  the generated texts of sequences, which clip with max length.
        '''
        assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
        gen_method = "greedy_search" if num_beams == 1 else "beam_search" 
        # 把问题送入模型中，获得问题的输出
        if hasattr(actor_model, 'module'):
            unwrapped_model = actor_model.module
        else:
            unwrapped_model = actor_model
        generate_ = unwrapped_model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=max_new_tokens,
                            num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                            output_hidden_states=False, return_dict_in_generate=True)
        sequences = generate_.sequences
        log_probs = self.get_log_prob(generated_outputs=generate_, input_ids=input_ids, gen_method=gen_method)
        gen_texts = self.tokenizer.batch_decode(sequences)
        gen_texts = [self.process_response(text) for text in gen_texts]

        for i, q in enumerate(query):
            cur_gen_texts = gen_texts[i * num_return_sequences : (i + 1) * num_return_sequences]
            if self.qa_logs is not None:
                if q not in self.qa_logs:
                    self.qa_logs[q] = []
                self.qa_logs[q] += cur_gen_texts # 将本query的答案保存在qa_logs中；对于同样的query，若多次生成回答，则使用extend方法进行全部存储

        return sequences, log_probs, gen_texts, None

    def generate_with_ft(self, actor_model, input_ids, last_assistant_content, gen_max_len):
        '''
        the target sentence is directly used to improve the probability of the RL. zh: 目标句直接用RL提升它的概率
        
        `params:`
            - input_ids: [batch_size, seq_len], query ids with answer
            - last_assistant_content: str, the standard answer
            - gen_max_len: str, the max length of answer ids
        
        `return:`
            - sequences:  the generated ids of sequences, in here is the original input_ids
            - log_probs:  the log_probs of the input_ids.
            - gen_texts:  the generated texts of sequences, in here is the original last_assistant_content.
        '''
        sequences = input_ids
        with torch.no_grad():
            log_probs, logits = self.get_log_probs_with_input_ids(actor_model, input_ids, gen_max_len=gen_max_len)
        should_gen_texts = last_assistant_content
        return sequences, log_probs, should_gen_texts, logits
    
    def get_decay_up_matrix_T(self, max_length=2048, gamma=0.99, tau=0.95):
        ''' 
        生成衰减矩阵
        
        `params:`
            - max_length: int
            - gamma: float
            - tau: float
        
        `return:`
            - decay_up_matrix_T: torch.Tensor
        '''
        decay = gamma * tau # 衰减系数
        decay_row = torch.ones(max_length).float() * decay
        decay_row[0] = 1
        decay_row_cross_time = decay_row.cumprod(dim=-1) # 使用cumprod进行连乘，形成(gamma*tau),(gamma*tau)^2,...,(gamma*tau)^2048这样的结构
        assert decay_row_cross_time.sign().min() == 0
        decay_up_matrix = torch.zeros((max_length, max_length)).float()
        for i in range(max_length):
            decay_row = decay_row_cross_time.roll(i)
            decay_row[:i] = 0 # 确保看不见前面的
            decay_up_matrix[i] = decay_row
        decay_up_matrix_T = decay_up_matrix.T # 先进行转置，因为后面需要用到矩阵乘法
        return decay_up_matrix_T
    
    def gae_vectorize(self, values, rewards, masks=None):
        """
        `params:`
            - values: `[batch_size, sequence_length]`, 表示各个时间步状态的状态值。
            - rewards: `[batch_size, sequence_length]`, 表示各个时间步做出的动作的奖励，对于gpt当前动作也是动作对应的下一状态。所以shape和values一样
                    **注意这里的`rewards`表示当前动作状态的`reward`**
            - masks: 由于是要对生成的`actions`做`gae`，也就是泛化优势估计，
                     所以类似以往的`mask`只需要对`padding`进行`mask`，
                     因为`padding`的`delta`会被放入加权计算，而`action`前面的`delta`，
                     由于生成的衰减矩阵就是上三角的，自然就看不到前面的。
                     `0`表示`mask`， `1`表示需要的。
        """
        action_rewards = rewards.roll(-1) # 当前状态的动作的奖励是下一个状态出现时给出的，而奖励是基于状态计算的，所以需要shift一个时间步回去
        # 为了学到最后输出的<eop>,所以给最后的状态赋予一个rewards试试
        action_rewards = (action_rewards + rewards) / 2 # 将奖励分配到最后两步

        values_estimator_1_order = action_rewards + values.roll(-1) # 这里要注意roll是循环的，所以最后一位的值可能不能用
        deltas = values_estimator_1_order - values  #必须要action+下一个时刻的值函数减去当前值函数，这是表示当前action的优势
        # 计算gae
        max_goal_length = deltas.shape[-1]
        sub_decay_up_matrix_T = self.decay_up_matrix_T[:max_goal_length, :max_goal_length].to(deltas.device)
        if masks is not None:
            deltas = deltas * masks
        gae = deltas.matmul(sub_decay_up_matrix_T)
        assert gae.shape == deltas.shape
        return gae
    
    def forward(self, is_rlhf, actor_model, reward_model, critic_model, input_ids, input_ids_without_last_turn, last_input_len, query, last_assistant_content, gold_answers, bad_answers, num_beams, num_return_sequences, ppo_epochs, **args):
        if is_rlhf:
            input_ids = input_ids_without_last_turn
            max_new_tokens = torch.max(last_input_len).item()
            sequences, log_probs, gen_texts, logits = self.generate_with_rlhf(actor_model, input_ids, query, num_beams=num_beams, num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens)
        
        else:
            max_new_tokens = torch.max(last_input_len).item()
            sequences, log_probs, gen_texts, logits = self.generate_with_ft(actor_model, input_ids, last_assistant_content, max_new_tokens)
        
        # compute reward for generated sequences
        reward = []
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            if is_rlhf:
                b_gen_texts = gen_texts[i * num_return_sequences : (i + 1) * num_return_sequences]
            else:
                b_gen_texts = [gen_texts[i]]
            b_gold_answers = gold_answers[i]
            b_bad_answers = bad_answers[i]
            b_reward = reward_model(gen_texts=b_gen_texts, gold_answers=b_gold_answers, bad_answers=b_bad_answers).unsqueeze(1)
            reward.append(b_reward)
        reward = torch.cat(reward, dim=0)
        assert reward.shape == (len(gen_texts), 1), "need unsqueeze for next scatter_"
        rewards = torch.zeros_like(sequences, dtype=reward.dtype)
        pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        masks = (sequences!=pad_id).long()
        final_position = (sequences[:,input_ids.size(-1):] != pad_id).sum(dim=-1) + input_ids.size(-1) - 1
        index = final_position.unsqueeze(-1)
        rewards.scatter_(dim=1, index=index, src=reward)
        # 确保都放到values所在的device

        torch.cuda.empty_cache()
        
        for ppo_epoch in range(ppo_epochs):
            # compute new log probs
            new_log_probs, _ = self.get_log_probs_with_input_ids(actor_model, sequences, log_probs.shape[1])
            entropy = 0 # 暂时不需要熵的约束
            # compute value
            # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
            # 生成序列同时包括state和next action
            # prepare input for critic model
            input_ids_critic = sequences
            values = critic_model(input_ids=input_ids_critic)
            # compute gae
            gae = self.gae_vectorize(values=values, rewards=rewards, masks=masks)
            advantages = gae[:, -log_probs.shape[-1]:]
            # 计算value的估计量的偏差作为actor loss
            # 以及ppo的actor_loss
            value_estimator_delta = advantages
            ratio = (new_log_probs - log_probs).exp()
            # print("reward",reward, "ratio:", ratio, sep="\n")
            if torch.isinf(ratio).any():
                break
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = value_estimator_delta.square().mean()
            loss = 0.5 * (critic_loss + actor_loss) - 0.001 * entropy
            yield loss, logits