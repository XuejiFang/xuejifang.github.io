# Your Group-Relative Advantage Is Biased
> 作者：Fengkai Yang1,3,4, Zherui Chen2, Xiaohan Wang4, Xiaodong Lu1,4, Jiajun Chai4, Guojun Yin4, Wei Lin4, Shuai Ma1, Fuzhen Zhuang1, Deqing Wang1, Yaodong Yang3, Jianxin Li1, Yikun Ban1
> 
> 单位：1 北航，2 UCB，3 北大，4 美团
## 摘要 
They proposed a optimaization algorithm that adaptively reweights advantage estimates to mitigate the bias induced by group-based advantage estimation. 

They fisrtly proved that the expectation of the group-based advantage $\hat{A}_{t,i}$ is lower than the true advantage $A_{t,i}$ for difficult prompts, and larger for easy prompts. (Theorem 1)

Then they provide a distribution-levle characterization of how likely group-relative advantage estimation is to underestimate or overestimate the true advantage, depending pn prompt difficulty. (Theorem 2)

Since the group-based advantage estimator is biased, we propose an algorithm to adjust the advantage estimation accordingly. The proposed approach consists of two key components. First, we introduce a framework that incorporates crossbatch information into RL training, enabling a history-aware anchor for prompt difficulty. Second, we design an adaptive advantage reweighting algorithm to correct the induced bias.

Finally, they theoretically analysis that, with an appropriate choice of a scaling parameter $\lambda_{scale}$, the HA-DW adjustment yields advantage estimatets that are closer to the true advantage $A_{t,i}$ in expectation. (Theorem 3)

Additionally, they extend their analysis to continuous bounded reward distributions, which suggests that the core bias phenomenon is not an artifact of the Bernoulli reward assumption but is prevalent across a borader class of bounded reward models. (Theorem 4)

## 符号定义
$t$, traing step

$D$, traning dataset

$x_t$, a prompt sampled from $D$ in step $t$

$G$, number of responses to $x_t$, i.e. Group Size

$R$, total reward within the group, $R=\sum_{i=1}^G r_{t,i}$

$\{y_{t,i}\}_{i=1}^G$, the responses to $x_t$

$r_{t,i}$, scalar reward for response $y_{t,i}$, $r_{t,i}\in \{0, 1\}$

$\pi_{\theta}$, policy model, i.e. LLM

## 定理 Theorem

### 定理1
Given a prompt $x_t \sim D$,let $y_{t,i} \sim \pi_{\theta}(\cdot|x_t)$ denote a sample response with reward $r_{t,i}$. Suppose $G\ge 2$, and condition on the event $S=\{1\le R \le G-1\}$. Then, for any $i\in [G]$, we have:

$\mathbf{E}[\hat{A}_{t,i}|S] < A_{t,i}$, if $p_t < 0.5$;

$\mathbf{E}[\hat{A}_{t,i}|S] > A_{t,i}$, if $p_t > 0.5$;

$\mathbf{E}[\hat{A}_{t,i}|S] = A_{t,i}$, if and only if $p_t = 0.5$.

### 定理2
Under the condition of Theorem 1, suppose $x_t$ is a hard prompt ($p_t < 0.5$). Then, for any $\epsilon \in (0, \mathbb{E}[\hat{p}_t|\mathcal{S}] - p_t)$, we have:

$\mathbb{P}(A_{t,i} - \hat{A}_{t,i} > \epsilon | \mathcal{S}) = \frac{\sum_{k=\lfloor G(p_t+\epsilon)\rfloor+1}^{G-1} \binom{G}{k} p_t^k (1-p_t)^{G-k}}{1 - (1-p_t)^G - p_t^G}$.

Simlarily, suppose $x_t$ is an easy prompt ($p_t > 0.5$). Then, for any $\epsilon \in (0, p_t - \mathbb{E}[\hat{p}_t|\mathcal{S}])$, we have:

$\mathbb{P}(\hat{A}_{t,i} - A_{t,i} > \epsilon | \mathcal{S}) = \frac{\sum_{k=1}^{\lceil G(p_t-\epsilon)\rceil-1} \binom{G}{k} p_t^k (1-p_t)^{G-k}}{1 - (1-p_t)^G - p_t^G}$.

### 定理3
Under the condition of Lemma 1, suppose there exists a scaling factor $\lambda_{scale}$ in Equation (16) such that:

$\lambda_{scale} \in (\frac{1+\frac{(1-c_{high})\hat{p}_t}{1-\hat{p}_t}}{\exp(D_{t,i} M_t)}, \frac{1+\frac{(1-c_{low})\hat{p}_t}{1-\hat{p}_t}}{\exp(D_{t,i} M_t)}) \cup (\frac{c_{low}}{\exp(D_{t,i} M_t)}, \frac{c_{high}}{\exp(D_{t,i} M_t)})$.

Then, HA-DW algorithm provably mitigates the bias of group-relative advantage:

$|\mathbb{E}[\hat{A}_{t,i} \cdot \Phi_{t,i} | \mathcal{S}] - A_{t,i}| < |\mathbb{E}[\hat{A}_{t,i} | \mathcal{S}] - A_{t,i}|$.


### 定理4
At training step $t$ and let $G \ge 2$, with CDF $F$ and PDF $f$. Given a prompt $x_t \sim \mathcal{D}$ and draw $G \ge 2$ i.i.d. rewards:
$r_{t,1}, ..., r_{t,G} \sim \mathcal{D}(p_t)$.

And we extend the binary reward setting to non-binary rewards: $r_{t,i} \in [0, 1]$.
The group-relative advantage can be denoted as:
$\hat{A}_{t,i} := r_{t,i} - \hat{p}_t$, $\hat{p}_t = \frac{1}{G} \sum_{i=1}^G r_{t,i}$,
while the expected advantage is defined as:
$A_{t,i} := r_{t,i} - p_t$.

Fix a constant $\sigma \in [0, 1]$ and define the update event:
$S_{\sigma} := \{\exists i \ne j : |r_{t,i} - r_{t,j}| > \sigma\} \Rightarrow S_{\sigma}^c = \{\max_i r_{t,i} - \min_i r_{t,i} \le \sigma\}$.

For $u \in [0, 1]$, define $u^+ := \min\{1, u+\sigma\}$, we have:
$q(u) := F(u^+) - F(u)$,
and:
$m(u) := \mathbb{E}[r_{t,1} | u \le r_{t,1} \le u^+] = \frac{\int_u^{u^+} x f(x) dx}{F(u^+) - F(u)}$ (when $q(u) > 0$).

Then the probability of a non-update is:
$\mathbb{P}(S_{\sigma}^c) = G \int_0^1 f(u) q(u)^{G-1} du$,
and:
$\mathbb{P}(S_{\sigma}) = 1 - \mathbb{P}(S_{\sigma}^c)$.

Moreover, we have:
$\mathbb{E}[\hat{p}_t | S_{\sigma}] = \frac{p_t - \mathbb{E}[\hat{p}_t \cdot 1_{\{S_{\sigma}^c\}}]}{\mathbb{P}(S_{\sigma})}$
with:
$\mathbb{E}[\hat{p}_t \cdot 1_{\{S_{\sigma}^c\}}] = \int_0^1 (u + (G-1)m(u)) f(u) q(u)^{G-1} du$.

Finally, the conditional bias transferred to advantages satisfies, for all $i$, we have:
$\mathbb{E}[\hat{A}_{t,i} - A_{t,i} | S_{\sigma}] = p_t - \mathbb{E}[\hat{p}_t | S_{\sigma}]$.



## 引理 Lemma

### 引理1
> Baseline Rectification

Given a prompt $x_t \sim D$ and the policy $\pi_{\theta_t}$, let $\tilde{p}_t = c \cdot \hat{p}_t$ be the rectified group baseline. Assume $p_t \in [\Delta, 1-\Delta]$ for some $\Delta \in (0, 1/2]$. Given any $\delta \in (0, 1)$, we can define that:

$\epsilon_{\delta} := \sqrt{\frac{1}{2G} \log(\frac{2}{\delta(1-(1-\Delta)^G-\Delta^G)})}$.

Let $I_t := [\hat{p}_t - \epsilon_{\delta}, \hat{p}_t + \epsilon_{\delta}] \cap [\Delta, 1-\Delta]$, $A(p) := 1 - (1-p)^G - p^G$.

Fix any $\epsilon > 0$, we define:

$c_{low} := \sup_{p \in I_t} \frac{(p-\epsilon) A(p)}{p(1-p^{G-1})}$,
$c_{high} := \inf_{p \in I_t} \frac{(p+\epsilon) A(p)}{p(1-p^{G-1})}$.

Then, with probability at least $1-\delta$ conditional on $\mathcal{S}$, for any choice $c \in (c_{low}, c_{high})$, we can derive that:

$\mathbb{E}[\tilde{p}_t | \mathcal{S}] \in (p_t - \epsilon, p_t + \epsilon)$.


### 引理2
Under the condition of Theorem 1, the bias induced by the group-relative advantage is formulated as:

$A_{t,i} - \mathbb{E}[\hat{A}_{t,i}|\mathcal{S}] = \frac{p_t(1-p_t)^G + p_t^{G+1} - p_t^G}{1 - (1-p_t)^G - p_t^G}$.





### 引理3
Define the non-degenerate event $\mathcal{S} := \{1 \le S \le G-1\}$, and $\epsilon \in (0, |p_t - \hat{p}_t|)$. If

$c \in (\frac{(p_t - \epsilon) \cdot (1 - (1-p_t)^G - p_t^G)}{p_t (1 - p_t^{G-1})}, \frac{(p_t + \epsilon) \cdot (1 - (1-p_t)^G - p_t^G)}{p_t (1 - p_t^{G-1})})$,

we have:
$\mathbb{E}[\tilde{p}_t | \mathcal{S}] \in (p_t - \epsilon, p_t + \epsilon)$.

### 引理4
> $p_t$-free concentration under S

Define the non-degenerate event $\mathcal{S} := \{1 \le S \le G-1\}$. Assume $p_t \in [\Delta, 1-\Delta]$ for some $\Delta \in (0, 1/2]$. Then for any $\zeta > 0$, we have:

$\mathbb{P}(|\hat{p}_t - p_t| < \zeta | \mathcal{S}) \ge \frac{1 - 2 \exp(-2G\zeta^2) - (1-\Delta)^G - \Delta^G}{1 - (1-\Delta)^G - \Delta^G}$.

### 引理5
> Conditional $p_t$-free concentration under S

Assume $p_t \in [\Delta, 1-\Delta]$ for some $\Delta \in (0, 1/2]$. Then for any $\delta \in (0, 1)$, with probability at least $1-\delta$ conditional on $\mathcal{S}$, we have:

$|\hat{p}_t - p_t| < \sqrt{\frac{1}{2G} \log(\frac{2}{\delta(1 - (1-\Delta)^G - \Delta^G)})}$.

### 引理6
> A $p_t$-free feasible range of c expressed via $\hat{p}_t$

Assume the conditions of Lemma 4 and define:

$\epsilon_{\delta} := \sqrt{\frac{1}{2G} \log(\frac{2}{\delta(1 - (1-\Delta)^G - \Delta^G)})}$.

Let:
$I_t := [\hat{p}_t - \epsilon_{\delta}, \hat{p}_t + \epsilon_{\delta}] \cap [\Delta, 1-\Delta]$,
$A(p) := 1 - (1-p)^G - p^G$.

Fix any $\epsilon > 0$, we define:

$c_{low} := \sup_{p \in I_t} \frac{(p-\epsilon) A(p)}{p(1-p^{G-1})}$,
$c_{high} := \inf_{p \in I_t} \frac{(p+\epsilon) A(p)}{p(1-p^{G-1})}$.

Then, on the event $\{|\hat{p}_t - p_t| < \epsilon_{\delta}\}$ (which holds with probability at least $1-\delta$ conditional on $\mathcal{S}$), any choice

$c \in (c_{low}, c_{high})$

implies that the condition (69) holds for the true $p_t$, and hence:

$\mathbb{E}[\tilde{p}_t | \mathcal{S}] \in (p_t - \epsilon, p_t + \epsilon)$.




## 推论 Corollary

### 推论1
Under the condition of Theorem 2, suppose the group size satisfies $2 \le G \le 8$, and assume that $p_t$ is uniformly distributed over $[0, 1]$. Then, for any $i \in [G]$, the following inequalities hold:

$\mathbb{P}(\hat{A}_{t,i} < A_{t,i} | \mathcal{S}, p_t < 0.5) > 0.63$,

$\mathbb{P}(\hat{A}_{t,i} > A_{t,i} | \mathcal{S}, p_t > 0.5) > 0.63$,

$\mathbb{P}(\hat{A}_{t,i} < A_{t,i} | \mathcal{S}, p_t < 0.25) > 0.78$,

$\mathbb{P}(\hat{A}_{t,i} > A_{t,i} | \mathcal{S}, p_t > 0.75) > 0.78$,

$\mathbb{P}(\hat{A}_{t,i} < A_{t,i} | \mathcal{S}, p_t < 0.125) = 1.00$,

$\mathbb{P}(\hat{A}_{t,i} > A_{t,i} | \mathcal{S}, p_t > 0.875) = 1.00$.

### 推论2
Under the condition of Corollary 1, suppose $G \ge 6$. The following inequalities hold:

$\mathbb{P}(\hat{A}_{t,i} < A_{t,i} | \mathcal{S}, p_t < \frac{2}{G}) > 0.78$,

$\mathbb{P}(\hat{A}_{t,i} > A_{t,i} | \mathcal{S}, p_t > \frac{G-2}{G}) > 0.78$.

### 推论3
Under the condition of Theorem 2, suppose $G \ge 2$. Then, for any $i \in [G]$, the following inequalities hold surely:

$\hat{A}_{t,i} < A_{t,i}$, if $p_t < \frac{1}{G}$,
$\hat{A}_{t,i} > A_{t,i}$, if $p_t > \frac{G-1}{G}$.



### 推论4
For $Beta(\alpha, \beta)$ reward distribution, the Beta density is:
$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$,
and the CDF is:
$F(x) = I_x(\alpha, \beta)$ for $x \in [0, 1]$,
where $B(\cdot, \cdot)$ is the Beta function and $I_x(\alpha, \beta)$ is the regularized incomplete beta function. In particular:
$p_t = \mathbb{E}[r_{t,1}] = \frac{\alpha}{\alpha+\beta}$.

Moreover, we have:
$q(u) = F(u^+) - F(u) = I_{u^+}(\alpha, \beta) - I_u(\alpha, \beta)$,
and the conditional mean over $[u, u^+]$ admits the closed form:
$m(u) = \frac{B_{u^+}(\alpha+1, \beta) - B_u(\alpha+1, \beta)}{B_{u^+}(\alpha, \beta) - B_u(\alpha, \beta)}$.

Consequently, substituting $F, f, q, m$ into conclusions obtained earlier yields explicit one-dimensional integral expressions (in standard special functions) for $\mathbb{P}(S_{\sigma}^c)$ and $\mathbb{E}[\hat{p}_t | S_{\sigma}]$.

### 推论5
Let the reward $Z_{t,1}, ..., Z_{t,G}$ be i.i.d. $\mathcal{N}(\mu, \xi^2)$ with $\xi > 0$, and define $r_{t,i}$ to be properly truncated to $[0, 1]$, i.e. $r_{t,i}$ has the conditional law:
$r_{t,i} = Z_{t,i} | (0 \le Z_{t,i} \le 1)$, $i=1, ..., G$.

Let $u^+ := \min\{1, u+\sigma\}$ and define, for $u \in [0, 1]$ with $q(u) > 0$ we have:
$q(u) := \mathbb{P}(u \le r_{t,1} \le u^+)$,
and:
$m(u) := \mathbb{E}[r_{t,1} | u \le r_{t,1} \le u^+]$.

Let $\Phi$ and $\varphi$ be the standard normal CDF and PDF, and set:
$a := \frac{0-\mu}{\xi}$, $b := \frac{1-\mu}{\xi}$.

Then the truncated-normal density on $[0, 1]$ is:
$f(x) = \frac{\varphi(\frac{x-\mu}{\xi})}{\sigma(\Phi(b) - \Phi(a))} 1_{[0,1]}(x)$.

Its CDF on $[0, 1]$ is:
$F(x) = \frac{\Phi(\frac{x-\mu}{\xi}) - \Phi(a)}{\Phi(b) - \Phi(a)}$.

The mean satisfies:
$p_t = \mathbb{E}[r_{t,1}] = \mu + \xi \frac{\varphi(a) - \varphi(b)}{\Phi(b) - \Phi(a)}$.

Moreover:
$q(u) = F(u^+) - F(u)$,
and the conditional mean over $[u, u^+]$ has the standard truncated-normal form:
$m(u) = \mu + \sigma \frac{\varphi(\frac{u-\mu}{\xi}) - \varphi(\frac{u^+-\mu}{\xi})}{\Phi(\frac{u^+-\mu}{\xi}) - \Phi(\frac{u-\mu}{\xi})}$.

Consequently, substituting $F, f, q, m$ to yield explicit one-dimensional integral expressions for $\mathbb{P}(S_{\sigma}^c)$ and $\mathbb{E}[\hat{p}_t | S_{\sigma}]$ in terms of $\Phi$ and $\varphi$.



## 实验
![训练流程](assets/pipe.png)
### 实验设置
**模型**：Qwen34B-Base, Qwen3-8B-Base and LLaMA-3.2-3B-Instruct

**benchmark**：MATH500, AIME25, AMC23, Minerva, OlympiadBench

**对比算法**：GRPO，GSPO，DAPO

**实验开销**：8 × NVIDIA A100 GPUs

**训练代码**：暂未开源，基于VeRL framework

![实验配置](assets/exp_hyperparam.png)

### 对比实验
![与其他算法对比](assets/exp_comp.png)
![在小group size实现更优效果](assets/exp_improve.png)

### 消融实验
![对lambda网格搜索](assets/exp_ab1.png)

![动态Ct与固定对比](assets/exp_ab2.png)


