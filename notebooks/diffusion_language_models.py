import marimo

__generated_with = "0.23.15"
app = marimo.App(layout_file="layouts/diffusion_language_models.slides.json")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Diffusion Language Models

    - Autoregressive text generation has proved very effective, but it is inherently sequential and slow.
    - Diffusion language model speeds this up by generating tokens in parallel
    - It is also better suited to certain types of problems (e.g. Sudoku problems)
    """)
    return


@app.cell
def _(mo):

    mo.image(
        src="./gemma.jpg",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Key Papers

    - LLou, Aaron, et al. "*Discrete diffusion modeling by estimating the ratios of the data distribution.*" ICML 2024 Best Paper Award 🏆
    - Sahoo, Subham Sekhar, et al. "*Simple and effective masked diffusion language models.*" NeurIPS 2024 🎭
    - Sahoo, Subham Sekhar, et al. "*The diffusion duality.*" ICML 2025 ☯️
    - Ni, Zanlin, et al. "*The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models.*" ICML 2026 Best Paper Award 🏆
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Masked Diffusion Language Model (MDLM) 🎭

    - MDLMs generate text with a diffusion model.
    - The key challenge is applying diffusion (continuous) to discrete data.
    - The solution is randomly masking tokens, then training a model to progressively unmask them.
    """)
    return


@app.cell
def _(mo):
    mo.image(
        src="https://s-sahoo.com/mdlm/static/images/mdlm.png",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ### MDLM From Scratch 🎭

    - Let's build a simple MDLM
    - We use the **Tiny Shakespeare** dataset
    - Code notebook can be found at: [github.com/harveymannering/DiffusionLanguageModels](https://github.com/harveymannering/DiffusionLanguageModels/)
    """)
    return


@app.cell
def _():
    import random
    import torch
    import urllib.request
    import marimo as mo
    from torch import nn
    from torch import Tensor, softmax, randn
    import math
    from tqdm import tqdm
    from torchview import draw_graph

    return Tensor, mo, nn, randn, random, softmax, torch, urllib


@app.cell
def _(mo, urllib):
    filename = "tinyshakespeare.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    urllib.request.urlretrieve(url, filename)
    preview_text = "".join(open(filename).readlines()[:10])

    mo.md(f"```text\n{preview_text}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tokenizer 🎭
    As a simple solution we will model each character as a single token.
    """)
    return


@app.cell
def _():

    class Tokenizer:

        def __init__(self, path, mask_token = True):

            # Load text dataset
            with open(path) as f:
                lines = f.readlines()
            text = '\n'.join(lines)

            # Construction dictionary as: {'character' : 'int'}
            index = 0
            self.dictionary = {} 
            if mask_token == True:
                self.dictionary = {'⒨': 0} # special "[Mask]" token
                index = 1
            for char in text: 
                if char in self.dictionary:
                    continue
                self.dictionary[char] = index
                index = index + 1

        def encode(self, text):
            # Maps : text -> list(int)
            return [self.dictionary[char] for char in text]

        def decode(self, encoded) -> list[int]:
            # Maps : list(int) -> list(character)
            inv_dict = {v: k for k, v in self.dictionary.items()}
            return [inv_dict[encoding] for encoding in encoded]

        def vocab(self):
            return self.dictionary.keys()

    Tokenizer
    return (Tokenizer,)


@app.cell
def _(Tokenizer):
    # Tokenizer then un-tokenize some example text
    tokenizer = Tokenizer('tinyshakespeare.txt')
    encoded = tokenizer.encode('Hello World!')
    decoded = tokenizer.decode(encoded)

    # Print outputs
    f"Encoded: `{encoded}`", f'Decoded: `{"".join(decoded)}`'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transformer 🎭

    We will use a simple transformer for our neural network.

    ><font color="blue">Input:</font> `(Tokens Indexes, timesteps)`

    ><font color="orange">Outputs:</font> `(Categorical Distibution on Each Token)`
    """)
    return


@app.cell
def _(Tensor, Tokenizer, nn, randn, softmax):
    class Transformer(nn.Module):

        def __init__(self, mask_token=True):
            super().__init__()

            # Initialize Tokenizer
            self.encoder = Tokenizer('tinyshakespeare.txt', mask_token)

            # Define network parameters
            vocab_size = len(self.encoder.vocab())
            max_seq_len = 1024
            embed_dim = 256
            hidden_dim = 4 * embed_dim
            num_layers = 16

            # Define embedding layers for tokens, positions, and timesteps
            self.token_emb = nn.Parameter(randn(vocab_size, embed_dim))
            self.pos_emb = nn.Parameter(randn(max_seq_len, embed_dim))
            self.t_emb = nn.Linear(1, embed_dim)

            # Define network layers
            self.layers = nn.ModuleList([
                # Attention block
                nn.ModuleList([
                    nn.LayerNorm(embed_dim), 
                    nn.Linear(embed_dim, embed_dim), 
                    nn.Linear(embed_dim, embed_dim), 
                    nn.Linear(embed_dim, embed_dim), 
                    nn.LayerNorm(embed_dim), 
                    nn.Linear(embed_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, embed_dim)
                ]) 
                for _ in range(num_layers)])
            self.output = nn.Linear(embed_dim, vocab_size)

        def embedding(self, x: Tensor, mask_prob: Tensor):
            # Combine all embeddings for every token
            t_emb = self.t_emb(mask_prob)
            return self.token_emb[x] + self.pos_emb[:x.shape[-1]] + t_emb

        def attention(self, Q, K, V):  
            d_k = K.shape[-1] 
            return softmax(Q @ K.transpose(-2, -1) / d_k ** 0.5, dim=-1) @ V 

        def forward(self, x, mask_prob):
            # Convert int tokens to embeddings
            x = self.embedding(x, mask_prob)

            # Run through attention layers
            for ln1, W_Q, W_K, W_V, ln2, linear1, relu, linear2 in self.layers:
                x = x + self.attention(W_Q(ln1(x)), W_K(ln1(x)), W_V(ln1(x)))
                x = x + linear2(relu(linear1(ln2(x))))

            # Outputs categorical distribution for each token
            return self.output(x)

    return (Transformer,)


@app.cell
def _(Transformer):
    Transformer()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forward Process 🎭
    - In diffusion, the forward process corrupts our data in some way.
    - In MDLMs, the forward process randomly changes tokens to the the special "⒨" token with probability $\alpha$.
    - When $\alpha=1.0$, all tokens are the "⒨" token.
    - When $\alpha=0.0$, we have normal uncorrupted text.
    """)
    return


@app.cell
def _(mo):
    mo.image(
        src="https://s-sahoo.com/mdlm/static/images/mdlm.png",
    )
    return


@app.cell
def _(device, random, seq_len, torch, train_text, transformer):
    def grab_chunk(src):
        # Randomly sample text from shakespeare dataset
        start = random.randint(0, len(src) - seq_len)
        return torch.tensor(transformer.encoder.encode(src[start:start + seq_len]), device=device)

    def add_noise(x, t):
        # Randomly corrupt text
        mask = (torch.rand_like(x, dtype=torch.float) < t).long()
        return (x * (1 - mask), mask)

    chunk = grab_chunk(train_text)
    timestep = random.uniform(0, 1)

    f"get_chunk : {chunk}", f"timestep : {timestep:.4f}", f"masked chunk : {add_noise(chunk, timestep)[0]}"
    return add_noise, grab_chunk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training 🎭

    Randomly mask tokens. Train the network to predict clean text using cross-entropy loss:
    $$\mathcal{L}_{MDLM} = -Σ ~x_{0,token}~\log(\hat x_{0,token})$$
    where $x_{0,token}$ is the ground truth text and $\hat{x}_{0,token}$ is the network prediction.
    """)
    return


@app.cell
def _(torch):
    # LOAD DATASET
    with open('tinyshakespeare.txt') as f:
        train_text = f.read()

    # DEFINE HYPERPARAMETERS
    seq_len = 128
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #{
    #    "device": device,
    #    "seq_len": seq_len,
    #    "batch_size": batch_size,
    #}
    return batch_size, device, seq_len, train_text


@app.cell
def _(Transformer, device, torch):
    # INITIALIZE NETWORK & OPTIMIZER
    transformer = Transformer().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
    return optimizer, transformer


@app.cell
def _(
    add_noise,
    batch_size,
    device,
    grab_chunk,
    optimizer,
    random,
    torch,
    train_text,
    transformer,
):
    # TRAINING LOOP 
    for _ in range(1):
        # Sample data
        batch = torch.stack([grab_chunk(train_text) for _ in range(batch_size)])
        t = random.uniform(0, 1)
        masked, mask = add_noise(batch, t)

        # Run neural network
        preds = transformer(masked, torch.tensor([t], device=device))

        # Calculate loss and backprop
        loss = torch.nn.functional.cross_entropy(preds[mask == 1], batch[mask == 1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling 🎭

    - Generation involves gradually unmasking tokens.
    - At every step, randomly select a few masked tokens and replace them with a sample from their categorical distribution.
    - Unmasked tokens stay unmasked and do not change over generation.
    """)
    return


@app.cell
def _(mo, torch):
    def sample(model, query, length, device, total_steps=20):

        # Tokenize text input
        tokens = model.encoder.encode(query)
        x = torch.zeros(length, dtype=torch.long, device=device)
        x[:len(tokens)] = torch.tensor(tokens, device=device)
        fixed = (x != 0)

        with torch.no_grad():
            for step in range(total_steps):

                # Predict probabilites for each token at each position
                mask_prob = torch.tensor([1.0 - step / total_steps], device=device)
                probs = torch.softmax(model(x, mask_prob), dim=-1)

                mask_positions = (x == 0) & ~fixed
                if not mask_positions.any():
                    break

                # Randomly unmask each masked token with probability 1 / (remaining steps)
                unmask = mask_positions & (torch.rand(length, device=device) < 1 / (total_steps - step))
                x[unmask] = torch.multinomial(probs[unmask], 1).squeeze(-1)

                # Stream the generated text in place as it updates
                mo.output.replace(
                    mo.md(f"```text\n{''.join(model.encoder.decode(x.tolist()))}\n```")
                )

        return ''.join(model.encoder.decode(x.tolist()))

    sample
    return (sample,)


@app.cell
def _(Transformer, device, torch):
    transformer_pt = Transformer().to(device)
    state_dict = torch.load('mdlm.pt')

    # Rename the keys directly in the dictionary
    state_dict['t_emb.weight'] = state_dict.pop('mask_prob_emb.weight')
    state_dict['t_emb.bias'] = state_dict.pop('mask_prob_emb.bias')

    # Load the modified state dict
    _ = transformer_pt.load_state_dict(state_dict)
    return (transformer_pt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generation 🎭

    Let's generate a sample of text from a trained model.
    """)
    return


@app.cell
def _(device, mo, sample, transformer_pt):
    mo.md(f"""
    ```text
    {sample(transformer_pt, 'To be, ', 128, device, 1000)}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.image(
        src="https://lf3-static.bytednsdoc.com/obj/eden-cn/hyvsmeh7uhobf/20250730-185422%20(1).jpeg",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Uniform State Diffusion Models (USDM) ☯️
    - This approach replaces the corruption process
    - Instead of masking tokens, they are randomly replaced with another token
    - Unlike masked diffusion,  uniform diffusion allows tokens to change multiple times during generation
    """)
    return


@app.cell
def _(mo):

    mo.image(
        src="./masked_and_uniform_diffusion.png",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoder ☯️

    - USDMs don't need a special "mask" token, since tokens are never masked, they are corrupted to a *uniformly random* token instead.
    - So the encoder is even simpler: it just maps every character to an index starting from 0.
    """)
    return


@app.cell
def _(Tokenizer):
    # Encode then decode some example text
    tokenizer_usdm = Tokenizer('tinyshakespeare.txt', mask_token=False)
    encoded_usdm = tokenizer_usdm.encode('Hello World!')
    decoded_usdm = tokenizer_usdm.decode(encoded_usdm)

    # Print outputs
    f"Encoded: `{encoded_usdm}`", f'Decoded: `{"".join(decoded_usdm)}`'
    return (tokenizer_usdm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forward Process ☯️
    - The forward process still corrupts our data, but with a *uniform* prior instead of an *absorbing* one.
    - In USDMs, each token is randomly replaced with a token sampled uniformly from the vocabulary with probability $t$.
    - When $t=1.0$, every token is uniform noise. When $t=0.0$, we have normal uncorrupted text.
    """)
    return


@app.cell
def _(device, random, seq_len, tokenizer_usdm, torch, train_text):
    def grab_chunk_usdm(src):
        # Randomly sample text from shakespeare dataset
        start = random.randint(0, len(src) - seq_len)
        return torch.tensor(tokenizer_usdm.encode(src[start:start + seq_len]), device=device)

    def add_noise_usdm(x, t):
        # Randomly corrupt text
        vocab_size = len(tokenizer_usdm.vocab())
        mask = (torch.rand_like(x, dtype=torch.float) < t).long()
        uniform_random_tokens = torch.randint(0, vocab_size, x.shape, dtype=torch.long, device=device)
        return (x * mask + uniform_random_tokens * (1 - mask), mask)

    chunk_usdm = grab_chunk_usdm(train_text)
    timestep_usdm = random.uniform(0, 1)

    f"get_chunk : {chunk_usdm}", f"timestep : {timestep_usdm:.4f}", f"noised chunk : {add_noise_usdm(chunk_usdm, timestep_usdm)[0]}"
    return add_noise_usdm, grab_chunk_usdm


@app.cell(disabled=True)
def _(mo):

    mo.image(
        src="https://raw.githubusercontent.com/s-sahoo/duo/refs/heads/gh-pages/static/images/duo_schematic.png",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training ☯️

    USDM corrupts tokens changes a subset of token to other tokens at random. It then predict the clean, uncorrupt text using this loss function:

    \begin{aligned}
    & \mathcal{L}_{\text {Duo }}^{\text {NELBO }}\left(q, p_\theta ; \mathbf{x}\right)
    =\mathbb{E}_{t \sim \mathcal{U}[0,1], q_t} \sum_{\ell=1}^L-\frac{\alpha_t^{\prime}}{K \alpha_t}\left[\frac{K}{\overline{\mathbf{x}}_i^{\ell}}-\frac{K}{\left(\overline{\mathbf{x}}_\theta^{\ell}\right)_i}\right. -\left(\kappa_t \mathbb{1}_{\mathbf{z}_t^{\ell}=\mathbf{x}^{\ell}}+\mathbb{1}_{\mathbf{z}_t^{\ell} \neq \mathbf{x}^{\ell}}\right) \sum_{j=1}^L \log \frac{\left(\overline{\mathbf{x}}_\theta^{\ell}\right)_i}{\left(\overline{\mathbf{x}}_\theta^{\ell}\right)_j} -K \frac{\alpha_t}{1-\alpha_t} \log \frac{\left(\overline{\mathbf{x}}_\theta^{\ell}\right)_i}{\left(\overline{\mathbf{x}}_\theta^{\ell}\right)_m} \mathbb{1}_{\mathbf{z}_t^{\ell} \neq \mathbf{x}^{\ell}} \left.-\left((K-1) \kappa_t \mathbb{1}_{\mathbf{z}_t^{\ell}=\mathbf{x}^{\ell}}-\frac{1}{\kappa_t} \mathbb{1}_{\mathbf{z}_t^{\ell} \neq \mathbf{x}^{\ell}}\right) \log \kappa_t\right],
    \end{aligned}

    ...but also cross-entropy loss seems to work...
    """)
    return


@app.cell
def _(Transformer, device, torch):
    # INITIALIZE NETWORK & OPTIMIZER
    transformer_usdm = Transformer(mask_token=False).to(device)
    optimizer_usdm = torch.optim.Adam(transformer_usdm.parameters(), lr=0.0001)
    return optimizer_usdm, transformer_usdm


@app.cell
def _(
    add_noise_usdm,
    batch_size,
    device,
    grab_chunk_usdm,
    optimizer_usdm,
    random,
    torch,
    train_text,
    transformer_usdm,
):
    # TRAINING LOOP
    for i in range(1):
        # Sample data
        batch_usdm = torch.stack([grab_chunk_usdm(train_text) for _ in range(batch_size)])
        t_usdm = random.uniform(0, 1)
        noised_usdm, _ = add_noise_usdm(batch_usdm, t_usdm)

        # Run neural network
        preds_usdm = transformer_usdm(noised_usdm, torch.tensor([t_usdm], device=device))

        # Calculate loss over every position (not just the corrupted ones) and backprop
        loss_usdm = torch.nn.functional.cross_entropy(preds_usdm.permute(0, 2, 1), batch_usdm)
        loss_usdm.backward()
        optimizer_usdm.step()
        optimizer_usdm.zero_grad()

    loss_usdm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling ☯️

    - Generation still involves gradually revealing the original tokens, but every position is resampled at every step.
    - At each step, we combine the model's prediction with an analytic posterior derived from the uniform-state noise schedule (Eq. 4 in [arxiv.org/abs/2506.10892](https://arxiv.org/pdf/2506.10892)).
    - This lets already-correct tokens stay stable while noisy tokens gradually converge to the model's prediction.
    """)
    return


@app.cell
def _(mo, torch):
    class LogLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eps = 1e-3

        def forward(self, t):
            t = (1 - self.eps) * t
            alpha_t = 1 - t
            dalpha_t = - (1 - self.eps)
            return dalpha_t, alpha_t

    def sample_usdm(model, query, length, device, total_steps=20):
        import torch.nn.functional as F

        noise = LogLinear()
        vocab_size = len(model.encoder.vocab())

        # Tokenize text input, all other positions start as uniformly random tokens
        tokens = model.encoder.encode(query)
        x = torch.randint(0, vocab_size, (1, length,), dtype=torch.long, device=device)
        x[0, :len(tokens)] = torch.tensor(tokens, device=device)
        prompt_mask = torch.arange(x.shape[1], device=device) >= len(tokens)

        with torch.no_grad():
            eps = 1e-5
            timesteps = torch.linspace(1, eps, total_steps + 1, device=device)
            for step in range(total_steps):

                # Predict probabilites for each token at each position
                t = timesteps[step].view(-1)
                probs = torch.softmax(model(x, t), dim=-1)[0]

                _, alpha_t = noise(t)
                alpha_s = torch.ones_like(alpha_t) if step == total_steps - 1 else noise(t - (1 - eps) / total_steps)[1]
                alpha_ts, d_alpha = alpha_t / alpha_s, alpha_s - alpha_t

                # Equation 4 in https://arxiv.org/pdf/2506.10892, for every position at once
                x_one_hot = F.one_hot(x[0], vocab_size)
                posterior = (alpha_t * vocab_size * x_one_hot * probs + (alpha_ts - alpha_t) * probs + d_alpha * x_one_hot + (1 - alpha_ts) * (1 - alpha_s) / vocab_size) / (alpha_t * vocab_size * x_one_hot.gather(-1, x[0, :, None]) + (1 - alpha_t))

                # Sample non-prompt tokens from their categorical distributions
                x[0, prompt_mask] = torch.multinomial(posterior[prompt_mask], 1).squeeze(-1)

                # Stream the generated text in place as it updates
                mo.output.replace(
                    mo.md(f"```text\n{''.join(model.encoder.decode(x[0].tolist()))}\n```")
                )

        return ''.join(model.encoder.decode(x[0].tolist()))

    sample_usdm
    return (sample_usdm,)


@app.cell
def _(Transformer, device, torch):
    transformer_usdm_pt = Transformer(mask_token=False).to(device)
    state_dict_usdm = torch.load('usdm.pt')

    # Keys already match USDMTransformer's parameter names, no renaming needed
    _ = transformer_usdm_pt.load_state_dict(state_dict_usdm)
    return (transformer_usdm_pt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generation ☯️

    Let's generate a sample of text from a trained USDM model.
    """)
    return


@app.cell
def _(device, mo, sample_usdm, transformer_usdm_pt):
    mo.md(f"""
    ```text
    {sample_usdm(transformer_usdm_pt, 'To be, ', 128, device, 1000)}
    ```
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
