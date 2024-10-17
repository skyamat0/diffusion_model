import torch

def langevin_step(state: torch.Tensor, score: function ,alpha: float):
    """
    Description:
        ランジュバン・ダイナミクスに従って時間発展させる関数

    Input:
        state: 現在の状態（d次元ベクトル）
        score: スコア関数（対数尤度の入力についての勾配，ベクトル値関数）
        alpha: ステップ幅（0に近いほど良いサンプリングとなる，その代わりKを増やす必要がある）

    Output:
        next_state: 次の状態
    """

    noise = torch.randn_like(state)
    next_state = state + (alpha ** 2 / 2) * score(state) + alpha * noise
    return next_state

def langevin_sampling(init_state: torch.Tensor, score: function, K: int):
    """
    Description:
        ランジュバン・モンテカルロ法によってサンプリングする関数

    Input:
        init_state: 初期状態（d次元ベクトル）
        score: スコア関数（対数尤度の入力についての勾配，ベクトル値関数）
        alpha: ステップ幅（0に近いほど良いサンプリングとなる，その代わりKを増やす必要がある）

    Output:
        result: サンプリング結果
    """
    current_state = init_state
    for i in range(K):
        current_state = langevin_step(current_state)
    return current_state
    