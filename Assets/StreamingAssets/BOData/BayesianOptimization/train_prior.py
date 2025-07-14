#!/usr/bin/env python3
"""
Train a global LinearMixedGP prior from historical users and save hyper-params.

Expected CSV columns:
    user_id, y1 … yM , u1 … u6 , x1 … xD
One CSV per past participant or one big file – pass paths via CLI.

Run once, e.g.:
python train_prior.py past_user_*.csv --m 2 --save global_prior.pt
"""
import argparse, torch, pandas as pd
from pathlib import Path
from botorch.models import LinearMixedGP, ModelListGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import settings as gpy_settings
import pickle, sys

def read_csv(path: Path):
    df = pd.read_csv(path, sep=';')
    user = torch.tensor(df['user_id'].values, dtype=torch.long)
    y     = torch.tensor(df.iloc[:,1:1+M].values, dtype=torch.double)
    u_x   = torch.tensor(df.iloc[:,1+M:].values, dtype=torch.double)  # (u||x)
    return user, u_x, y

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--m", type=int, required=True, help="# objectives")
    p.add_argument("--save", default="global_prior.pt")
    args = p.parse_args()

    M = args.m
    users, Xs, Ys = [], [], []
    for pth in args.csvs:
        usr, X, Y = read_csv(Path(pth))
        users.append(usr);  Xs.append(X);  Ys.append(Y)

    user_ids = torch.cat(users)
    train_X  = torch.cat(Xs)
    train_Ys = [torch.cat([y[:,i].unsqueeze(-1) for y in Ys]) for i in range(M)]

    # one LinearMixedGP per objective
    gplist = []
    for i in range(M):
        model = LinearMixedGP(train_X, train_Ys[i],
                              random_effects=user_ids,
                              outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpy_settings.fast_computations(): model.fit_mll(mll)
        gplist.append(model)
        print(f"Objective {i+1}: fitted")

    state_dicts = [m.state_dict() for m in gplist]
    pickle.dump(state_dicts, open(args.save,'wb'))
    print(f"Saved prior hyper-parameters to {args.save}")
