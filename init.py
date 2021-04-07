import os

if not os.path.exists('result/'):
    os.mkdir('result/')
if not os.path.exists('result/model_states/'):
    os.mkdir('result/model_states/')
if not os.path.exists('result/predictions/'):
    os.mkdir('result/predictions/')
if not os.path.exists('result/features/'):
    os.mkdir('result/features/')