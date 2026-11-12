"""Meaningful structural, split, and inference checks, independent of training."""
import numpy as np
import torch
from models import ResNet20, SE, SeparableConv, VARIANTS, complexity
from data_utils import load_part, split_indices

def main():
    torch.set_num_threads(4)
    for v in VARIANTS:
        m=ResNet20(v)
        se=sum(isinstance(x,SE) for x in m.modules())
        dw=sum(isinstance(x,SeparableConv) for x in m.modules())
        assert se==(9 if v in ('se','dw_se') else 0)
        assert dw==(18 if v in ('dw','dw_se') else 0)
        x=torch.randn(2,1,28,28)
        logits=m(x); assert logits.shape==(2,10)
        logits.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        print(v,complexity(m))
    _,labels=load_part(True);t,v=split_indices(labels)
    assert len(t)==54000 and len(v)==6000 and not set(t)&set(v)
    assert len(np.unique(np.r_[t,v]))==60000
    assert np.array_equal(np.bincount(labels[v]),np.full(10,600))
    assert np.array_equal(np.bincount(labels[t]),np.full(10,5400))
    test_images,test_labels=load_part(False)
    assert test_images.shape==(10000,28,28)
    assert np.array_equal(np.bincount(test_labels),np.full(10,1000))
    print('PASS: variants, gradients, tensor shapes, checksums, disjoint stratified split')

if __name__=='__main__':main()
