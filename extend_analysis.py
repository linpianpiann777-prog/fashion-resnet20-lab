"""Additional descriptive analyses of frozen checkpoints; no training."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_utils import load_part,tensor_images,CLASSES_ZH
from models import ResNet20
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
A=json.loads((R/'results/analysis.json').read_text()); runs=json.loads((R/'results/summary.json').read_text())
variants=['baseline','dw','se','dw_se']; names=['ResNet-20','DW','SE','DW+SE']; colors=['#0072B2','#D55E00','#009E73','#CC79A7']
out={'source_sha256':{},'selected':{},'paired_gain_pp':{},'calibration':{},'risk_coverage':{}}
def save(fig,name):
 fig.tight_layout();fig.savefig(R/'figures'/f'{name}.png',dpi=220,bbox_inches='tight');fig.savefig(R/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)
preds={}
for v in variants:
 seed=A['aggregates'][v]['selected_seed'];path=R/'results'/f'{v}_seed{seed}'/'test_predictions.npz';d=np.load(path)
 probs=d['probabilities'].astype(float);y=d['labels'];assert probs.shape==(10000,10);assert np.allclose(probs.sum(1),1,atol=1e-6)
 preds[v]=(y,probs);out['selected'][v]=seed;out['source_sha256'][str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
fig,ax=plt.subplots(figsize=(7,3.7))
for j,(b,v) in enumerate([('baseline','se'),('dw','dw_se')]):
 gains=[100*(next(r['test_accuracy'] for r in runs if r['variant']==v and r['seed']==s)-next(r['test_accuracy'] for r in runs if r['variant']==b and r['seed']==s)) for s in [42,43,44]]
 out['paired_gain_pp'][v]=gains
 for i,g in enumerate(gains):ax.scatter(j+(i-1)*.09,g,color=colors[i],marker=['o','s','^'][i],label=f'seed {42+i}' if j==0 else None);ax.annotate(f'{g:+.2f}',(j+(i-1)*.09,g),xytext=(6,4),textcoords='offset points')
ax.axhline(0,color='gray',ls='--');ax.set(xticks=[0,1],xticklabels=['SE − 原版','DW+SE − DW'],ylabel='准确率变化（百分点）',xlim=(-.5,1.5));ax.legend();save(fig,'paired_se_gain')
f1=np.array([np.mean([r['per_class_f1'] for r in runs if r['variant']==v],axis=0) for v in variants]);out['per_class_f1_mean']=f1.tolist()
fig,ax=plt.subplots(figsize=(9,3.4));im=ax.imshow(f1*100,cmap='YlGnBu',vmin=75,vmax=100,aspect='auto')
ax.set(xticks=range(10),xticklabels=CLASSES_ZH,yticks=range(4),yticklabels=names);ax.tick_params(axis='x',rotation=30)
for i in range(4):
 for j in range(10):ax.text(j,i,f'{f1[i,j]*100:.1f}',ha='center',va='center',color='white' if f1[i,j]>.93 else 'black',fontsize=9)
fig.colorbar(im,ax=ax,label='三个种子的平均 F1（%）');save(fig,'class_f1')
fig,axes=plt.subplots(1,2,figsize=(9,3.6));fig2,ax2=plt.subplots(figsize=(7,3.6))
cms={}
for k,v in enumerate(variants):
 y,p=preds[v];conf=p.max(1);correct=p.argmax(1)==y;bins=np.minimum((conf*10).astype(int),9);rows=[]
 for b in range(10):
  take=bins==b;n=int(take.sum());rows.append({'lower':b/10,'upper':(b+1)/10,'n':n,'confidence':float(conf[take].mean()) if n else None,'accuracy':float(correct[take].mean()) if n else None})
 ece=sum(r['n']/len(y)*abs(r['accuracy']-r['confidence']) for r in rows if r['n']);brier=float(np.mean(np.sum((p-np.eye(10)[y])**2,axis=1)))
 out['calibration'][v]={'bins':rows,'ece':ece,'brier':brier};assert sum(r['n'] for r in rows)==10000
 good=[r for r in rows if r['n']];axes[0].plot([r['confidence'] for r in good],[r['accuracy'] for r in good],marker=['o','s','^','D'][k],color=colors[k],label=names[k])
 axes[1].plot([r['lower']+.05 for r in rows],[r['n'] for r in rows],marker=['o','s','^','D'][k],color=colors[k])
 order=np.argsort(-conf,kind='stable');sorted_conf=conf[order];end=np.r_[np.flatnonzero(sorted_conf[:-1]!=sorted_conf[1:]),len(y)-1];counts=end+1;risks=np.cumsum(~correct[order])[end]/counts;coverage=counts/len(y)
 out['risk_coverage'][v]={'threshold':sorted_conf[end].tolist(),'coverage':coverage.tolist(),'risk':risks.tolist()};ax2.plot(coverage*100,risks*100,color=colors[k],ls=['-','--','-.',':'][k],label=names[k])
 cm=np.zeros((10,10),int);np.add.at(cm,(y,p.argmax(1)),1);cms[v]=cm
axes[0].plot([0,1],[0,1],'k--',lw=1);axes[0].set(xlabel='分箱平均置信度',ylabel='分箱实际准确率',xlim=(0,1),ylim=(0,1));axes[0].legend(fontsize=8);axes[1].set(xlabel='置信度箱中心（宽度 0.1）',ylabel='测试样本数');save(fig,'calibration')
ax2.set(xlabel='自动接受覆盖率（%）',ylabel='已接受样本错误率（%）',xlim=(0,100));ax2.legend();save(fig2,'risk_coverage')
delta=cms['se']-cms['baseline'];out['confusion_delta']=delta.tolist();assert delta.sum()==0
fig,ax=plt.subplots(figsize=(7,5.7));lim=max(abs(delta.min()),abs(delta.max()));im=ax.imshow(delta,cmap='RdBu_r',vmin=-lim,vmax=lim)
ax.set(xticks=range(10),xticklabels=CLASSES_ZH,yticks=range(10),yticklabels=CLASSES_ZH,xlabel='预测类别',ylabel='真实类别');ax.tick_params(axis='x',rotation=45)
for i in range(10):
 for j in range(10):ax.text(j,i,f'{delta[i,j]:+d}' if delta[i,j] else '0',ha='center',va='center',fontsize=8,color='white' if abs(delta[i,j])>lim*.6 else 'black')
fig.colorbar(im,ax=ax,label='SE − 原版（样本数）');save(fig,'confusion_delta')
torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';model=ResNet20('se').to(device);checkpoint=R/'results/se_seed44/best.pt';model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['state_dict']);model.eval();images,labels=load_part(False);weights=[]
handle=model.blocks[-1].attention.fc.register_forward_hook(lambda m,i,o:weights.append(o.detach().cpu().numpy().reshape(-1,64)))
with torch.no_grad():
 for start in range(0,len(images),256):model(tensor_images(images[start:start+256],device))
handle.remove();weights=np.concatenate(weights);means=np.stack([weights[labels==c].mean(0) for c in range(10)]);assert weights.shape==(10000,64) and weights.min()>=0 and weights.max()<=1
out['se_channel_weights']={'run':'se_seed44','block':8,'samples':'all 10000 official test samples, original order','class_counts':np.bincount(labels).tolist(),'class_means':means.tolist(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
np.savez_compressed(R/'results/se_channel_weights.npz',weights=weights,labels=labels)
fig,ax=plt.subplots(figsize=(9,3.5));im=ax.imshow(means,aspect='auto',cmap='viridis',vmin=0,vmax=1);ax.set(yticks=range(10),yticklabels=CLASSES_ZH,xlabel='末尾 SE 模块通道编号（0–63）');fig.colorbar(im,ax=ax,label='类别平均门控权重');save(fig,'se_channels')
def flow(name,steps):
 fig,ax=plt.subplots(figsize=(9,4.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 for i,s in enumerate(steps):
  y=.91-i*.155;ax.text(.5,y,s,ha='center',va='center',bbox=dict(boxstyle='round,pad=.55',fc='#E6EFE9',ec='#276653'),fontsize=11)
  if i<len(steps)-1:ax.annotate('',xy=(.5,y-.105),xytext=(.5,y-.045),arrowprops=dict(arrowstyle='->',color='#276653',lw=1.5))
 save(fig,name)
flow('training_flow',['官方数据下载与 MD5 校验','训练集分层划分：54000 训练 / 6000 验证；测试集封存','4 种结构 × 3 个种子；仅训练集反向传播','仅按验证准确率选取最佳 checkpoint','固定权重 → 官方测试集 10000 张 → 指标与逐样本概率','一致性核验 → 消融与成本分析 → 图表和报告'])
flow('inference_flow',['测试样本切换 / 上传单件服饰图像','灰度化 → 等比例缩放和补边 → 28×28 → 归一化','加载选定模型的验证集最优权重；eval 模式推理','Softmax 概率 → 最大概率 q 与预测类别','阈值判断：q ≥ τ 自动接受；q < τ 待人工复核','显示类别、Top-5、状态；调整阈值仅改变接受状态'])
out['checks']={'prediction_shapes_and_probability_sums':True,'calibration_bin_counts':True,'confusion_difference_sum':True,'se_weight_shape_range':True,'no_training_performed':True}
(R/'results/extended_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print({v:out['calibration'][v]['ece'] for v in variants})
