"""Generate all report metrics and figures exclusively from completed run artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import VARIANTS
from data_utils import ROOT, CLASSES, load_part

NAMES={'baseline':'ResNet-20','dw':'DW-ResNet-20','se':'SE-ResNet-20','dw_se':'DW+SE-ResNet-20'}
COLORS=['#267568','#c88928','#526ca2','#ab5269']


def main():
    results=ROOT/'results'; figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    runs=json.loads((results/'summary.json').read_text())
    runtime=json.loads((results/'runtime.json').read_text())
    assert runtime['complete'] and len(runs)==12
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    manifest={}
    def save(fig,name):
        fig.savefig(figures/f'{name}.png',dpi=180)
        fig.savefig(figures/f'{name}.svg')
        manifest[name]={'source':'results/*/history.csv or test_predictions.npz; results/summary.json',
                        'dpi':180,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),
                        'sha256':hashlib.sha256((figures/f'{name}.png').read_bytes()).hexdigest()}
        plt.close(fig)
    agg={}
    for variant in VARIANTS:
        group=[r for r in runs if r['variant']==variant]
        assert len(group)==3
        agg[variant]={k:group[0][k] for k in ('parameters','macs','flops')}
        for metric in ('test_accuracy','macro_f1','train_seconds','latency_ms_median'):
            values=[r[metric] for r in group]
            agg[variant][metric+'_mean']=float(np.mean(values))
            agg[variant][metric+'_sd']=float(np.std(values,ddof=1))
        selected=max(group,key=lambda r:r['best_validation_accuracy'])
        agg[variant]['selected_seed']=selected['seed']
    # Every saved prediction must reproduce the reported accuracy and confusion matrix.
    for r in runs:
        run=results/f'{r["variant"]}_seed{r["seed"]}'
        pred=np.load(run/'test_predictions.npz');y=pred['labels'];prob=pred['probabilities']
        assert np.allclose(prob.sum(1),1,atol=1e-6)
        cm=np.bincount(y*10+prob.argmax(1),minlength=100).reshape(10,10)
        assert np.array_equal(cm,np.loadtxt(run/'confusion_matrix.csv',delimiter=',',dtype=int))
        assert np.trace(cm)/10000==r['test_accuracy']
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for v,color,marker in zip(VARIANTS,COLORS,['o','s','^','D']):
        histories=[]
        for r in runs:
            if r['variant']==v:
                with (results/f'{v}_seed{r["seed"]}'/'history.csv').open() as f:histories.append(list(csv.DictReader(f)))
        for ax,key,scale in zip(axes,['train_loss','validation_accuracy'],[1,100]):
            a=np.array([[float(row[key])*scale for row in h] for h in histories])
            epochs=np.arange(1,a.shape[1]+1);mean=a.mean(0);sd=a.std(0,ddof=1)
            ax.plot(epochs,mean,label=NAMES[v],color=color,marker=marker,markevery=4,markersize=4)
            ax.fill_between(epochs,mean-sd,mean+sd,color=color,alpha=.12)
            ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].set_ylabel('Training cross-entropy');axes[1].set_ylabel('Validation accuracy (%)')
    axes[1].legend(fontsize=8);fig.suptitle('Training dynamics | mean ± SD across 3 seeds');save(fig,'learning_curves')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
    for ax,metric,label,scale in zip(axes,['parameters','macs','test_accuracy_mean'],['Parameters (thousands)','Conv/Linear MACs (millions)','Test accuracy (%)'],[1e-3,1e-6,100]):
        values=[agg[v][metric]*scale for v in VARIANTS]
        ax.bar(range(4),values,color=COLORS,width=.65)
        if metric=='test_accuracy_mean':ax.errorbar(range(4),values,yerr=[agg[v]['test_accuracy_sd']*100 for v in VARIANTS],fmt='none',color='black',capsize=4)
        for i,val in enumerate(values):ax.text(i,val+max(values)*.025,f'{val:.2f}',ha='center',fontsize=8)
        ax.set_xticks(range(4),['Base','DW','SE','DW+SE']);ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.17)
    fig.suptitle('Architecture trade-offs | accuracy error bars: SD, n=3');save(fig,'tradeoffs')
    fig,ax=plt.subplots(figsize=(7,3.4),layout='constrained')
    for v,color,marker in zip(VARIANTS,COLORS,['o','s','^','D']):
        a=agg[v];ax.errorbar(a['latency_ms_median_mean'],a['test_accuracy_mean']*100,
                    xerr=a['latency_ms_median_sd'],yerr=a['test_accuracy_sd']*100,
                    fmt=marker,color=color,capsize=4,label=NAMES[v],markersize=8)
    ax.set(xlabel='GPU batch-1 latency (ms): mean of run medians',ylabel='Test accuracy (%)')
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'latency_accuracy')
    chosen=max(runs,key=lambda r:r['best_validation_accuracy'])
    selected_path=results/f'{chosen["variant"]}_seed{chosen["seed"]}'
    pred=np.load(selected_path/'test_predictions.npz');y=pred['labels'];probs=pred['probabilities'];yp=probs.argmax(1)
    cm=np.bincount(y*10+yp,minlength=100).reshape(10,10)
    fig,ax=plt.subplots(figsize=(7.4,5.4),layout='constrained')
    im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1000)
    for i in range(10):
        for j in range(10):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>500 else '#193a4a')
    ax.set_xticks(range(10),CLASSES,rotation=45,ha='right');ax.set_yticks(range(10),CLASSES)
    ax.set(xlabel='Predicted class',ylabel='True class',title=f'{NAMES[chosen["variant"]]}, seed {chosen["seed"]} | selected by validation')
    fig.colorbar(im,ax=ax,label='Samples');save(fig,'confusion_matrix')
    thresholds=[];confidence=probs.max(1)
    for t in [0,.5,.6,.7,.8,.85,.9,.95,.99,1.]:
        accepted=confidence>=t;n=int(accepted.sum());correct=int((yp[accepted]==y[accepted]).sum())
        thresholds.append({'threshold':t,'accepted':n,'rejected':len(y)-n,'coverage':n/len(y),
                           'accepted_correct':correct,'accepted_accuracy':correct/n if n else None})
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    axes[0].plot([r['threshold'] for r in thresholds],[r['coverage']*100 for r in thresholds],'o-',color=COLORS[0])
    axes[1].plot([r['threshold'] for r in thresholds],[np.nan if r['accepted_accuracy'] is None else r['accepted_accuracy']*100 for r in thresholds],'s-',color=COLORS[1])
    for ax,label in zip(axes,['Coverage (%)','Accuracy among accepted (%)']):ax.set(xlabel='Confidence threshold',ylabel=label,ylim=(0,102));ax.grid(alpha=.2)
    fig.suptitle('Test-set descriptive threshold sweep (not used for tuning)');save(fig,'threshold_sweep')
    images,_=load_part(False)
    wrong=np.flatnonzero(yp!=y)[:8]
    fig,axes=plt.subplots(2,4,figsize=(9,4.6),layout='constrained')
    for ax,i in zip(axes.flat,wrong):
        ax.imshow(images[i],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.axis('off')
        ax.set_title(f'#{i}: true {CLASSES[y[i]]}\npred {CLASSES[yp[i]]} ({confidence[i]*100:.1f}%)',fontsize=9)
    fig.suptitle('First 8 misclassified test samples in original order');save(fig,'error_examples')
    fig,ax=plt.subplots(figsize=(10,2.2),layout='constrained');ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
    boxes=[('Input\n1 × 28 × 28',.1),('3×3 stem\n16 × 28 × 28',1.7),('3 blocks\n16 × 28 × 28',3.3),('3 blocks\n32 × 14 × 14',4.9),('3 blocks\n64 × 7 × 7',6.5),('GAP + FC\n10 classes',8.1)]
    for text,x in boxes:
        ax.add_patch(plt.Rectangle((x,.65),1.45,.8,facecolor='#e5efe9',edgecolor='#267568',lw=1.3))
        ax.text(x+.725,1.05,text,ha='center',va='center',fontsize=10)
        if x<8:ax.annotate('',xy=(x+1.6,1.05),xytext=(x+1.45,1.05),arrowprops={'arrowstyle':'->','color':'#267568'})
    ax.text(5,.25,'Residual branch: Conv → BN → ReLU → Conv → BN → [SE] → Add shortcut → ReLU',ha='center',fontsize=10)
    save(fig,'architecture')
    derived={'aggregates':agg,'selected_run':chosen,'threshold_sweep':thresholds,'runtime':runtime,
             'parameter_reduction_dw_percent':(1-agg['dw']['parameters']/agg['baseline']['parameters'])*100,
             'mac_reduction_dw_percent':(1-agg['dw']['macs']/agg['baseline']['macs'])*100,
             'se_delta_accuracy_pp':(agg['se']['test_accuracy_mean']-agg['baseline']['test_accuracy_mean'])*100,
             'dw_se_delta_accuracy_pp':(agg['dw_se']['test_accuracy_mean']-agg['dw']['test_accuracy_mean'])*100}
    (results/'analysis.json').write_text(json.dumps(derived,indent=2),encoding='utf-8')
    with (results/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['variant']+list(agg['baseline']));w.writeheader()
        for v,a in agg.items():w.writerow({'variant':v,**a})
    (figures/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(derived,indent=2))


if __name__=='__main__':main()
