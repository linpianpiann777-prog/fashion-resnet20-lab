"""Run in PyCharm or `python app.py`; local-only visual classifier."""
import argparse
import base64
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image, ImageOps
import torch
from models import ResNet20, VARIANTS, complexity
from data_utils import ROOT, CLASSES, CLASSES_ZH, load_part, tensor_images


class Predictor:
    def __init__(self):
        torch.set_num_threads(4)
        self.images,self.labels=load_part(False)
        self.models={};self.info={};self.lock=threading.Lock()
        for variant in VARIANTS:
            candidates=[]
            for p in (ROOT/'results').glob(f'{variant}_seed*/metrics.json'):
                metrics=json.loads(p.read_text())
                if metrics['variant']==variant:candidates.append((metrics,p.parent/'best.pt'))
            if not candidates:continue
            metrics,path=max(candidates,key=lambda item:item[0]['best_validation_accuracy'])
            model=ResNet20(variant)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
            model.eval();self.models[variant]=model
            self.info[variant]={'seed':metrics['seed'],**complexity(model)}
        if not self.models:raise RuntimeError('没有已训练的模型。请先运行 train.py，或下载仓库附带的 results 权重。')

    def predict(self,payload):
        variant=payload.get('variant','baseline')
        if variant not in self.models:raise ValueError('模型不可用')
        threshold=float(payload.get('threshold',.8))
        if not 0<=threshold<=1:raise ValueError('阈值必须在 0 到 1 之间')
        if payload.get('image'):
            content=base64.b64decode(payload['image'].split(',')[-1],validate=True)
            if len(content)>5*1024*1024:raise ValueError('图片不能超过5MB')
            img=Image.open(io.BytesIO(content)).convert('L')
            if payload.get('invert'):img=ImageOps.invert(img)
            img=ImageOps.pad(img,(28,28),method=Image.Resampling.BILINEAR,color=0)
            array=np.array(img);index=None;truth=None
        else:
            index=int(payload.get('index',0))
            if not 0<=index<len(self.images):raise ValueError('样本编号超出范围')
            array=self.images[index];truth=int(self.labels[index]);img=Image.fromarray(array)
        with self.lock,torch.inference_mode():
            probs=self.models[variant](tensor_images(array[None],torch.device('cpu'))).softmax(1)[0].numpy()
        pred=int(probs.argmax());confidence=float(probs[pred]);buf=io.BytesIO();img.save(buf,format='PNG')
        return {'index':index,'true_label':None if truth is None else CLASSES_ZH[truth],
                'predicted_label':CLASSES_ZH[pred],'confidence':confidence,'accepted':confidence>=threshold,
                'probabilities':probs.tolist(),'image':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
                'variant':variant,'model_info':self.info[variant]}


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,data,kind='application/json; charset=utf-8'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(code);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            parsed=urlparse(self.path)
            if parsed.path=='/':return self.send(200,(ROOT/'ui.html').read_bytes(),'text/html; charset=utf-8')
            if parsed.path=='/api/info':return self.send(200,{'classes':CLASSES_ZH,'models':predictor.info,'count':len(predictor.images)})
            if parsed.path=='/api/sample':
                query=parse_qs(parsed.query); label=int(query.get('class',['-1'])[0])
                ids=np.arange(len(predictor.images)) if label==-1 else np.flatnonzero(predictor.labels==label)
                if not len(ids):return self.send(400,{'error':'无此类别'})
                return self.send(200,{'index':int(np.random.default_rng().choice(ids))})
            return self.send(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/predict':return self.send(404,{'error':'Not found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=7*1024*1024:raise ValueError('请求过大或为空')
                self.send(200,predictor.predict(json.loads(self.rfile.read(length))))
            except (ValueError,KeyError,OSError) as exc:self.send(400,{'error':str(exc)})
        def log_message(self,fmt,*args):pass
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    predictor=Predictor();server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(predictor))
    url=f'http://127.0.0.1:{args.port}'
    print(f'Fashion Lab 已启动: {url}  | Ctrl+C 停止',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
