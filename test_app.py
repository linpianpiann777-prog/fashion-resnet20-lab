"""End-to-end HTTP checks against the real predictor, including checkpoint parity."""
import base64
import io
import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from app import Predictor,make_handler
from data_utils import ROOT
from models import VARIANTS

def main():
    predictor=Predictor()
    assert set(predictor.models)==set(VARIANTS)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(predictor))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    checks=[]
    def get(path):
        with urllib.request.urlopen(url+path) as r:return json.load(r)
    def post(payload):
        req=urllib.request.Request(url+'/api/predict',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as r:return json.load(r)
    try:
        info=get('/api/info');assert info['count']==10000
        for v in VARIANTS:
            r=post({'variant':v,'index':0,'threshold':0})
            assert r['accepted'] and abs(sum(r['probabilities'])-1)<1e-6
            seed=predictor.info[v]['seed']
            stored=np.load(ROOT/f'results/{v}_seed{seed}/test_predictions.npz')['probabilities'][0]
            assert np.allclose(r['probabilities'],stored,atol=3e-5),v
            checks.append(v+': CPU live prediction matches saved GPU test prediction')
        r0=post({'index':0,'threshold':0});r1=post({'index':1,'threshold':1})
        assert r0['image']!=r1['image'] and r1['index']==1
        assert r1['accepted']==(r1['confidence']>=1)
        # Choose an interior confidence so the threshold change must affect acceptance.
        r=post({'index':1,'threshold':0})
        assert r['confidence']<1
        assert post({'index':1,'threshold':1})['accepted'] is False
        checks.append('threshold endpoints and sample switching')
        for c in range(10):
            idx=get('/api/sample?class='+str(c))['index'];assert predictor.labels[idx]==c
        checks.append('random sample class filters for all 10 classes')
        buf=io.BytesIO();Image.fromarray(predictor.images[0]).save(buf,format='PNG')
        r=post({'image':base64.b64encode(buf.getvalue()).decode(),'threshold':.8})
        assert r['index'] is None and r['true_label'] is None
        assert np.allclose(r['probabilities'],r0['probabilities'],atol=1e-7)
        post({'image':base64.b64encode(buf.getvalue()).decode(),'invert':True})
        jpeg=io.BytesIO();Image.fromarray(predictor.images[1]).save(jpeg,format='JPEG')
        assert post({'image':base64.b64encode(jpeg.getvalue()).decode()})['index'] is None
        checks.append('PNG/JPEG upload, preprocessing parity, invert option')
        for invalid in [{'index':-1},{'index':10000},{'threshold':1.1},{'variant':'unknown'},{'image':'broken'}]:
            try:post(invalid);raise AssertionError('Invalid input unexpectedly accepted')
            except urllib.error.HTTPError as e:assert e.code==400
        checks.append('invalid inputs return HTTP 400')
    finally:server.shutdown();server.server_close()
    output={'passed':True,'checks':checks}
    (ROOT/'results/app_checks.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
