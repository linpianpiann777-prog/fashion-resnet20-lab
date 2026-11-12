import pathlib, urllib.request, concurrent.futures, hashlib, json
from pypdf import PdfReader
P=pathlib.Path('papers');P.mkdir(exist_ok=True)
items=[('01_ResNet_CVPR2016','Deep Residual Learning for Image Recognition','https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf'),('02_MobileNets_2017','MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications','https://arxiv.org/pdf/1704.04861'),('03_SE_CVPR2018','Squeeze-and-Excitation Networks','https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf'),('04_FashionMNIST_2017','Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms','https://arxiv.org/pdf/1708.07747'),('05_Calibration_ICML2017','On Calibration of Modern Neural Networks','https://proceedings.mlr.press/v70/guo17a/guo17a.pdf')]
items=[item for item in items if item[0].startswith(('01_', '03_'))]
def download(item):
 name,title,url=item; path=P/(name+'.pdf')
 if not path.exists():
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(req,timeout=90) as response:data=response.read(50*1024*1024)
  assert data.startswith(b'%PDF');path.write_bytes(data)
 r=PdfReader(path);assert len(r.pages)>0
 return dict(file=path.name,title=title,url=url,pages=len(r.pages),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),first_page_excerpt=r.pages[0].extract_text()[:250])
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(download,items))
(P/'manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=True,indent=2))
