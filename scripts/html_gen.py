import base64, pandas as pd, os
from pathlib import Path

#df = pd.read_csv("./models/kfold_cnn_rgbs_5l/false_positives_with_saliency.csv")
df = pd.read_csv("./models/kfold_cnn_rgbs_5l/false_negatives_with_saliency.csv")

html = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body { font-family: sans-serif; background:#fafafa; margin:40px; }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; }
.block { border:1px solid #ccc; background:#fff; box-shadow:0 2px 5px rgba(0,0,0,.1);}
.top { display:flex; border-bottom:1px solid #ccc;}
.rank{flex:1; background:#2e3192;color:white;text-align:center;padding:8px;}
.rank span{display:block;font-size:1.3em;font-weight:700;}
.prob{flex:1;text-align:center;padding:8px;color:#2e3192;font-weight:600;}
.prob span{display:block;font-size:1.3em;color:black;}
img{width:100%;height:220px;object-fit:contain;background:#fafafa;}
</style></head><body><h1>Malaria Classifier Error Explorer - False Negatives</h1>
 <p style="max-width:900px;font-size:1.1em;line-height:1.5;color:#333;">
    This gallery displays examples of false negative classifications 
    produced by our trained deep learning model. The values "0" and "1" correspond to "uninfected" and "infected," respectively.  
    Each block shows the model’s predicted probability for false negative in rank order of greatest to least error.  For example, a predicted probability of 0.00 indicates the model made the most confident false negative error.  
    Alongside each block is the original cell image and the corresponding model explanation (contrastive saliency map), highlighting the regions that most influenced 
    the model’s decision.  Correcting these potentially incorrectly assigned images is crucial for improving the model's accuracy and reliability in malaria detection. Tthe number of false negative 
    shown below (260) is a small fraction of the total 27,558 images in the original data set. 
  </p>

<div class="grid">
"""

for i, row in df.iterrows():
    img_path = Path(row["saliency_map_path"])
    img64 = ""
    if img_path.exists():
        with open(img_path, "rb") as f: img64 = base64.b64encode(f.read()).decode()
    html += f"""
    <div class="block">
      <div class="top">
        <div class="rank">Rank<span>#{i+1} / {len(df)}</span></div>
        <div class="prob">Predicted Probability<span>{row['predicted_probability']:.2f}</span></div>
      </div>
      <img src="data:image/png;base64,{img64}">
    </div>"""

html += "</div></body></html>"

Path("saliency_gallery_fn.html").write_text(html)
print("Wrote saliency_gallery_fn.html")