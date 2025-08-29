# Pyton-Generator-PHPAiModel-Transformer — app.py
# Flask-based server for training a character-level Transformer model with GPU/CPU support.
#
# Developed by: Artur Strazewicz — concept, architecture, Python Transformer training, UI.
# Year: 2025. License: MIT.
#
# Links:
#   GitHub:      https://github.com/iStark/Pyton-Generator-PHPAiModel-Transformer
#   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
#   TruthSocial: https://truthsocial.com/@strazewicz
#   X (Twitter): https://x.com/strazewicz
import os, time, json, threading
from dataclasses import asdict
from flask import Flask, request, Response, render_template_string, send_from_directory
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Transformer, Config

app = Flask(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT, 'Datasets')
MODELS_DIR = os.path.join(ROOT, 'Models')
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- HTML шаблон ----------
HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Transformer Trainer</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:960px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:inline-block;margin:6px 12px 6px 0}
input,select{padding:8px;border:1px solid #ddd;border-radius:8px}
button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;cursor:pointer}
.bar{height:8px;background:#ececff;border-radius:999px;overflow:hidden}
.bar i{display:block;height:100%;width:0%;background:#4f46e5}
.grid{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;margin-top:8px}
.kv{background:#fafafe;border:1px solid #eee;border-radius:8px;padding:8px}
.kv b{display:block;font-size:12px;color:#666;margin-bottom:4px}
.kv span{font-variant-numeric:tabular-nums}
pre{white-space:pre-wrap}
.small{color:#666;font-size:12px;margin-top:6px}
footer{
    margin: 0 auto;
    margin-bottom: 2rem;
    font-size: .9em;font-size:.9em;color:var(--mut)}
        footer div a{color:inherit}
        .links a{margin-right:.75rem}
</style></head>
<body><div class="wrap">
  <h1>Char-level Transformer Trainer (CUDA)</h1>
  <div class="card">
    <form id="f" onsubmit="start();return false;">
      <label>Dataset
        <select id="ds">
          {% for f in datasets %}<option value="{{f}}">{{f}}</option>{% endfor %}
        </select>
      </label>
      <label>D_model <input id="dmodel" type="number" value="256" min="64" max="1024"></label>
      <label>Heads <input id="heads" type="number" value="4" min="1" max="16"></label>
      <label>Layers <input id="layers" type="number" value="4" min="1" max="48"></label>
      <label>FF <input id="dff" type="number" value="1024" min="128" max="8192"></label>
      <label>Seq <input id="seq" type="number" value="256" min="32" max="2048"></label>
      <label>Batch <input id="bs" type="number" value="64" min="1" max="512"></label>
      <label>Epochs <input id="ep" type="number" value="2" min="1" max="100"></label>
      <label>LR <input id="lr" step="0.0001" type="number" value="0.001"></label>
      <label>Out name <input id="out" type="text" value="char-transformer"></label>
      <label><input id="amp" type="checkbox" checked> Use AMP (Tensor Cores)</label>
      <button>Start training</button>
    </form>
  </div>

  <div class="card">
    <h3>Прогресс</h3>
    <div class="bar"><i id="p"></i></div>
    <div class="grid">
      <div class="kv"><b>%</b><span id="s_pct">0.00%</span></div>
      <div class="kv"><b>Прошло</b><span id="s_elapsed">0.00m</span></div>
      <div class="kv"><b>ETA</b><span id="s_eta">—</span></div>
      <div class="kv"><b>Эпоха</b><span id="s_epoch">0 / 0</span></div>
      <div class="kv"><b>Итерации</b><span id="s_step">0 / 0</span></div>
      <div class="kv"><b>Loss</b><span id="s_loss">—</span></div>
    </div>
    <div class="small" id="s_device">device: —</div>
    <pre id="log"></pre>
  </div>
    <footer>

        <div><strong>Pyton-Generator-PHPAiModel-Transformer</strong> — pre-LN, GELU, MHA, tied output head.</div>
        <div>© <span id="year">2025</span>. Developed by <strong>Artur Strazewicz</strong> — concept, architecture, Pyton-Generator-PHPAiModel-Transformer, UI,  <strong>MIT license</strong>.</div>
        <div class="links">
            <a href="https://github.com/iStark/Pyton-Generator-PHPAiModel-Transformer" target="_blank" rel="noopener">GitHub</a>
            <a href="https://www.linkedin.com/in/arthur-stark/" target="_blank" rel="noopener">LinkedIn</a>
            <a href="https://truthsocial.com/@strazewicz" target="_blank" rel="noopener">Truth Social</a>
            <a href="https://x.com/strazewicz" target="_blank" rel="noopener">X (Twitter)</a>
        </div>
    </footer>
</div>
<script>
(function(){
  const $ = (id)=>document.getElementById(id);

  function val(id){ return $(id).value; }

  function parseProgress(line){
    // Пример:
    // Progress:  12.50% | epoch 1/2 | step 10/80 | loss 3.9123 | elapsed 0.30m | ETA 2.10m
    const pct = (line.match(/Progress:\\s+([\\d.]+)%/) || [,''])[1];
    const epoch = (line.match(/epoch\\s+(\\d+)\\/(\\d+)/) || [,,])[1];
    const epochT = (line.match(/epoch\\s+(\\d+)\\/(\\d+)/) || [,,,''])[2];
    const step  = (line.match(/step\\s+(\\d+)\\/(\\d+)/) || [,,])[1];
    const stepT = (line.match(/step\\s+(\\d+)\\/(\\d+)/) || [,,,''])[2];
    const loss  = (line.match(/loss\\s+([\\d.]+)/) || [,''])[1];
    const elapsed = (line.match(/elapsed\\s+([\\d.]+m)/) || [,''])[1];
    const eta     = (line.match(/ETA\\s+([\\d.]+m)/) || [,''])[1];

    if (pct) { $('p').style.width = pct + '%'; $('s_pct').textContent = pct + '%'; }
    if (epoch && epochT) $('s_epoch').textContent = epoch + ' / ' + epochT;
    if (step && stepT) $('s_step').textContent = step + ' / ' + stepT;
    if (loss) $('s_loss').textContent = loss;
    if (elapsed) $('s_elapsed').textContent = elapsed;
    if (eta) $('s_eta').textContent = eta || '—';
  }

  window.start = function(){
    const sel = $('ds');
    if (!sel || !sel.value || sel.value.indexOf('.txt') === -1) {
      alert('Выбери dataset (.txt) в списке.');
      return;
    }
    const params = new URLSearchParams({
      ds: sel.value,
      dmodel: val('dmodel'),
      heads:  val('heads'),
      layers: val('layers'),
      dff:    val('dff'),
      seq:    val('seq'),
      bs:     val('bs'),
      ep:     val('ep'),
      lr:     val('lr'),
      out:    val('out'),
      amp:    $('amp').checked ? '1':'0'
    });

    const es  = new EventSource('/train?' + params.toString());
    const log = $('log');

    es.onmessage = function(e){
      const line = e.data || '';
      if (line.startsWith('PCT:')){
        $('p').style.width = line.slice(4) + '%';
        $('s_pct').textContent = line.slice(4) + '%';
      } else {
        // Стартовый баннер: "Training started (device=..., AMP=...)"
        if (line.startsWith('Training started')){
          const mdev = line.match(/device=([^,\\)]+)/);
          const mamp = line.match(/AMP=(True|False)/);
          $('s_device').textContent = 'device: ' + (mdev ? mdev[1] : '—') + ' | AMP: ' + (mamp ? mamp[1] : '—');
        }
        // Парсим строку прогресса для обновления статуса
        if (line.startsWith('Progress:')) parseProgress(line);
        log.textContent += line + "\\n";
        log.scrollTop = log.scrollHeight;
      }
      if (line === 'DONE') es.close();
    };

    es.onerror = function(){
      es.close();
      alert('Поток прерван. Смотри Network/Console и логи в терминале Flask.');
    };
  };
})();
</script>
</body></html>"""

# ---------- Dataset ----------
class CharDataset(Dataset):
    def __init__(self, text, seq):
        chars = sorted(list(set(text)))
        if '<unk>' not in chars:
            chars = ['<unk>'] + chars
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab = chars
        self.data = torch.tensor([self.stoi.get(ch, 0) for ch in text], dtype=torch.long)
        self.seq = seq
    def __len__(self):
        return max(0, len(self.data) - self.seq - 1)
    def __getitem__(self, i):
        chunk = self.data[i:i+self.seq+1]
        return chunk[:-1], chunk[1:]

# ---------- Export JSON ----------
@torch.no_grad()
def export_json(model: Transformer, vocab, out_path):
    cfg = model.cfg
    obj = {
        'config': asdict(cfg),
        'vocab': vocab,
        'tok_emb': model.tok_emb.weight.detach().cpu().tolist(),
        'pos_emb': model.pos_emb.weight.detach().cpu().tolist(),
        'ln_f_w': model.ln_f.weight.detach().cpu().tolist(),
        'ln_f_b': model.ln_f.bias.detach().cpu().tolist(),
        'layers': []
    }
    for blk in model.blocks:
        attn = blk.attn
        qkv_w = attn.in_proj_weight.detach().cpu()
        qkv_b = attn.in_proj_bias.detach().cpu()
        proj_w = attn.out_proj.weight.detach().cpu()
        proj_b = attn.out_proj.bias.detach().cpu()
        item = {
            'ln1_w': blk.ln1.weight.detach().cpu().tolist(),
            'ln1_b': blk.ln1.bias.detach().cpu().tolist(),
            'attn_qkv_w': qkv_w.tolist(),
            'attn_qkv_b': qkv_b.tolist(),
            'attn_proj_w': proj_w.tolist(),
            'attn_proj_b': proj_b.tolist(),
            'ln2_w': blk.ln2.weight.detach().cpu().tolist(),
            'ln2_b': blk.ln2.bias.detach().cpu().tolist(),
            'fc1_w': blk.fc1.weight.detach().cpu().tolist(),
            'fc1_b': blk.fc1.bias.detach().cpu().tolist(),
            'fc2_w': blk.fc2.weight.detach().cpu().tolist(),
            'fc2_b': blk.fc2.bias.detach().cpu().tolist(),
        }
        obj['layers'].append(item)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)

# ---------- SSE Trainer ----------
class Trainer:
    def __init__(self):
        self.listeners = []
        self.lock = threading.Lock()
    def send(self, msg):
        with self.lock:
            for q in list(self.listeners):
                q.append(msg)
    def stream(self):
        q = []
        with self.lock:
            self.listeners.append(q)
        try:
            last = 0
            while True:
                while last < len(q):
                    data = q[last]; last += 1
                    yield f"data: {data}\n\n"
                time.sleep(0.05)
        finally:
            with self.lock:
                if q in self.listeners:
                    self.listeners.remove(q)

trainer = Trainer()

def device_setup(use_amp):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    return dev

# ---------- Routes ----------
@app.route('/')
def home():
    files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.txt')]
    if not files:
        files = ['(put .txt file into Datasets/)']
    return render_template_string(HTML, datasets=files)

@app.route('/train')
def train():
    ds_name = request.args.get('ds')
    if not ds_name or not ds_name.endswith('.txt'):
        def err():
            yield "data: ERROR: no dataset selected\n\n"
            yield "data: DONE\n\n"
        return Response(err(), mimetype='text/event-stream')

    d_model = int(request.args.get('dmodel', 256))
    heads   = int(request.args.get('heads', 4))
    layers  = int(request.args.get('layers', 4))
    dff     = int(request.args.get('dff', 1024))
    seq     = int(request.args.get('seq', 256))
    bs      = int(request.args.get('bs', 64))
    ep      = int(request.args.get('ep', 2))
    lr      = float(request.args.get('lr', 1e-3))
    outn    = request.args.get('out', 'char-transformer')
    use_amp = request.args.get('amp', '1') == '1'

    def run():
        try:
            path = os.path.join(DATASETS_DIR, ds_name)
            if not os.path.isfile(path):
                trainer.send(f"ERROR: dataset not found: {path}")
                trainer.send("DONE"); return

            text = open(path, 'r', encoding='utf-8', errors='ignore').read()
            if not text.strip():
                trainer.send("ERROR: dataset is empty"); trainer.send("DONE"); return

            dataset = CharDataset(text, seq)
            if len(dataset) == 0:
                trainer.send("ERROR: dataset too small for seq"); trainer.send("DONE"); return

            cfg = Config(vocab_size=len(dataset.vocab), d_model=d_model,
                         n_head=heads, n_layer=layers, d_ff=dff, max_seq=seq)
            dev = device_setup(use_amp)
            model = Transformer(cfg).to(dev)
            optim = torch.optim.AdamW(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            # split train/val
            N = len(dataset)
            n_val = max(1, N // 100)
            train_ds = torch.utils.data.Subset(dataset, range(0, N - n_val))
            val_ds   = torch.utils.data.Subset(dataset, range(N - n_val, N))
            train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
            val_dl   = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)

            total_steps = ep * len(train_dl)
            step = 0; start = time.time()
            trainer.send(f"Training started (device={dev}, AMP={use_amp})")

            for epoch in range(1, ep+1):
                model.train()
                for xb, yb in train_dl:
                    xb = xb.to(dev); yb = yb.to(dev)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(xb)
                        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), yb.reshape(-1))
                    optim.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optim); scaler.update()

                    step += 1
                    pct = 100.0 * step / max(1, total_steps)
                    elapsed = time.time() - start
                    eta = (elapsed/step)*(total_steps-step) if step>0 else 0.0
                    trainer.send(f"PCT:{pct:.2f}")
                    trainer.send(f"Progress: {pct:6.2f}% | epoch {epoch}/{ep} | step {step}/{total_steps} | loss {loss.item():.4f} | elapsed {elapsed/60:.2f}m | ETA {eta/60:.2f}m")

                # quick val
                model.eval(); vloss=0.0; vcount=0
                with torch.no_grad():
                    for xb,yb in val_dl:
                        xb=xb.to(dev); yb=yb.to(dev)
                        logits=model(xb)
                        loss=loss_fn(logits.reshape(-1,cfg.vocab_size), yb.reshape(-1))
                        vloss+=loss.item(); vcount+=1
                trainer.send(f"Validation loss: {vloss/max(1,vcount):.4f}")

            # save
            torch_path = os.path.join(MODELS_DIR, outn+'.pt')
            json_path  = os.path.join(MODELS_DIR, outn+'.json')
            torch.save({'cfg':asdict(cfg), 'state_dict':model.state_dict()}, torch_path)
            export_json(model, dataset.vocab, json_path)
            trainer.send(f"Saved: {torch_path}\\nSaved: {json_path}")
            trainer.send("DONE")

        except Exception as e:
            trainer.send("ERROR: " + repr(e))
            trainer.send("DONE"); return

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return Response(trainer.stream(), mimetype='text/event-stream')

@app.route('/Models/<path:fn>')
def dl_models(fn):
    return send_from_directory(MODELS_DIR, fn, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
