# Pyton-Generator-PHPAiModel-Transformer — interface.py
# Flask-based web chat interface for interacting with trained Transformer models (.json weights).
#
# Developed by: Artur Strazewicz — concept, architecture, Python Transformer inference, UI.
# Year: 2025. License: MIT.
#
# Links:
#   GitHub:      https://github.com/iStark/Pyton-Generator-PHPAiModel-Transformer
#   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
#   TruthSocial: https://truthsocial.com/@strazewicz
#   X (Twitter): https://x.com/strazewicz

import os, json, time
from dataclasses import dataclass
from typing import Dict, Any
from flask import Flask, request, Response, render_template_string
import torch
import torch.nn as nn

# ---------- Модель (та же, что в model.py) ----------
@dataclass
class Config:
    vocab_size: int
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 4
    d_ff: int = 1024
    max_seq: int = 256
    tie_weights: bool = True

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x

class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.readout.weight = self.tok_emb.weight
        self.register_buffer('causal_mask', torch.triu(torch.ones(cfg.max_seq, cfg.max_seq), 1).bool(), persistent=False)
    def forward(self, idx):
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        mask = self.causal_mask[:t, :t]
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.ln_f(x)
        return self.readout(x)

# ---------- Flask ----------
app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, 'Models')
os.makedirs(MODELS_DIR, exist_ok=True)

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Transformer Chat</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:960px;margin:0 auto;padding:16px;display:flex;flex-direction:column;height:100vh}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:inline-block;margin:6px 12px 6px 0}
input,select{padding:8px;border:1px solid #ddd;border-radius:8px}
button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;cursor:pointer}
.row{display:flex;gap:8px;align-items:center}
.grow{flex:1}
.chat{flex:1;overflow-y:auto;background:#fff;border:1px solid #eee;border-radius:12px;padding:12px}
.msg{background:#fafafe;border:1px solid #eee;border-radius:10px;padding:10px 12px;margin:8px 0}
.msg.ai{border-left:4px solid #4f46e5}
.msg.user{border-left:4px solid #9ca3af}
.small{color:#666;font-size:12px;margin-left:4px}
:root {--mut:#556;}
        footer{
    margin: 0 auto;
    margin-bottom: 2rem;
    font-size: .9em;font-size:.9em;color:var(--mut)}
        footer div a{color:inherit}
        .links a{margin-right:.75rem}
</style></head>
<body><div class="wrap">
  <div class="card">
    <div class="row">
      <label>Model
        <select id="model">
          {% for m in models %}<option value="{{m}}">{{m}}</option>{% endfor %}
        </select>
      </label>
      <label>Max new <input id="max_new" type="number" value="200" min="1" max="2048"></label>
      <label>Temp <input id="temp" type="number" step="0.1" value="0.9" min="0.1" max="2.0"></label>
      <label>Top-k <input id="topk" type="number" value="40" min="0" max="500"></label>
      <span class="small" id="dev">device: —</span>
    </div>
  </div>

  <div class="chat" id="chat"></div>

  <div class="card">
    <div class="row">
      <textarea id="ta" class="grow" placeholder="Введите сообщение..." rows="4"></textarea>
      <button id="send">Send</button>
    </div>
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
  const chat = $('chat');

  function add(role, text){
    const el = document.createElement('div');
    el.className = 'msg ' + (role==='ai'?'ai':'user');
    el.textContent = text;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
  }

  async function startGen(prompt){
    const params = new URLSearchParams({
      model: $('model').value,
      max_new: $('max_new').value,
      temperature: $('temp').value,
      top_k: $('topk').value,
      prompt: prompt
    });
    const es = new EventSource('/gen?'+params.toString());
    let aiEl = add('ai', '');
    es.onmessage = (e)=>{
      const s = e.data || '';
      if (s.startsWith('DEV:')){
        $('dev').textContent = 'device: ' + s.slice(4);
        return;
      }
      if (s.startsWith('TOK:')){
        aiEl.textContent += s.slice(4);
        chat.scrollTop = chat.scrollHeight;
        return;
      }
      if (s === 'DONE'){ es.close(); return; }
      if (s.startsWith('ERR:')){ aiEl.textContent += '\\n['+s.slice(4)+']'; es.close(); }
    };
    es.onerror = ()=>{ es.close(); };
  }

  $('send').onclick = ()=>{
    const t = $('ta').value.trim();
    if(!t) return;
    add('user', t);
    $('ta').value = '';
    startGen(t);
  };
  $('ta').addEventListener('keydown', (e)=>{
    if(e.key==='Enter' && (e.ctrlKey||e.metaKey)){ $('send').click(); }
  });
})();
</script>

</body></html>"""

# ---------- Загрузка .json модели ----------
def load_json_model(path: str, device: torch.device):
    j = json.load(open(path, 'r', encoding='utf-8'))
    cfgj = j['config']
    cfg = Config(
        vocab_size=int(cfgj['vocab_size']),
        d_model=int(cfgj['d_model']),
        n_head=int(cfgj['n_head']),
        n_layer=int(cfgj['n_layer']),
        d_ff=int(cfgj['d_ff']),
        max_seq=int(cfgj['max_seq']),
        tie_weights=bool(cfgj.get('tie_weights', True))
    )
    model = Transformer(cfg).to(device)
    # Embeddings & ln_f
    with torch.no_grad():
        model.tok_emb.weight.copy_(torch.tensor(j['tok_emb'], dtype=torch.float32, device=device))
        model.pos_emb.weight.copy_(torch.tensor(j['pos_emb'], dtype=torch.float32, device=device))
        if 'ln_f_w' in j:
            model.ln_f.weight.copy_(torch.tensor(j['ln_f_w'], dtype=torch.float32, device=device))
            model.ln_f.bias.copy_(torch.tensor(j['ln_f_b'], dtype=torch.float32, device=device))
        # Layers
        for blk, lj in zip(model.blocks, j['layers']):
            blk.ln1.weight.copy_(torch.tensor(lj['ln1_w'], dtype=torch.float32, device=device))
            blk.ln1.bias.copy_(torch.tensor(lj['ln1_b'], dtype=torch.float32, device=device))
            blk.attn.in_proj_weight.copy_(torch.tensor(lj['attn_qkv_w'], dtype=torch.float32, device=device))
            blk.attn.in_proj_bias.copy_(torch.tensor(lj['attn_qkv_b'], dtype=torch.float32, device=device))
            blk.attn.out_proj.weight.copy_(torch.tensor(lj['attn_proj_w'], dtype=torch.float32, device=device))
            blk.attn.out_proj.bias.copy_(torch.tensor(lj['attn_proj_b'], dtype=torch.float32, device=device))
            blk.ln2.weight.copy_(torch.tensor(lj['ln2_w'], dtype=torch.float32, device=device))
            blk.ln2.bias.copy_(torch.tensor(lj['ln2_b'], dtype=torch.float32, device=device))
            blk.fc1.weight.copy_(torch.tensor(lj['fc1_w'], dtype=torch.float32, device=device))
            blk.fc1.bias.copy_(torch.tensor(lj['fc1_b'], dtype=torch.float32, device=device))
            blk.fc2.weight.copy_(torch.tensor(lj['fc2_w'], dtype=torch.float32, device=device))
            blk.fc2.bias.copy_(torch.tensor(lj['fc2_b'], dtype=torch.float32, device=device))
    vocab = j['vocab']
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for i,ch in enumerate(vocab)}
    return model.eval(), cfg, vocab, stoi, itos

# ---------- Кэш моделей ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}

def get_model(model_name: str):
    path = os.path.join(MODELS_DIR, model_name)
    if path not in MODEL_CACHE:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"model not found: {path}")
        m, cfg, vocab, stoi, itos = load_json_model(path, DEVICE)
        MODEL_CACHE[path] = {'model': m, 'cfg': cfg, 'vocab': vocab, 'stoi': stoi, 'itos': itos}
    return MODEL_CACHE[path]

# ---------- Токенизация и генерация ----------
def str_to_ids(s: str, stoi: Dict[str,int], unk_id: int) -> torch.Tensor:
    ids = []
    for ch in s:
        ids.append(stoi.get(ch, unk_id))
    return torch.tensor(ids, dtype=torch.long, device=DEVICE)

def ids_to_str(ids, itos):
    return ''.join(itos.get(int(i), '') for i in ids)

@torch.no_grad()
def generate_stream(model: Transformer, cfg: Config, ids: torch.Tensor, max_new=200, temperature=0.9, top_k=40):
    # вернуть контекст не длиннее max_seq
    def tail(x, tmax):
        return x[-tmax:] if x.numel() > tmax else x
    unk = 0  # по экспорту <unk> обычно индекс 0
    ids = ids.clone()
    start = time.time()
    yield f"DEV:{('CUDA' if torch.cuda.is_available() else 'CPU')}"
    for _ in range(max_new):
        ctx = tail(ids, cfg.max_seq).unsqueeze(0)  # [1, T]
        logits = model(ctx)[:, -1, :]  # [1, V]
        logits = logits / max(1e-6, float(temperature))
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
            th = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < th, torch.full_like(logits, -1e10), logits)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
        ids = torch.cat([ids, next_id[0,0].unsqueeze(0)], dim=0)
        yield int(next_id[0,0].item())
    yield ('DONE', time.time() - start)

# ---------- Маршруты ----------
@app.route('/')
def home():
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.json')]
    if not models:
        models = ['(put .json from trainer into Models/)']
    return render_template_string(HTML, models=models)

@app.route('/gen')
def gen():
    model_name = request.args.get('model', '')
    prompt = request.args.get('prompt', '')
    max_new = int(request.args.get('max_new', 200))
    temperature = float(request.args.get('temperature', 0.9))
    top_k = int(request.args.get('top_k', 40))

    def stream():
        try:
            if not model_name.endswith('.json'):
                yield "data: ERR: select a .json model from Models/\n\n"; return
            pack = get_model(model_name)
            model, cfg, vocab, stoi, itos = pack['model'], pack['cfg'], pack['vocab'], pack['stoi'], pack['itos']
            unk_id = stoi.get('<unk>', 0)
            ids = str_to_ids(prompt, stoi, unk_id)
            # Отдать устройство
            yield f"data: DEV:{('CUDA' if torch.cuda.is_available() else 'CPU')}\n\n"
            # Генерация по токену
            for out in generate_stream(model, cfg, ids, max_new=max_new, temperature=temperature, top_k=top_k):
                if isinstance(out, tuple) and out[0]=='DONE':
                    yield "data: DONE\n\n"; return
                if isinstance(out, int):
                    ch = vocab[out] if 0 <= out < len(vocab) else ''
                    yield f"data: TOK:{ch}\n\n"
        except Exception as e:
            yield f"data: ERR: {repr(e)}\n\n"

    return Response(stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
