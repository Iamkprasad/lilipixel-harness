const $ = (id) => document.getElementById(id);
const statusPill = $("statusPill");
const modelInfo = $("modelInfo");
const downloadStatus = $("downloadStatus");
const genStatus = $("genStatus");
const results = $("results");

async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  try { return JSON.parse(text); } catch { return { success: false, error: text }; }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c]));
}

async function refreshStatus() {
  const r = await api("/api/status");
  if (!r.success) { statusPill.textContent = "Error"; return; }
  const d = r.data;
  const dl = d.download || {};
  const parts = [];
  parts.push(d.installed ? "Model downloaded" : "Model not downloaded");
  parts.push(d.loaded ? `Loaded (${d.device}, ${d.dtype || ""})` : "Not loaded");
  if (d.memory && d.memory.free_mb) parts.push(`RAM free: ${d.memory.free_mb} MB`);
  modelInfo.textContent = parts.join(" · ");
  statusPill.textContent = d.loaded ? "Ready" : (d.installed ? "Downloaded" : "Needs download");
  statusPill.className = "status-pill " + (d.loaded ? "ok" : "");
  if (dl.status && dl.status !== "idle") {
    downloadStatus.textContent = dl.message || dl.status + (dl.error ? ": " + dl.error : "");
  } else {
    downloadStatus.textContent = "";
  }
}

$("downloadBtn").onclick = async () => {
  $("downloadBtn").disabled = true;
  downloadStatus.textContent = "Starting download...";
  const r = await api("/api/download", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
  if (!r.success) downloadStatus.textContent = "Error: " + r.error;
  $("downloadBtn").disabled = false;
  pollDownload();
};

async function pollDownload() {
  for (;;) {
    const r = await api("/api/download/status");
    if (!r.success) break;
    const d = r.data || {};
    if (d.status === "downloading") {
      downloadStatus.textContent = d.message || "Downloading...";
      await new Promise(res => setTimeout(res, 1500));
      continue;
    }
    downloadStatus.textContent = d.message || d.status || "";
    if (d.error) downloadStatus.textContent = "Error: " + d.error;
    break;
  }
  refreshStatus();
}

$("loadBtn").onclick = async () => {
  $("loadBtn").disabled = true;
  statusPill.textContent = "Loading...";
  statusPill.className = "status-pill busy";
  const r = await api("/api/load", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
  $("loadBtn").disabled = false;
  if (!r.success) { genStatus.textContent = "Load error: " + r.error; }
  refreshStatus();
};

$("unloadBtn").onclick = async () => {
  await api("/api/unload", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
  refreshStatus();
};

$("size").onchange = () => {
  const sz = parseInt($("size").value, 10);
  $("steps").value = sz === 64 ? 50 : 30;
};

function renderImages(images, params) {
  results.classList.remove("empty");
  results.innerHTML = "";
  for (const img of images) {
    const card = document.createElement("div");
    card.className = "card";
    const src = img.path;
    card.innerHTML = `
      <img class="pixel" src="${esc(src)}" alt="generated">
      <div class="meta">
        seed ${esc(img.seed)} · ${esc(img.ms)}ms · ${esc(params.width)}×${esc(params.height)}<br>
        <a href="${esc(src)}" download="lilipixel-${esc(img.seed)}.png">Download PNG</a>
      </div>`;
    results.appendChild(card);
  }
}

$("genBtn").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { genStatus.textContent = "Prompt is required."; return; }
  $("genBtn").disabled = true;
  genStatus.textContent = "Starting...";
  statusPill.textContent = "Generating...";
  statusPill.className = "status-pill busy";
  results.classList.remove("empty");
  results.innerHTML = '<div class="muted">Generating...</div>';

  const body = {
    prompt,
    size: parseInt($("size").value, 10),
    steps: parseInt($("steps").value, 10),
    cfg: parseFloat($("cfg").value),
    batch: parseInt($("batch").value, 10),
  };
  const seedVal = $("seed").value.trim();
  if (seedVal !== "") body.seed = parseInt(seedVal, 10);

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let doneData = null;
    let err = null;
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        let evt;
        try { evt = JSON.parse(line); } catch { continue; }
        if (evt.t === "progress") {
          genStatus.textContent = evt.message || (evt.stage || "Working...");
        } else if (evt.t === "done") {
          doneData = evt;
        } else if (evt.t === "error") {
          err = evt.e || "Generation failed";
        }
      }
    }
    if (err) {
      genStatus.textContent = "Error: " + err;
      results.innerHTML = '<div class="err">' + esc(err) + '</div>';
    } else if (doneData) {
      genStatus.textContent = `Done — ${doneData.images.length} image(s).`;
      renderImages(doneData.images, doneData.params);
    } else {
      genStatus.textContent = "No result returned.";
    }
  } catch (e) {
    genStatus.textContent = "Request failed: " + e;
    results.innerHTML = '<div class="err">' + esc(String(e)) + '</div>';
  } finally {
    $("genBtn").disabled = false;
    refreshStatus();
  }
};

refreshStatus();
pollDownload();
setInterval(refreshStatus, 10000);
