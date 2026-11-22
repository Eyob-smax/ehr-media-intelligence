// EHR Media Intelligence: search UI (vanilla JS, no build step).
// All server text is escaped before rendering; highlighting runs on escaped text only.
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const els = {
    form: $("#search-form"), q: $("#q"), type: $("#type"), from: $("#from"), to: $("#to"), topk: $("#topk"),
    clear: $("#clear"), results: $("#results"), status: $("#status"), stats: $("#stats"), filterError: $("#filter-error"),
    drawerRoot: $("#drawer-root"), drawer: $("#drawer"), backdrop: $("#backdrop"), drawerClose: $("#drawer-close"),
    drawerTitle: $("#drawer-title"), drawerSub: $("#drawer-sub"), drawerBody: $("#drawer-body"),
  };

  const EXAMPLES = ["blood clot in the leg", "high potassium", "pneumonia on chest x-ray", "kidney stone",
                    "abnormal troponin", "heart failure with fluid overload"];
  const TYPE_STYLE = {
    DiagnosticReport: ["Diagnostic report", "bg-sky-100 text-sky-800 ring-sky-200"],
    DocumentReference: ["Clinical document", "bg-violet-100 text-violet-800 ring-violet-200"],
    Patient: ["AI summary", "bg-amber-100 text-amber-800 ring-amber-200"],
  };

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const STOP = new Set("the and with for from that this was were are has have into over under show find any all".split(" "));
  const terms = (q) => [...new Set(q.toLowerCase().match(/[a-z0-9][a-z0-9-]{2,}/g) || [])].filter((t) => !STOP.has(t));
  function highlight(text, q) {
    let html = esc(text);
    for (const t of terms(q)) {
      html = html.replace(new RegExp(`(${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"), "<mark>$1</mark>");
    }
    return html;
  }
  const badge = (type) => {
    const [label, cls] = TYPE_STYLE[type] || [type, "bg-slate-100 text-slate-700 ring-slate-200"];
    return `<span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${cls}">${esc(label)}</span>`;
  };
  const fmtDate = (d) => d ? new Date(d + "T00:00:00").toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "Undated";
  const announce = (msg) => { els.status.textContent = msg; };

  // ---------------------------------------------------------------- state
  let controller = null;
  let lastQuery = "";
  let lastFocus = null;

  // ---------------------------------------------------------------- search
  async function runSearch() {
    const q = els.q.value.trim();
    els.filterError.classList.add("hidden");
    if (els.from.value && els.to.value && els.from.value > els.to.value) {
      els.filterError.textContent = "“From date” must be on or before “To date”.";
      els.filterError.classList.remove("hidden");
      return;
    }
    if (!q) { renderWelcome(); return; }

    const params = new URLSearchParams();
    if (els.type.value) params.append("resource_type", els.type.value);
    if (els.from.value) params.append("date_from", els.from.value);
    if (els.to.value) params.append("date_to", els.to.value);

    controller?.abort();
    controller = new AbortController();
    renderLoading();
    try {
      const res = await fetch(`/search?${params}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: Number(els.topk.value) }), signal: controller.signal,
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const data = await res.json();
      lastQuery = q;
      renderResults(data);
    } catch (err) {
      if (err.name === "AbortError") return;
      renderError(err.message);
    }
  }
  const searchSoon = debounce(runSearch, 300);

  // ---------------------------------------------------------------- render: list
  function renderWelcome() {
    els.results.innerHTML = `
      <div class="text-center py-12 px-4 bg-white rounded-xl border border-dashed border-slate-300">
        <h2 class="text-base font-medium text-slate-700">Search across every patient's notes, labs, imaging and AI summaries</h2>
        <p class="mt-1 text-sm text-slate-500">Plain language works. Try one of these:</p>
        <div class="mt-4 flex flex-wrap justify-center gap-2">
          ${EXAMPLES.map((e) => `<button type="button" data-example="${esc(e)}" class="rounded-full border border-slate-300 bg-white px-3 py-1 text-sm text-slate-700 hover:border-brand-500 hover:text-brand-700 focus:outline-none focus:ring-2 focus:ring-brand-500">${esc(e)}</button>`).join("")}
        </div>
      </div>`;
    announce("");
  }

  function renderLoading() {
    els.results.setAttribute("aria-busy", "true");
    els.results.innerHTML = `<ul class="space-y-3" aria-hidden="true">${Array.from({ length: 3 }, () => `
      <li class="bg-white rounded-xl border border-slate-200 p-4">
        <div class="skeleton h-4 w-1/3 rounded"></div><div class="skeleton mt-3 h-3 w-full rounded"></div>
        <div class="skeleton mt-2 h-3 w-5/6 rounded"></div></li>`).join("")}</ul>`;
    announce("Searching…");
  }

  function renderError(message) {
    els.results.removeAttribute("aria-busy");
    els.results.innerHTML = `
      <div class="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800" role="alert">
        <p class="font-medium">Search failed</p><p class="mt-1">${esc(message)}</p>
        <button type="button" id="retry" class="mt-3 rounded-lg bg-white px-3 py-1.5 text-sm border border-red-300 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-400">Try again</button>
      </div>`;
    $("#retry").addEventListener("click", runSearch);
    announce("Search failed");
  }

  function renderResults(data) {
    els.results.removeAttribute("aria-busy");
    const { results, took_ms, filters } = data;
    if (!results.length) {
      const why = filters?.mrn_not_found ? `No patient has ${esc(filters.mrn)}.` : "No records matched. Try broader wording or remove filters.";
      els.results.innerHTML = `
        <div class="text-center py-12 px-4 bg-white rounded-xl border border-dashed border-slate-300">
          <h2 class="text-base font-medium text-slate-700">No results for “${esc(data.query)}”</h2>
          <p class="mt-1 text-sm text-slate-500">${why}</p>
        </div>`;
      announce("No results");
      return;
    }
    const mrnNote = filters?.mrn ? ` · filtered to ${esc(filters.mrn)}` : "";
    els.results.innerHTML = `
      <p class="mb-2 text-xs text-slate-500">${results.length} result${results.length > 1 ? "s" : ""} in ${Math.round(took_ms)} ms${mrnNote}</p>
      <ol class="space-y-3" role="list" aria-label="Ranked results">
        ${results.map(card).join("")}
      </ol>`;
    announce(`${results.length} results in ${Math.round(took_ms)} milliseconds`);
  }

  function card(r) {
    const pct = Math.round(r.score * 100);
    return `
      <li>
        <article tabindex="0" data-card data-patient="${esc(r.patient_id)}"
                 aria-label="${esc(`Rank ${r.rank}: ${r.title} for ${r.patient_name}, ${fmtDate(r.date)}, relevance ${pct} percent`)}"
                 class="group bg-white rounded-xl border border-slate-200 p-4 shadow-sm hover:border-brand-500 hover:shadow focus:outline-none focus:ring-2 focus:ring-brand-500 cursor-pointer transition">
          <div class="flex flex-wrap items-start justify-between gap-2">
            <div class="min-w-0">
              <h3 class="font-semibold text-slate-900 truncate">${esc(r.patient_name)} <span class="font-normal text-slate-500 text-sm">· ${esc(r.mrn)}</span></h3>
              <p class="text-sm text-slate-600">${esc(r.title)} · <time datetime="${esc(r.date || "")}">${esc(fmtDate(r.date))}</time></p>
            </div>
            <div class="flex items-center gap-3 shrink-0">
              ${badge(r.resource_type)}
              <div class="w-24" title="Semantic ${Math.round(r.semantic_score * 100)}% · keyword ${Math.round(r.keyword_score * 100)}%">
                <div class="flex justify-between text-[11px] text-slate-500"><span>Relevance</span><span>${pct}%</span></div>
                <div class="mt-0.5 h-1.5 rounded-full bg-slate-100" aria-hidden="true"><div class="h-1.5 rounded-full bg-brand-500" style="width:${Math.min(100, pct)}%"></div></div>
              </div>
            </div>
          </div>
          <p class="mt-2 text-sm text-slate-700 leading-relaxed">${highlight(r.snippet, lastQuery)}</p>
          ${r.summary_snippet ? `
          <p class="mt-2 text-sm text-slate-600 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
            <span class="font-medium text-amber-800">AI summary:</span> ${esc(r.summary_snippet)}
          </p>` : ""}
        </article>
      </li>`;
  }

  // Keyboard navigation between result cards.
  els.results.addEventListener("keydown", (e) => {
    const cards = [...els.results.querySelectorAll("[data-card]")];
    const i = cards.indexOf(document.activeElement);
    if (i === -1) return;
    if (e.key === "ArrowDown") { e.preventDefault(); cards[Math.min(i + 1, cards.length - 1)].focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); i === 0 ? els.q.focus() : cards[i - 1].focus(); }
    else if (e.key === "Home") { e.preventDefault(); cards[0].focus(); }
    else if (e.key === "End") { e.preventDefault(); cards[cards.length - 1].focus(); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPatient(cards[i].dataset.patient); }
  });
  els.results.addEventListener("click", (e) => {
    const ex = e.target.closest("[data-example]");
    if (ex) { els.q.value = ex.dataset.example; runSearch(); return; }
    const c = e.target.closest("[data-card]");
    if (c) openPatient(c.dataset.patient);
  });
  els.q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      const first = els.results.querySelector("[data-card]");
      if (first) { e.preventDefault(); first.focus(); }
    }
  });

  // ---------------------------------------------------------------- drawer
  async function openPatient(pid) {
    lastFocus = document.activeElement;
    els.drawerRoot.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    els.drawerTitle.textContent = "Loading patient…";
    els.drawerSub.textContent = "";
    els.drawerBody.innerHTML = `<div class="space-y-3" aria-hidden="true"><div class="skeleton h-24 rounded-lg"></div><div class="skeleton h-40 rounded-lg"></div></div>`;
    els.drawerClose.focus();
    try {
      const res = await fetch(`/patients/${encodeURIComponent(pid)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      renderPatient(await res.json());
    } catch (err) {
      els.drawerTitle.textContent = "Could not load patient";
      els.drawerBody.innerHTML = `<p class="text-sm text-red-700" role="alert">${esc(err.message)}</p>`;
    }
  }

  function closeDrawer() {
    els.drawerRoot.classList.add("hidden");
    document.body.style.overflow = "";
    lastFocus?.focus();
  }

  function cite(ids, citations) {
    return (ids || []).map((id) => {
      const c = citations[id];
      const label = c ? `${c.title} (${c.date})` : id;
      return `<sup class="ml-0.5"><a href="#res-${esc(c?.resource_id || "")}" data-jump="${esc(c?.resource_id || "")}" class="text-brand-700 hover:underline" title="${esc(label)}" aria-label="Source: ${esc(label)}">[${esc(id)}]</a></sup>`;
    }).join("");
  }

  function renderPatient(p) {
    const s = p.summary;
    els.drawerTitle.textContent = p.name;
    const bits = [p.mrn, p.gender, p.birth_date ? `DOB ${p.birth_date}` : null].filter(Boolean);
    els.drawerSub.textContent = bits.join(" · ") + (p.alternate_mrns.length ? ` · also known as ${p.alternate_mrns.join(", ")}` : "");
    const conf = s ? { high: "bg-emerald-100 text-emerald-800", medium: "bg-amber-100 text-amber-800", low: "bg-red-100 text-red-800" }[s.confidence.level] : "";
    const C = p.citations || {};

    const summaryHtml = !s ? `<p class="text-sm text-slate-500">No summary available.</p>` : `
      <div class="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900" role="note">
        <strong>AI-generated.</strong> ${esc(s.disclaimer)}
      </div>
      <dl class="mt-4 space-y-4 text-sm">
        <div><dt class="font-medium text-slate-500 text-xs uppercase tracking-wide">Chief concern</dt>
             <dd class="mt-1 text-slate-900">${esc(s.chief_concern)}</dd></div>
        <div><dt class="font-medium text-slate-500 text-xs uppercase tracking-wide">Key diagnoses</dt>
             <dd class="mt-1"><ul class="list-disc pl-5 space-y-0.5">${s.key_diagnoses.map((d) => `<li>${esc(d.text)}${cite(d.source_ids, C)}</li>`).join("") || "<li class='list-none -ml-5 text-slate-500'>None documented</li>"}</ul></dd></div>
        <div><dt class="font-medium text-slate-500 text-xs uppercase tracking-wide">Recent imaging &amp; labs</dt>
             <dd class="mt-1"><ul class="space-y-1">${s.recent_media.map((m) => `<li><span class="text-slate-500">${esc(m.date)}</span> · <span class="font-medium">${esc(m.title)}</span>: ${esc(m.finding)}${cite([m.source_id], C)}</li>`).join("") || "<li class='text-slate-500'>None</li>"}</ul></dd></div>
        <div><dt class="font-medium text-red-700 text-xs uppercase tracking-wide">Flagged anomalies</dt>
             <dd class="mt-1"><ul class="space-y-1">${s.flagged_anomalies.map((a) => `<li class="flex gap-2"><span aria-hidden="true" class="text-red-600">⚑</span><span>${esc(a.text)}${cite(a.source_ids, C)}</span></li>`).join("") || "<li class='text-slate-500'>None flagged</li>"}</ul></dd></div>
      </dl>
      <div class="mt-4 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span class="rounded-full px-2 py-0.5 font-medium ${conf}">Confidence: ${esc(s.confidence.level)} (${Math.round(s.confidence.score * 100)}%)</span>
        <span>${esc(s.word_count)} words</span><span>·</span>
        <span>${esc(s.provenance.generator === "claude" ? `Claude (${s.provenance.model})` : "Offline extractive summarizer")}</span>
      </div>
      <details class="mt-2 text-xs text-slate-500"><summary class="cursor-pointer">How confidence was computed</summary>
        <ul class="mt-1 list-disc pl-5">${s.confidence.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}${(s.quality_issues || []).map((q) => `<li class="text-amber-700">${esc(q)}</li>`).join("")}</ul>
        ${s.provenance.fallback_reason ? `<p class="mt-1">Fallback reason: ${esc(s.provenance.fallback_reason)}</p>` : ""}
      </details>`;

    const resources = p.resources.map((r) => `
      <li id="res-${esc(r.resource_id)}" class="rounded-lg border border-slate-200 p-3 scroll-mt-4">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <p class="font-medium text-sm text-slate-900">${esc(r.title)}</p>${badge(r.resource_type)}
        </div>
        <p class="text-xs text-slate-500">${esc(r.category || "")} · ${esc(fmtDate(r.date))}</p>
        ${r.conclusion ? `<p class="mt-1 text-sm text-slate-700"><span class="font-medium">Conclusion:</span> ${esc(r.conclusion)}</p>` : ""}
        <details class="mt-1" data-resource="${esc(r.resource_id)}">
          <summary class="cursor-pointer text-xs text-brand-700 hover:underline">View text and FHIR JSON</summary>
          <p class="mt-2 whitespace-pre-line text-sm text-slate-700">${esc(r.preview)}${r.preview.length >= 280 ? "…" : ""}</p>
          <pre class="mt-2 max-h-64 overflow-auto rounded bg-slate-900 p-2 text-[11px] text-slate-100" data-json>Loading…</pre>
        </details>
      </li>`).join("");

    els.drawerBody.innerHTML = `
      <section aria-labelledby="sum-h"><h3 id="sum-h" class="text-sm font-semibold text-slate-900 mb-2">Clinical summary</h3>${summaryHtml}</section>
      <section class="mt-6" aria-labelledby="res-h">
        <div class="flex items-center justify-between mb-2">
          <h3 id="res-h" class="text-sm font-semibold text-slate-900">Linked FHIR resources (${p.resources.length})</h3>
          <a href="/patients/${encodeURIComponent(p.patient_id)}/bundle" target="_blank" rel="noopener" class="text-xs text-brand-700 hover:underline">Open FHIR Bundle ↗</a>
        </div>
        ${p.bundle_valid ? "" : `<p class="mb-2 text-xs text-red-700">This bundle has validation errors.</p>`}
        <ul class="space-y-2">${resources}</ul>
      </section>`;
  }

  els.drawerBody.addEventListener("toggle", async (e) => {
    const d = e.target;
    if (!(d instanceof HTMLDetailsElement) || !d.open || !d.dataset.resource) return;
    const pre = d.querySelector("[data-json]");
    if (pre.dataset.loaded) return;
    try {
      const res = await fetch(`/resources/${encodeURIComponent(d.dataset.resource)}`);
      const json = await res.json();
      if (json.content?.[0]?.attachment?.data) json.content[0].attachment.data = "<base64 omitted>";
      if (json.presentedForm?.[0]?.data) json.presentedForm[0].data = "<base64 omitted>";
      pre.textContent = JSON.stringify(json, null, 2);
      pre.dataset.loaded = "1";
    } catch { pre.textContent = "Could not load resource."; }
  }, true);

  els.drawerBody.addEventListener("click", (e) => {
    const a = e.target.closest("[data-jump]");
    if (!a) return;
    e.preventDefault();
    const target = document.getElementById(`res-${a.dataset.jump}`);
    if (target) { target.scrollIntoView({ behavior: "smooth", block: "start" }); target.classList.add("ring-2", "ring-brand-500"); setTimeout(() => target.classList.remove("ring-2", "ring-brand-500"), 1500); }
  });

  // Focus trap + Escape for the dialog.
  els.drawerRoot.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); closeDrawer(); return; }
    if (e.key !== "Tab") return;
    const focusable = [...els.drawer.querySelectorAll('a[href], button, summary, [tabindex]:not([tabindex="-1"])')].filter((x) => x.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
  els.drawerClose.addEventListener("click", closeDrawer);
  els.backdrop.addEventListener("click", closeDrawer);

  // ---------------------------------------------------------------- wiring
  els.form.addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  els.q.addEventListener("input", searchSoon);
  [els.type, els.from, els.to, els.topk].forEach((el) => el.addEventListener("change", () => els.q.value.trim() && runSearch()));
  els.clear.addEventListener("click", () => {
    els.type.value = ""; els.from.value = ""; els.to.value = ""; els.topk.value = "5";
    els.filterError.classList.add("hidden");
    els.q.value.trim() ? runSearch() : renderWelcome();
  });

  fetch("/health").then((r) => r.json()).then((h) => {
    els.stats.textContent = `${h.patients} patients · ${h.indexed_chunks} indexed passages · summaries: ${h.summarizer}`;
  }).catch(() => {});

  renderWelcome();
  els.q.focus();
})();
