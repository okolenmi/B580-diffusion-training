/* ---------------------------------------------------------------------------
   Dataset Manager — handles data generation, inspection, and curation
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    const state = {
        activeDataset: localStorage.getItem("comfy_active_dataset"),
        activeTab: localStorage.getItem("comfy_active_tab") || "library",
        datasets: [],
        pendingTrajectories: [],
        archivedTrajectories: [],
        selectedTrajs: new Set(),
        activeTasks: [],
        pollInterval: null,
        activeDetailTrajId: null,
        activeInspectionTrajId: null
    };

    const STORAGE_KEY = "comfy_generator_settings";

    function escHtml(s) {
        if (!s) return "";
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    function initTabs() {
        const tabBtns = document.querySelectorAll(".tab-btn");
        const tabContents = document.querySelectorAll(".tab-content");

        // Restore active tab visually
        tabBtns.forEach(btn => {
            if (btn.dataset.tab === state.activeTab) btn.classList.add("active");
            else btn.classList.remove("active");
        });
        tabContents.forEach(c => {
            if (c.id === `tab-${state.activeTab}`) c.classList.add("active");
            else c.classList.remove("active");
        });

        tabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const tab = btn.dataset.tab;
                state.activeTab = tab;
                localStorage.setItem("comfy_active_tab", tab);
                
                tabBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");

                tabContents.forEach(c => {
                    c.classList.remove("active");
                    if (c.id === `tab-${tab}`) c.classList.add("active");
                });

                if (tab === "library") loadDatasets();
                if (tab === "inspection") loadPending();
                if (tab === "explorer") loadArchived();
                if (tab === "generator") restoreSettings();
            });
        });
    }

    function initGeneratorUI() {
        const typeSelect = document.getElementById("gen-source-type");
        const posModeSelect = document.getElementById("gen-pos-mode");
        const negModeSelect = document.getElementById("gen-neg-mode");

        if (typeSelect) {
            typeSelect.onchange = () => {
                const isTeacher = typeSelect.value === "teacher";
                document.getElementById("gen-teacher-options").style.display = isTeacher ? "grid" : "none";
                document.getElementById("gen-real-options").style.display = isTeacher ? "none" : "block";
                saveSettings();
            };
        }

        const setupToggle = (select, listId, dictId) => {
            if (!select) return;
            select.onchange = () => {
                const isList = select.value === "list";
                document.getElementById(listId).style.display = isList ? "block" : "none";
                document.getElementById(dictId).style.display = isList ? "none" : "block";
                saveSettings();
            };
        };

        setupToggle(posModeSelect, "pos-list-area", "pos-dict-area");
        setupToggle(negModeSelect, "neg-list-area", "neg-dict-area");

        // Attach auto-save to all generator inputs
        document.querySelectorAll("input[id^='gen-'], select[id^='gen-'], textarea[id^='gen-']").forEach(el => {
            if (el.type === "checkbox") {
                el.addEventListener("click", saveSettings);
            } else {
                el.addEventListener("input", saveSettings);
            }
        });
    }

    function saveSettings() {
        const settings = {};
        document.querySelectorAll("input[id^='gen-'], select[id^='gen-'], textarea[id^='gen-']").forEach(el => {
            if (el.type === "checkbox") {
                settings[el.id] = el.checked;
            } else {
                settings[el.id] = el.value;
            }
        });
        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }

    function restoreSettings() {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        try {
            const s = JSON.parse(raw);
            Object.entries(s).forEach(([key, val]) => {
                // Support both new keys (IDs) and old keys (legacy)
                let el = document.getElementById(key);
                if (!el) {
                    // Try legacy mapping
                    const legacyId = key.startsWith("gen-") ? key : `gen-${key.replace(/_/g, "-")}`;
                    el = document.getElementById(legacyId);

                    // Specific legacy overrides for mismatched names
                    if (!el && key === "type") el = document.getElementById("gen-source-type");
                    if (!el && key === "n_conditions") el = document.getElementById("gen-unique-conds");
                    if (!el && key === "n_samples") el = document.getElementById("gen-samples-per-cond");
                    if (!el && key === "neg_prompt") el = document.getElementById("gen-negative-prompt");
                }

                if (el && val !== undefined) {
                    if (el.type === "checkbox") el.checked = val;
                    else el.value = val;
                }
            });

            // Trigger visual updates
            const typeSelect = document.getElementById("gen-source-type");
            if (typeSelect) {
                const isTeacher = typeSelect.value === "teacher";
                document.getElementById("gen-teacher-options").style.display = isTeacher ? "grid" : "none";
                document.getElementById("gen-real-options").style.display = isTeacher ? "none" : "block";
            }

            const ev = new Event("change");
            ["gen-pos-mode", "gen-neg-mode"].forEach(id => {
                document.getElementById(id)?.dispatchEvent(ev);
            });
        } catch (e) {
            console.error("Failed to restore settings:", e);
        }
    }

    function loadDatasets() {
        return fetch("/api/datasets")
            .then(r => r.json())
            .then(data => {
                state.datasets = data;
                // Validate activeDataset against loaded list (clear stale localStorage)
                if (state.activeDataset && !data.some(d => d.name === state.activeDataset)) {
                    state.activeDataset = null;
                    localStorage.removeItem("comfy_active_dataset");
                }
                renderLibrary();
                renderSidebar();
            });
    }

    function renderSidebar() {
        const list = document.getElementById("dataset-list-sidebar");
        if (!list) return;
        list.innerHTML = "";
        state.datasets.forEach(ds => {
            const row = document.createElement("div");
            row.style = "display:flex; align-items:center; gap:0.25rem; margin-bottom:0.25rem;";
            
            const btn = document.createElement("button");
            btn.className = `btn btn-secondary ${state.activeDataset === ds.name ? "active" : ""}`;
            btn.style = "flex:1; justify-content:flex-start; padding:0.6rem 0.85rem; font-size:0.85rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";
            btn.innerHTML = `<span>📁</span> ${escHtml(ds.name)}`;
            btn.onclick = () => {
                state.activeDataset = ds.name;
                localStorage.setItem("comfy_active_dataset", ds.name);
                renderSidebar();
                renderLibrary();
                loadPending();
                loadArchived();
                startPolling();
            };
            
            const delBtn = document.createElement("button");
            delBtn.className = "btn btn-kill btn-small";
            delBtn.style = "padding:0.4rem 0.6rem; font-size:0.7rem;";
            delBtn.innerHTML = "×";
            delBtn.onclick = e => {
                e.stopPropagation();
                window.deleteDataset(ds.name);
            };

            row.appendChild(btn);
            row.appendChild(delBtn);
            list.appendChild(row);
        });
    }

    function renderLibrary() {
        const grid = document.getElementById("dataset-grid");
        if (!grid) return;
        grid.innerHTML = "";

        state.datasets.forEach(ds => {
            const card = document.createElement("div");
            card.className = `dataset-card ${state.activeDataset === ds.name ? "active" : ""}`;
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="font-size: 2rem;">📦</div>
                    <button class="btn btn-kill btn-small btn-delete-dataset" style="padding: 0.3rem 0.6rem; margin-top: -0.5rem; margin-right: -0.5rem;">
                        ×
                    </button>
                </div>
                <h3 style="margin:0">${escHtml(ds.name)}</h3>
                <p class="text-dim" style="font-size:0.9rem; flex:1;">${escHtml(ds.description) || "No description provided."}</p>
                <div style="margin-top:auto; font-size:0.75rem; color:var(--accent); font-weight:600;">
                    CREATED: ${new Date(ds.created_at * 1000).toLocaleDateString()}
                </div>
            `;
            card.querySelector(".btn-delete-dataset").onclick = e => {
                e.stopPropagation();
                window.deleteDataset(ds.name);
            };
            card.onclick = () => {
                state.activeDataset = ds.name;
                localStorage.setItem("comfy_active_dataset", ds.name);
                renderLibrary();
                renderSidebar();
                loadPending();
                loadArchived();
                startPolling();
            };
            grid.appendChild(card);
        });
    }

    window.deleteDataset = function(name) {
        if (!confirm(`Are you sure you want to PERMANENTLY delete the dataset '${name}'? This will remove all shards and previews.`)) {
            return;
        }

        fetch(`/api/datasets/${name}/delete`, { method: "POST" })
            .then(r => r.json())
            .then(data => {
                if (data.ok) {
                    if (state.activeDataset === name) {
                        state.activeDataset = null;
                        if (state.pollInterval) clearInterval(state.pollInterval);
                    }
                    loadDatasets();
                } else {
                    alert(`Delete failed: ${data.detail}`);
                }
            });
    };

    function loadPending() {
        if (!state.activeDataset) {
            state.pendingTrajectories = [];
            renderInspection();
            return;
        }
        fetch(`/api/datasets/${state.activeDataset}/trajectories/pending`)
            .then(r => r.json())
            .then(data => {
                state.pendingTrajectories = data;
                renderInspection();
            });
    }

    function loadArchived() {
        if (!state.activeDataset) {
            state.archivedTrajectories = [];
            renderExplorer();
            return;
        }
        fetch(`/api/datasets/${state.activeDataset}/trajectories/archived`)
            .then(r => r.json())
            .then(data => {
                state.archivedTrajectories = data;
                renderExplorer();
            });
    }

    function renderExplorer() {
        const gallery = document.getElementById("explorer-gallery");
        const countEl = document.getElementById("explorer-count");
        if (!gallery) return;

        state.activeDetailTrajId = null;
        
        countEl.textContent = `${state.archivedTrajectories.length} ITEMS ARCHIVED`;

        if (!state.activeDataset) {
            gallery.innerHTML = '<p class="placeholder p-2">Please select a dataset from the library.</p>';
            showDetail(null, "explorer");
            return;
        }

        if (state.archivedTrajectories.length === 0) {
            gallery.innerHTML = '<p class="placeholder p-2">No items in archive yet. Go to Inspection to curate data.</p>';
            showDetail(null, "explorer");
            return;
        }

        const fragment = document.createDocumentFragment();
        state.archivedTrajectories.forEach(traj => {
            const item = document.createElement("div");
            item.className = "preview-item";
            const previewSrc = traj.preview_path ? `/datasets/${state.activeDataset}/${traj.preview_path}` : "";
            const meta = traj.metadata ? JSON.parse(traj.metadata) : {};
            const cfgLabel = meta.cfg ? ` | CFG: ${meta.cfg.toFixed(1)}` : "";
            const isBad = meta.type === "bad";

            item.dataset.trajId = traj.id;
            item.innerHTML = `
                ${previewSrc ? `<img src="${previewSrc}" loading="lazy" onerror="this.src='/static/broken.png'; this.className='broken-img'">` : '<div class="no-preview">NO PREVIEW</div>'}
                <div class="preview-meta">ID: ${traj.id}${cfgLabel}</div>
                <button class="btn btn-small traj-type-btn ${isBad ? 'btn-danger' : 'btn-secondary'}" 
                        style="position:absolute; bottom:4px; right:4px; font-size:0.65rem; padding:0.15rem 0.4rem; z-index:3;"
                        onclick="event.stopPropagation(); window.toggleType(${traj.id})">
                    ${isBad ? 'BAD' : 'good'}
                </button>
            `;
            item.onclick = () => showDetail(traj.id, "explorer");
            fragment.appendChild(item);
        });

        gallery.innerHTML = "";
        gallery.appendChild(fragment);

        // Clear detail panel
        showDetail(null, "explorer");
    }

    function showDetail(trajId, tabType) {
        const panelId = tabType === "explorer" ? "explorer-detail" : "inspection-detail";
        const panel = document.getElementById(panelId);
        if (!panel) return;

        // Update selection highlight for current gallery
        const galleryId = tabType === "explorer" ? "explorer-gallery" : "inspection-gallery";
        document.querySelectorAll(`#${galleryId} .preview-item`).forEach(el => {
            el.classList.toggle("selected", el.dataset.trajId == trajId);
        });
        
        if (tabType === "explorer") state.activeDetailTrajId = trajId;
        else state.activeInspectionTrajId = trajId;

        if (!trajId) {
            panel.innerHTML = '<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:var(--text-dim); font-size:0.85rem;">Click a trajectory to inspect or edit details</div>';
            return;
        }

        const allTrajs = [...state.pendingTrajectories, ...state.archivedTrajectories];
        const traj = allTrajs.find(t => t.id === trajId);
        if (!traj) {
            panel.innerHTML = '<div style="padding:1rem; color:var(--red);">Trajectory not found</div>';
            return;
        }

        const meta = traj.metadata ? JSON.parse(traj.metadata) : {};
        const isBad = meta.type === "bad";

        panel.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <span style="font-weight:700; font-size:1.1rem;">Edit Trajectory #${traj.id}</span>
                <button class="btn btn-small ${isBad ? 'btn-danger' : 'btn-secondary'}" 
                      style="padding:0.25rem 0.6rem; font-size:0.75rem;"
                      onclick="event.stopPropagation(); window.toggleType(${traj.id})">
                    ${isBad ? 'BAD' : 'good'}
                </button>
            </div>

            <div class="option-row mb-1">
                <label class="option-label">Positive Prompt</label>
                <textarea id="edit-${tabType}-prompt" class="cfg-input" style="height:140px; width:100%; resize:vertical;">${escHtml(traj.prompt || "")}</textarea>
            </div>

            <div class="option-row mb-1">
                <label class="option-label">Negative Prompt</label>
                <textarea id="edit-${tabType}-neg-prompt" class="cfg-input" style="height:80px; width:100%; resize:vertical;">${escHtml(meta.neg || "")}</textarea>
            </div>

            <div class="option-row mb-1-5">
                <label class="option-label">Guidance Scale (CFG)</label>
                <div style="display:flex; gap:0.5rem; align-items:center;">
                    <input type="number" id="edit-${tabType}-cfg" class="cfg-input" step="0.1" value="${meta.cfg || 1.0}" style="width:100px;">
                    <span class="text-dim" style="font-size:0.75rem;">Effective only for CFG-aware training.</span>
                </div>
            </div>

            <div style="margin-top:auto; padding-top:1rem; border-top:1px solid var(--surface-alt);">
                <button class="btn btn-start btn-block" onclick="window.saveTrajectory(${traj.id}, '${tabType}')">
                    SAVE CHANGES
                </button>
            </div>

            <div class="mt-1-5 text-dim" style="font-size:0.75rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                    <span>Samples: ${traj.sample_count}</span>
                    <span>Seed: ${traj.seed}</span>
                </div>
                <div>Size: ${traj.width || '?' }x${traj.height || '?'}</div>
            </div>
        `;
    }

    window.saveTrajectory = function(trajId, tabType) {
        const prompt = document.getElementById(`edit-${tabType}-prompt`).value;
        const neg_prompt = document.getElementById(`edit-${tabType}-neg-prompt`).value;
        const cfg = parseFloat(document.getElementById(`edit-${tabType}-cfg`).value);

        if (isNaN(cfg)) {
            alert("CFG must be a number");
            return;
        }

        const btn = document.querySelector(`#${tabType}-detail .btn-start`);
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = "SAVING...";

        fetch(`/api/datasets/${state.activeDataset}/trajectories/${trajId}/edit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, neg_prompt, cfg })
        }).then(r => r.json()).then(data => {
            btn.disabled = false;
            btn.textContent = originalText;

            if (data.ok) {
                // Update local state in both lists
                for (const list of [state.pendingTrajectories, state.archivedTrajectories]) {
                    const traj = list.find(t => t.id === trajId);
                    if (traj) {
                        traj.prompt = prompt;
                        const meta = traj.metadata ? JSON.parse(traj.metadata) : {};
                        meta.neg = neg_prompt;
                        meta.cfg = cfg;
                        traj.metadata = JSON.stringify(meta);
                    }
                }
                // Notify user
                btn.style.background = "var(--green)";
                setTimeout(() => { btn.style.background = ""; }, 1000);
            } else {
                alert(`Save failed: ${data.detail}`);
            }
        });
    };

    // Bulk edit: apply a universal trigger word (prepended, so any existing per-image
    // captions are kept) and/or a common negative prompt / CFG value across every
    // currently-selected trajectory (use "Select All" first for the whole dataset) in a
    // single request, instead of editing trajectories one at a time.
    window.showBulkEditPanel = function() {
        const ids = Array.from(state.selectedTrajs);
        if (ids.length === 0) {
            alert("Nothing selected. Click trajectories to select them, or use \"Select All\", then try again.");
            return;
        }
        const panel = document.getElementById("inspection-detail");
        if (!panel) return;
        state.activeInspectionTrajId = null;

        panel.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <span style="font-weight:700; font-size:1.1rem;">Bulk Edit (${ids.length} selected)</span>
            </div>

            <div class="option-row mb-1">
                <label class="option-label">Trigger Word / Prompt</label>
                <textarea id="bulk-prompt" class="cfg-input" style="height:80px; width:100%; resize:vertical;" placeholder="e.g. sks_character"></textarea>
                <div style="display:flex; gap:1rem; margin-top:0.5rem; font-size:0.8rem;">
                    <label><input type="radio" name="bulk-prompt-mode" value="prepend" checked> Prepend to existing prompt</label>
                    <label><input type="radio" name="bulk-prompt-mode" value="set"> Replace prompt entirely</label>
                </div>
                <div class="text-dim" style="font-size:0.75rem; margin-top:0.25rem;">
                    Leave blank to leave prompts untouched. Prepend skips a trajectory if its
                    prompt already starts with this text, so it's safe to re-apply.
                </div>
            </div>

            <div class="option-row mb-1">
                <label class="option-label">Negative Prompt</label>
                <textarea id="bulk-neg-prompt" class="cfg-input" style="height:60px; width:100%; resize:vertical;" placeholder="Leave blank to leave unchanged"></textarea>
            </div>

            <div class="option-row mb-1-5">
                <label class="option-label">Guidance Scale (CFG)</label>
                <div style="display:flex; gap:0.5rem; align-items:center;">
                    <input type="number" id="bulk-cfg" class="cfg-input" step="0.1" style="width:100px;" placeholder="unset">
                    <span class="text-dim" style="font-size:0.75rem;">Leave blank to leave unchanged. Only affects CFG-aware training.</span>
                </div>
            </div>

            <div style="margin-top:auto; padding-top:1rem; border-top:1px solid var(--surface-alt);">
                <button class="btn btn-start btn-block" onclick="window.applyBulkEdit()">
                    APPLY TO ${ids.length} SELECTED
                </button>
            </div>
        `;
    };

    window.applyBulkEdit = function() {
        const ids = Array.from(state.selectedTrajs);
        if (ids.length === 0 || !state.activeDataset) return;

        const promptVal = document.getElementById("bulk-prompt").value;
        const negVal = document.getElementById("bulk-neg-prompt").value;
        const cfgRaw = document.getElementById("bulk-cfg").value;
        const modeEl = document.querySelector('input[name="bulk-prompt-mode"]:checked');
        const promptMode = modeEl ? modeEl.value : "prepend";

        const body = { traj_ids: ids, prompt_mode: promptMode };
        if (promptVal.trim() !== "") body.prompt = promptVal;
        if (negVal.trim() !== "") body.neg_prompt = negVal;
        if (cfgRaw.trim() !== "") {
            const cfgNum = parseFloat(cfgRaw);
            if (isNaN(cfgNum)) { alert("CFG must be a number"); return; }
            body.cfg = cfgNum;
        }

        const btn = document.querySelector("#inspection-detail .btn-start");
        const originalText = btn ? btn.textContent : "";
        if (btn) { btn.disabled = true; btn.textContent = "APPLYING..."; }

        fetch(`/api/datasets/${state.activeDataset}/trajectories/bulk-edit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        }).then(r => r.json()).then(data => {
            if (data.ok) {
                loadPending();
                loadArchived();
                alert(`Updated ${data.updated} trajectories.`);
            } else {
                alert(`Bulk edit failed: ${data.detail}`);
                if (btn) { btn.disabled = false; btn.textContent = originalText; }
            }
        }).catch(err => {
            alert(`Bulk edit failed: ${err}`);
            if (btn) { btn.disabled = false; btn.textContent = originalText; }
        });
    };

    function renderInspection() {
        const gallery = document.getElementById("inspection-gallery");
        const countEl = document.getElementById("inspection-count");
        if (!gallery) return;

        state.activeInspectionTrajId = null;
        
        countEl.textContent = `${state.pendingTrajectories.length} ITEMS VISIBLE`;

        if (!state.activeDataset) {
            gallery.innerHTML = '<p class="placeholder p-2">Please select a dataset from the library.</p>';
            showDetail(null, "inspection");
            return;
        }

        if (state.pendingTrajectories.length === 0) {
            gallery.innerHTML = '<p class="placeholder p-2">No items found in staging area.</p>';
            showDetail(null, "inspection");
            return;
        }

        const fragment = document.createDocumentFragment();
        
        state.pendingTrajectories.forEach(traj => {
            const item = document.createElement("div");
            item.className = `preview-item ${state.selectedTrajs.has(traj.id) ? "selected" : ""}`;
            
            const previewSrc = traj.preview_path ? `/datasets/${state.activeDataset}/${traj.preview_path}` : "";
            const meta = traj.metadata ? JSON.parse(traj.metadata) : {};
            const cfgLabel = meta.cfg ? ` | CFG: ${meta.cfg.toFixed(1)}` : "";

            const isBad = meta.type === "bad";

            item.dataset.trajId = traj.id;
            item.innerHTML = `
                ${previewSrc ? `<img src="${previewSrc}" loading="lazy" onerror="this.src='/static/broken.png'; this.className='broken-img'">` : '<div class="no-preview">NO PREVIEW</div>'}
                <div class="preview-meta">ID: ${traj.id}${cfgLabel}</div>
                <button class="btn btn-small traj-type-btn ${isBad ? 'btn-danger' : 'btn-secondary'}" 
                        style="position:absolute; bottom:4px; right:4px; font-size:0.65rem; padding:0.15rem 0.4rem; z-index:3;"
                        onclick="event.stopPropagation(); window.toggleType(${traj.id})">
                    ${isBad ? 'BAD' : 'good'}
                </button>
            `;

            item.onclick = () => {
                if (state.selectedTrajs.has(traj.id)) {
                    state.selectedTrajs.delete(traj.id);
                    item.classList.remove("selected");
                } else {
                    state.selectedTrajs.add(traj.id);
                    item.classList.add("selected");
                }
                showDetail(traj.id, "inspection");
            };
            fragment.appendChild(item);
        });

        gallery.innerHTML = "";
        gallery.appendChild(fragment);

        // Clear detail panel
        showDetail(null, "inspection");
    }

    function startPolling() {
        if (state.pollInterval) clearInterval(state.pollInterval);
        state.pollInterval = setInterval(pollTasks, 2000);
        pollTasks();
    }

    function pollTasks() {
        if (!state.activeDataset) return;
        fetch(`/api/datasets/${state.activeDataset}/tasks/active`)
            .then(r => r.json())
            .then(tasks => {
                state.activeTasks = tasks;
                renderActiveTasks();
            });
    }

    function renderActiveTasks() {
        let container = document.getElementById("active-tasks-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "active-tasks-container";
            container.style = "position:fixed; bottom:20px; right:20px; width:350px; z-index:1000;";
            document.body.appendChild(container);
        }
        container.innerHTML = "";

        state.activeTasks.forEach(task => {
            const pct = Math.round((task.current_val / task.total_val) * 100) || 0;
            const card = document.createElement("div");
            card.className = "card p-1";
            card.style = "margin-bottom:10px; background:var(--surface-alt); border:1px solid var(--accent);";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.85rem;">
                    <span style="font-weight:700; color:var(--accent)">RUNNING: ${task.type.toUpperCase()}</span>
                    <span>${task.current_val} / ${task.total_val}</span>
                </div>
                <div style="height:8px; background:var(--bg); border-radius:4px; overflow:hidden; margin-bottom:10px;">
                    <div style="height:100%; width:${pct}%; background:var(--accent); transition:width 0.3s;"></div>
                </div>
                <button class="btn btn-kill btn-small btn-block" style="padding:0.4rem;" onclick="window.killTask(${task.id})">
                    KILL TASK
                </button>
            `;
            container.appendChild(card);
        });
    }

    window.killTask = function(taskId) {
        if (!confirm("Really kill this background task?")) return;
        fetch(`/api/datasets/${state.activeDataset}/tasks/${taskId}/kill`, { method: "POST" })
            .then(() => pollTasks());
    };

    window.updateTypeButton = function(trajId) {
        document.querySelectorAll(`[data-traj-id="${trajId}"] .traj-type-btn`).forEach(btn => {
            const isBad = btn.textContent.trim() === "good";
            btn.textContent = isBad ? "BAD" : "good";
            btn.className = `btn btn-small traj-type-btn ${isBad ? 'btn-danger' : 'btn-secondary'}`;
        });
    };

    window.toggleType = function(trajId) {
        // Optimistic: toggle immediately, no flash
        window.updateTypeButton(trajId);
        for (const list of [state.pendingTrajectories, state.archivedTrajectories]) {
            const t = list.find(t => t.id === trajId);
            if (t && t.metadata) {
                try {
                    const m = JSON.parse(t.metadata);
                    m.type = m.type === "bad" ? "good" : "bad";
                    t.metadata = JSON.stringify(m);
                } catch(e) {}
            }
        }
        // Re-render detail panels if they're showing this trajectory
        if (state.activeDetailTrajId === trajId) {
            showDetail(trajId, "explorer");
        }
        if (state.activeInspectionTrajId === trajId) {
            showDetail(trajId, "inspection");
        }
        fetch(`/api/datasets/${state.activeDataset}/trajectories/${trajId}/toggle-type`, { method: "POST" });
    };

    window.rejectSelected = function() {
        const ids = Array.from(state.selectedTrajs);
        if (ids.length === 0 || !state.activeDataset) return;

        Promise.all(ids.map(id => 
            fetch(`/api/datasets/${state.activeDataset}/trajectories/${id}/reject`, { 
                method: "POST"
            })
        )).then(() => {
            state.selectedTrajs.clear();
            loadPending();
        });
    };

    window.commitSelected = function() {
        const ids = Array.from(state.selectedTrajs);

        if (ids.length === 0 || !state.activeDataset) {
            alert("Please select items to commit first.");
            return;
        }

        // Set a default name based on timestamp to avoid blocking with prompts
        const defaultName = `set_${Math.floor(Date.now() / 1000)}`;

        fetch(`/api/datasets/${state.activeDataset}/training-sets/create`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                set_name: defaultName,
                traj_ids: ids
            })
        }).then(r => r.json()).then(data => {
            if (data.ok) {
                state.selectedTrajs.clear();
                // Crucial: Reload both views so items "move" visually
                loadPending();
                loadArchived();
            } else {
                alert(`Commit failed: ${data.detail}`);
            }
        });
    };

    function startGeneration() {
        if (!state.activeDataset) {
            alert("Select a dataset first.");
            return;
        }
        
        saveSettings();

        const type = document.getElementById("gen-source-type").value;
        const getInt = id => parseInt(document.getElementById(id).value);
        const getFloat = id => parseFloat(document.getElementById(id).value);
        const getStr = id => document.getElementById(id).value;

        const payload = {
            dataset_name: state.activeDataset,
            type: type
        };

        if (type === "teacher") {
            payload.model_name = getStr("gen-teacher-model");
            payload.steps_min = getInt("gen-steps-min");
            payload.steps_max = getInt("gen-steps-max");
            payload.cfg_min = getFloat("gen-cfg-min");
            payload.cfg_max = getFloat("gen-cfg-max");
            payload.batch_size = getInt("gen-batch-size");
            payload.n_conditions = getInt("gen-unique-conds");
            payload.n_samples_per_cond = getInt("gen-samples-per-cond");
            payload.seed = getInt("gen-seed");
            payload.t_mode = getStr("gen-teacher-t-mode");
            payload.t_low = getInt("gen-teacher-t-low");
            payload.t_high = getInt("gen-teacher-t-high");
            
            payload.prompt_mode = getStr("gen-pos-mode");
            if (payload.prompt_mode === "list") {
                payload.prompts = getStr("gen-prompts");
            } else {
                payload.keywords = getStr("gen-keywords");
                payload.keywords_file = getStr("gen-keywords-file");
                payload.template = getStr("gen-pos-template");
                payload.min_keywords = getInt("gen-pos-min");
                payload.max_keywords = getInt("gen-pos-max");
            }

            payload.neg_mode = getStr("gen-neg-mode");
            if (payload.neg_mode === "list") {
                payload.negative_prompt = getStr("gen-negative-prompt");
            } else {
                payload.neg_keywords = getStr("gen-neg-keywords");
                payload.neg_keywords_file = getStr("gen-neg-keywords-file");
                payload.neg_template = getStr("gen-neg-template");
                payload.neg_min_keywords = getInt("gen-neg-min");
                payload.neg_max_keywords = getInt("gen-neg-max");
            }

        } else {
            payload.model_name = getStr("gen-vae-model");
            payload.image_dir = getStr("gen-image-dir");
            payload.recursive = document.getElementById("gen-recursive").checked;
            payload.auto_caption = document.getElementById("gen-auto-caption").checked;
            payload.resize_mode = getStr("gen-resize-mode");
            payload.ingest_latent_size = parseInt(getStr("gen-ingest-latent-size"));
            payload.t_mode = getStr("gen-real-t-mode");
            payload.t_low = getInt("gen-real-t-low");
            payload.t_high = getInt("gen-real-t-high");
            payload.n_timesteps = getInt("gen-real-n-timesteps");
        }

        fetch("/api/datasets/tasks/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        }).then(r => r.json()).then(data => {
            if (data.ok) {
                const seedEl = document.getElementById("gen-seed");
                if (seedEl) {
                    seedEl.value = parseInt(seedEl.value) + 1;
                    saveSettings();
                }
            } else {
                alert(`Error: ${data.detail}`);
            }
        });
    }

    function newDataset() {
        const name = prompt("Dataset Name:");
        if (!name) return;
        const desc = prompt("Description (optional):");

        fetch("/api/datasets/create", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, description: desc })
        }).then(r => r.json()).then(() => loadDatasets());
    }

    function loadCheckpoints() {
        fetch("/api/datasets/checkpoints")
            .then(r => r.json())
            .then(data => {
                const selects = [
                    document.getElementById("gen-teacher-model"),
                    document.getElementById("gen-vae-model")
                ];
                selects.forEach(select => {
                    if (!select) return;
                    const oldVal = select.value;
                    select.innerHTML = "";
                    data.forEach(cp => {
                        const opt = document.createElement("option");
                        opt.value = cp;
                        opt.textContent = cp;
                        select.appendChild(opt);
                    });
                    if (oldVal) select.value = oldVal;
                });
                restoreSettings();
            });
    }

    function init() {
        initTabs();
        initGeneratorUI();
        loadDatasets().then(() => {
            if (state.activeDataset) {
                loadPending();
                loadArchived();
                startPolling();
            }
        });
        loadCheckpoints();

        const btnKill = document.querySelector(".btn-kill");
        if (btnKill) btnKill.onclick = window.rejectSelected;

        const btnCommit = document.querySelector("#tab-inspection .btn-start");
        if (btnCommit) btnCommit.onclick = window.commitSelected;
        
        const btnNew = document.getElementById("btn-new-dataset");
        if (btnNew) btnNew.onclick = newDataset;

        const btnGenT = document.querySelector("#gen-teacher-options .btn-start");
        if (btnGenT) btnGenT.onclick = startGeneration;
        
        const btnGenR = document.querySelector("#gen-real-options .btn-start");
        if (btnGenR) btnGenR.onclick = startGeneration;

        const btnSelectAll = document.getElementById("btn-select-all");
        if (btnSelectAll) {
            btnSelectAll.onclick = () => {
                state.pendingTrajectories.forEach(t => state.selectedTrajs.add(t.id));
                renderInspection();
            };
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
