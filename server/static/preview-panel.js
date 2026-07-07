/* ---------------------------------------------------------------------------
   Preview Panel — mid-training preview settings + gallery
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    var pollTimer = null;

    var SETTINGS_EXPANDED_KEY = "comfy_preview_settings_expanded";

    function configPath() {
        var el = document.getElementById("cfg-file-path");
        return el ? el.value.trim() : "convert-cfg.toml";
    }

    function loadSettings() {
        fetch("/api/config?path=" + encodeURIComponent(configPath()))
            .then(function (r) { return r.json(); })
            .then(function (cfg) {
                if (cfg.error || !cfg.preview) return;
                var p = cfg.preview;
                document.getElementById("prev-enabled").checked = !!p.enabled;
                document.getElementById("prev-every-n").value = p.every_n_steps;
                document.getElementById("prev-prompts").value = p.prompts;
                document.getElementById("prev-negative").value = p.negative_prompt;
                document.getElementById("prev-steps").value = p.steps;
                document.getElementById("prev-cfg").value = p.cfg;
                document.getElementById("prev-resolution").value = p.resolution;
                document.getElementById("prev-seed").value = p.seed;
                document.getElementById("prev-max-batch").value = p.max_batch_size;
            })
            .catch(function (err) { console.error("Failed to load preview settings:", err); });
    }

    function saveSettings() {
        var fd = new FormData();
        fd.append("config", configPath());
        fd.append("preview.enabled", document.getElementById("prev-enabled").checked ? "true" : "false");
        fd.append("preview.every_n_steps", document.getElementById("prev-every-n").value);
        fd.append("preview.prompts", document.getElementById("prev-prompts").value);
        fd.append("preview.negative_prompt", document.getElementById("prev-negative").value);
        fd.append("preview.steps", document.getElementById("prev-steps").value);
        fd.append("preview.cfg", document.getElementById("prev-cfg").value);
        fd.append("preview.resolution", document.getElementById("prev-resolution").value);
        fd.append("preview.seed", document.getElementById("prev-seed").value);
        fd.append("preview.max_batch_size", document.getElementById("prev-max-batch").value);

        var btn = document.getElementById("btn-save-preview");
        var orig = btn.textContent;
        btn.textContent = "Saving...";
        btn.disabled = true;

        fetch("/api/config", { method: "PUT", body: fd })
            .then(function (r) { return r.json(); })
            .then(function (cfg) {
                btn.textContent = cfg.error ? ("Error: " + cfg.error) : "Saved ✓";
            })
            .catch(function () {
                btn.textContent = "Error!";
            })
            .finally(function () {
                setTimeout(function () { btn.textContent = orig; btn.disabled = false; }, 1500);
            });
    }

    function openLightbox(url, caption) {
        var box = document.getElementById("preview-lightbox");
        var img = document.getElementById("preview-lightbox-img");
        var cap = document.getElementById("preview-lightbox-caption");
        if (!box || !img) return;
        img.src = url;
        if (cap) cap.textContent = caption || "";
        box.classList.add("open");
    }

    function closeLightbox() {
        var box = document.getElementById("preview-lightbox");
        if (box) box.classList.remove("open");
    }

    function renderGallery(steps) {
        var gallery = document.getElementById("preview-gallery");
        if (!gallery) return;

        if (!steps || !steps.length) {
            gallery.innerHTML = '<p class="text-dim" style="padding:2rem; text-align:center;">' +
                "No previews yet. Enable preview generation above and start (or wait for) a training run." +
                "</p>";
            return;
        }

        var sorted = steps.slice().sort(function (a, b) { return b.step - a.step; });
        gallery.innerHTML = sorted.map(function (entry) {
            var imgs = entry.urls.map(function (url, i) {
                return '<img src="' + url + '" class="preview-thumb" loading="lazy" alt="preview" ' +
                    'data-url="' + url + '" data-caption="Step ' + entry.step + ' \u2014 #' + (i + 1) + '">';
            }).join("");
            return '<div class="preview-step-group">' +
                '<div class="preview-step-label">STEP ' + entry.step + "</div>" +
                '<div class="preview-step-images">' + imgs + "</div>" +
                "</div>";
        }).join("");
    }

    function pollGallery() {
        var runId = (window.UIStateManager && window.UIStateManager.getState().runId) || null;

        var idPromise = runId
            ? Promise.resolve(runId)
            // No active run -- fall back to the most recent one so previews
            // from a just-finished run stay visible instead of disappearing.
            : fetch("/api/runs?limit=1")
                .then(function (r) { return r.json(); })
                .then(function (rows) { return (rows && rows.length) ? rows[0].id : null; })
                .catch(function () { return null; });

        idPromise.then(function (id) {
            if (!id) { renderGallery([]); return; }
            fetch("/api/runs/" + id + "/previews")
                .then(function (r) { return r.json(); })
                .then(function (data) { renderGallery(data.steps || []); })
                .catch(function (err) { console.error("Failed to fetch previews:", err); });
        });
    }

    function initSettingsToggle() {
        var panel = document.getElementById("preview-settings-panel");
        var toggle = document.getElementById("preview-settings-toggle");
        if (!panel || !toggle) return;

        var expanded = localStorage.getItem(SETTINGS_EXPANDED_KEY) === "true";
        panel.classList.toggle("expanded", expanded);

        toggle.addEventListener("click", function () {
            var next = !panel.classList.contains("expanded");
            panel.classList.toggle("expanded", next);
            localStorage.setItem(SETTINGS_EXPANDED_KEY, next ? "true" : "false");
        });
    }

    function init() {
        var saveBtn = document.getElementById("btn-save-preview");
        if (saveBtn) saveBtn.onclick = saveSettings;

        initSettingsToggle();

        var gallery = document.getElementById("preview-gallery");
        if (gallery) {
            // Event delegation -- the gallery's innerHTML is replaced on every
            // poll, so binding to individual <img> elements would silently
            // stop working after the first refresh.
            gallery.addEventListener("click", function (e) {
                if (e.target && e.target.classList.contains("preview-thumb")) {
                    openLightbox(e.target.dataset.url, e.target.dataset.caption);
                }
            });
        }

        var lightbox = document.getElementById("preview-lightbox");
        if (lightbox) {
            lightbox.addEventListener("click", function (e) {
                if (e.target === lightbox || e.target.id === "preview-lightbox-close") {
                    closeLightbox();
                }
            });
        }
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") closeLightbox();
        });

        loadSettings();
        pollGallery();
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollGallery, 10000);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    window.PreviewPanel = { refresh: pollGallery, reloadSettings: loadSettings };
})();
