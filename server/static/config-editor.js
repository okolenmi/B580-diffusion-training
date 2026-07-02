/* ---------------------------------------------------------------------------
   Config Editor — handles config file loading and saving
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    function loadConfig() {
        var pathEl = document.getElementById("cfg-file-path");
        if (!pathEl) return;
        var path = pathEl.value.trim();
        if (!path) return;
        fetch("/api/config/raw?path=" + encodeURIComponent(path))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var editor = document.getElementById("cfg-editor");
                var result = document.getElementById("cfg-result");
                if (!editor && !result) return;
                if (data.error) {
                    if (result) result.innerHTML =
                        '<span class="error-msg">' + data.error + '</span>';
                } else {
                    if (editor) editor.value = data.content;
                    if (result) result.innerHTML = "";
                }
            });
    }

    function saveConfig() {
        var pathEl = document.getElementById("cfg-file-path");
        var editor = document.getElementById("cfg-editor");
        if (!pathEl || !editor) return;
        var path = pathEl.value.trim();
        var content = editor.value;
        if (!path) {
            var result = document.getElementById("cfg-result");
            if (result) result.innerHTML =
                '<span class="error-msg">Config path is required</span>';
            return;
        }
        var fd = new FormData();
        fd.append("path", path);
        fd.append("content", content);
        fetch("/api/config/raw", { method: "PUT", body: fd })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var result = document.getElementById("cfg-result");
                if (!result) return;
                if (data.error) {
                    result.innerHTML =
                        '<span class="error-msg">' + data.error + '</span>';
                } else {
                    result.innerHTML =
                        '<span class="success-msg">Saved to ' + escapeHtml(path) + '</span>';
                }
            });
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // Public API
    window.ConfigEditor = {
        load: loadConfig,
        save: saveConfig,
    };
})();
