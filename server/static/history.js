/* ---------------------------------------------------------------------------
   History & Log Manager — handles run history and log viewer
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    function loadHistory() {
        fetch("/api/runs?limit=50")
            .then(function (r) { return r.json(); })
            .then(function (runs) {
                var container = document.getElementById("history-list");
                if (!runs.length) {
                    container.innerHTML = '<p class="placeholder">No runs yet.</p>';
                    return;
                }
                var html = "";
                runs.forEach(function (run) {
                    var duration = "";
                    if (run.started_at && run.finished_at) {
                        duration = formatDuration(run.finished_at - run.started_at);
                    } else if (run.started_at) {
                        duration = "running " + formatDuration((Date.now() / 1000) - run.started_at);
                    }
                    var lossStr = run.avg_loss != null ? " avg=" + run.avg_loss.toFixed(4) : "";
                    html += '<div class="history-item" data-run-id="' + run.id + '" style="cursor:pointer">' +
                        '<span class="history-id">#' + run.id + '</span>' +
                        '<div class="history-meta">' +
                            '<span class="config-name">' + escapeHtml(run.config_path) + '</span>' +
                            ' <span class="detail">[' + escapeHtml(run.mode) + ']' +
                            ' ' + run.done_steps + '/' + run.total_steps + ' steps' +
                            lossStr +
                            (duration ? ' — ' + duration : '') +
                            '</span>' +
                        '</div>' +
                        '<span class="history-status ' + run.status + '">' + escapeHtml(run.status) + '</span>' +
                        '</div>';
                });
                container.innerHTML = html;

                // Add click handlers to load logs for each run
                container.querySelectorAll(".history-item").forEach(function (item) {
                    item.addEventListener("click", function () {
                        var runId = this.dataset.runId;
                        loadRunLog(runId);
                    });
                });
            });
    }

    function loadRunLog(runId) {
        fetch("/api/runs/" + runId + "/log?lines=500")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                document.querySelectorAll(".tab").forEach(function (t) { t.classList.remove("active"); });
                document.querySelectorAll(".tab-content").forEach(function (p) { p.classList.remove("active"); });
                document.querySelector('[data-tab="log"]').classList.add("active");
                document.getElementById("tab-log").classList.add("active");

                var el = document.getElementById("log-output");
                if (data.error) {
                    el.textContent = "Run #" + runId + ": " + data.error;
                    return;
                }
                if (!data.log) {
                    el.textContent = "Run #" + runId + ": No log output available.";
                    return;
                }

                var lines = data.log.split("\n");
                var latestProgressBar = null;
                var importantLines = [];
                var hasRunEnded = false;

                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i];
                    var isProgressBar = (line.indexOf("[") >= 0 && line.indexOf("]") >= 0 &&
                                        (line.indexOf("|") >= 0 || line.indexOf("%") >= 0)) ||
                                        (line.indexOf("step=") >= 0 && line.indexOf("loss=") >= 0);

                    if (line.indexOf("--- RUN ENDED") >= 0) {
                        importantLines.push(line);
                        hasRunEnded = true;
                    } else if (line.indexOf("--- RUN STARTED") >= 0) {
                        importantLines.push(line);
                    } else if (isProgressBar) {
                        latestProgressBar = line.trim();
                    } else if (line.indexOf("PID=") >= 0) {
                        importantLines.push(line);
                    } else if (line.indexOf("Exited with code") >= 0 ||
                               line.indexOf("Final:") >= 0 ||
                               line.indexOf("Monitor error") >= 0 ||
                               line.indexOf("Traceback") >= 0) {
                        importantLines.push(line);
                    } else if (!isProgressBar && line.trim()) {
                        importantLines.push(line);
                    }
                }

                if (latestProgressBar && !hasRunEnded) {
                    importantLines.push(">> " + latestProgressBar);
                }

                var header = "=== Run #" + runId + " Log ===";
                el.textContent = header + "\n" + importantLines.join("\n");
                el.scrollTop = el.scrollHeight;
            });
    }

    function loadLog() {
        fetch("/api/run/log?lines=500")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var el = document.getElementById("log-output");
                if (!data.log) {
                    el.textContent = "No log output available.";
                    return;
                }

                var lines = data.log.split("\n");
                var latestProgressBar = null;
                var importantLines = [];

                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i];
                    var isProgressBar = (line.indexOf("[") >= 0 && line.indexOf("]") >= 0 &&
                                        (line.indexOf("|") >= 0 || line.indexOf("%") >= 0)) ||
                                        (line.indexOf("step=") >= 0 && line.indexOf("loss=") >= 0);

                    if (line.indexOf("--- RUN STARTED") >= 0 ||
                        line.indexOf("--- RUN ENDED") >= 0) {
                        importantLines.push(line);
                    } else if (isProgressBar) {
                        latestProgressBar = line.trim();
                    } else if (line.indexOf("PID=") >= 0) {
                        importantLines.push(line);
                    } else if (line.indexOf("Exited with code") >= 0 ||
                               line.indexOf("Final:") >= 0 ||
                               line.indexOf("Monitor error") >= 0 ||
                               line.indexOf("Traceback") >= 0) {
                        importantLines.push(line);
                    } else if (!isProgressBar && line.trim()) {
                        importantLines.push(line);
                    }
                }

                if (latestProgressBar) {
                    importantLines.push(">> " + latestProgressBar);
                }

                el.textContent = importantLines.join("\n") || "No significant log output.";
                el.scrollTop = el.scrollHeight;
            });
    }

    function formatDuration(seconds) {
        if (seconds < 0) seconds = 0;
        if (seconds < 60) return Math.round(seconds) + "s";
        if (seconds < 3600) return Math.floor(seconds / 60) + "m " + Math.round(seconds % 60) + "s";
        var h = Math.floor(seconds / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        return h + "h " + m + "m";
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function clearLogs() {
        if (!confirm("Permanently delete all log files from disk?")) return;
        
        fetch("/api/runs/logs/clear", { method: "POST" })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                window.logToConsole("Deleted " + data.count + " log files.", "success");
                loadHistory();
            });
    }

    // Public API
    window.HistoryManager = {
        load: loadHistory,
        loadLog: loadLog,
        clearLogs: clearLogs,
    };
})();
