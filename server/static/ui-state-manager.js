/* ---------------------------------------------------------------------------
   UI State Manager — orchestrates global UI updates from server state
   --------------------------------------------------------------------------- */

(function () {
    "use strict";
var state = {
    status: "idle",
    runId: null,
    lastStatus: null,
    lastPhase: null,
    loadedEventsFor: null
};

function fetchRecentEvents(runId) {
    if (state.loadedEventsFor === runId) return;
    state.loadedEventsFor = runId;

    fetch("/api/runs/" + runId + "/events?limit=500")
        .then(function (r) { return r.json(); })
        .then(function (events) {
            if (!events || !events.length) return;

            window.ChartManager.reset();
            events.forEach(function (ev) {
                if (ev.event_type === "step") {
                    window.ChartManager.addPoint(ev.step, ev.loss, ev.avg_loss);
                }
            });
        })
        .catch(function (err) {
            console.error("Failed to fetch events:", err);
        });
}

function updateStatusUI(status, runId) {    if (runId && runId !== state.runId) {
        // New run detected or switched
        state.runId = runId;
        fetchRecentEvents(runId);
    }

    state.status = status;
    state.runId = runId;

    // ...

        // Update badge
        var badge = document.getElementById("status-badge");
        if (badge) {
            badge.className = "status-badge status-" + status;
            badge.textContent = status.toUpperCase();
        }

        // Update sidebar buttons
        window.TrainingControl.updateButtons(status);

        // Console log on change
        if (status !== state.lastStatus) {
            var label = status.toUpperCase();
            var type = "info";
            if (status === "running") type = "success";
            if (status === "failed" || status === "killed") type = "error";
            if (status === "stopped") type = "warn";
            
            var msg = "Status changed to " + label;
            if (runId) msg += " (Run #" + runId + ")";
            window.logToConsole(msg, type);
            state.lastStatus = status;
        }
    }

    function updateProgress(data) {
        // Main progress
        var fill = document.getElementById("progress-fill");
        var text = document.getElementById("progress-text");
        if (fill && data.total) {
            var pct = (data.step / data.total) * 100;
            fill.style.width = pct + "%";
            
            if (data.phase === "cache") {
                text.textContent = "Caching... (" + data.step.toLocaleString() + " / " + data.total.toLocaleString() + ")";
            } else {
                text.textContent = data.step.toLocaleString() + " / " + data.total.toLocaleString() + " steps";
            }
        }

        // Metrics
        if (data.loss !== undefined && data.loss !== null) {
            var l = document.getElementById("metric-loss");
            if (l) l.textContent = data.loss.toFixed(4);
            window.ChartManager.addPoint(data.step, data.loss, data.avg_loss);
        }
        if (data.avg_loss !== undefined && data.avg_loss !== null) {
            var a = document.getElementById("metric-avg");
            if (a) a.textContent = data.avg_loss.toFixed(4);
        }
        if (data.lr !== undefined && data.lr !== null) {
            var lr = document.getElementById("metric-lr");
            if (lr) lr.textContent = data.lr.toExponential(2);
        }

        // Phase tracking
        if (data.phase && data.phase !== state.lastPhase) {
            window.logToConsole("Entering " + data.phase + " phase...", "info");
            state.lastPhase = data.phase;
        }

        // Cache progress
        var cacheContainer = document.getElementById("cache-progress-bar");
        var cacheFill = document.getElementById("cache-progress-fill");
        var cacheText = document.getElementById("cache-progress-text");
        var mainWrap = document.getElementById("progress-main-wrap");

        if (data.phase === "cache" && data.cache_total > 0) {
            if (cacheContainer) cacheContainer.classList.add("active");
            if (mainWrap) mainWrap.classList.add("with-cache");
            if (cacheFill) {
                var cPct = (data.cache_done / data.cache_total) * 100;
                cacheFill.style.width = cPct + "%";
            }
            if (cacheText) {
                cacheText.textContent = "Cache: " + data.cache_done + " / " + data.cache_total;
            }
        } else {
            if (cacheContainer) cacheContainer.classList.remove("active");
            if (mainWrap) mainWrap.classList.remove("with-cache");
        }
    }

    function fetchCurrentState() {
        fetch("/api/run/status")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.status === "idle") {
                    updateStatusUI("idle", null);
                } else {
                    updateStatusUI(data.status, data.id);
                    updateProgress({
                        step: data.done_steps || 0,
                        total: data.total_steps || 0,
                        loss: data.current_loss,
                        avg_loss: data.avg_loss,
                        phase: data.phase,
                        cache_done: data.cache_done,
                        cache_total: data.cache_total
                    });
                }
            });
    }

    // Public API
    window.UIStateManager = {
        updateStatus: updateStatusUI,
        updateProgress: updateProgress,
        fetchCurrentState: fetchCurrentState,
        getState: function () { return state; }
    };
})();
