/* ---------------------------------------------------------------------------
   Training Control — handles start/stop/reset actions
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    function startRun() {
        var btn = document.getElementById("btn-start");
        btn.disabled = true;
        
        window.logToConsole("Starting training...", "info");

        var fd = window.OptionTree.collectFormData();
        
        fetch("/api/run/start", { method: "POST", body: fd })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) {
                    window.logToConsole("Start failed: " + data.error, "error");
                    btn.disabled = false;
                } else {
                    window.logToConsole("Run #" + data.run_id + " launched successfully.", "success");
                    window.OptionTree.load();
                }
            })
            .catch(function (err) {
                window.logToConsole("Network error during start: " + err, "error");
                btn.disabled = false;
            });
    }

    function stopRun(force) {
        var action = force ? "Killing" : "Stopping";
        window.logToConsole(action + " training process...", "warn");

        var url = "/api/run/stop?force=" + (force || false);
        fetch(url, { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) {
                    window.logToConsole("Stop failed: " + data.error, "error");
                } else {
                    var msg = force ? "Kill signal sent." : "Stop requested (saving checkpoint...).";
                    window.logToConsole(msg, "info");
                }
            });
    }

    function resetWorker() {
        if (!confirm("Force-reset the worker state? Only use if the UI is stuck but the process is gone.")) {
            return;
        }
        window.logToConsole("Resetting worker state...", "warn");
        fetch("/api/run/reset", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                window.logToConsole("Worker reset complete.", "success");
                window.UIStateManager.fetchCurrentState();
            });
    }

    function updateControlButtons(status) {
        var startBtn = document.getElementById("btn-start");
        var stopBtn = document.getElementById("btn-stop");
        var killBtn = document.getElementById("btn-kill");

        if (status === "running") {
            startBtn.style.display = "none";
            stopBtn.style.display = "block";
            killBtn.style.display = "block";
        } else {
            startBtn.style.display = "block";
            startBtn.disabled = false;
            stopBtn.style.display = "none";
            killBtn.style.display = "none";
        }
    }

    // Public API
    window.TrainingControl = {
        start: startRun,
        stop: function() { stopRun(false); },
        kill: function() { stopRun(true); },
        reset: resetWorker,
        updateButtons: updateControlButtons
    };
})();
