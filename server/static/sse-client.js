/* ---------------------------------------------------------------------------
   SSE Client — handles live progress streaming
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    var sseSource = null;

    function connectSSE() {
        if (sseSource) {
            sseSource.close();
        }
        // run_id=0 means "subscribe to active run"
        sseSource = new EventSource("/api/sse?run_id=0");

        sseSource.onmessage = function (e) {
            try {
                var msg = JSON.parse(e.data);
                if (msg.type === "connected") {
                    console.log("SSE connected");
                    return;
                }
                handleSSE(msg);
            } catch (err) {
                console.error("SSE parse error:", err);
            }
        };

        sseSource.onerror = function () {
            console.warn("SSE connection lost, retrying...");
            setTimeout(connectSSE, 3000);
        };
    }

    function handleSSE(msg) {
        if (msg.type === "progress") {
            window.UIStateManager.updateProgress(msg);
        } else if (msg.type === "status") {
            window.UIStateManager.updateStatus(msg.status, msg.run_id);
            if (msg.error) {
                window.logToConsole("Error in run #" + msg.run_id + ": " + msg.error, "error");
            }
        }
    }

    // Public API
    window.SSEClient = {
        connect: connectSSE,
    };
})();
