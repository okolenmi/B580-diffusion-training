/* ---------------------------------------------------------------------------
   Distillation Converter — Main application initializer
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    // Console logging helper
    window.logToConsole = function(msg, type) {
        var consoleOutput = document.getElementById("console-output");
        if (!consoleOutput) return;

        var line = document.createElement("div");
        line.className = "console-line " + (type || "info");
        line.textContent = "[" + new Date().toLocaleTimeString() + "] " + msg;
        
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;

        // Keep only last 50 lines to prevent memory bloat
        while (consoleOutput.childNodes.length > 50) {
            consoleOutput.removeChild(consoleOutput.firstChild);
        }
    };

    // Sidebar navigation management
    function initSidebar() {
        var navItems = document.querySelectorAll(".nav-item");
        var panels = document.querySelectorAll(".tab-content");

        navItems.forEach(function (item) {
            item.addEventListener("click", function () {
                var target = item.dataset.tab;
                navItems.forEach(function (t) { t.classList.remove("active"); });
                panels.forEach(function (p) { p.classList.remove("active"); });
                item.classList.add("active");
                var panel = document.getElementById("tab-" + target);
                if (panel) panel.classList.add("active");

                if (target === "dashboard") window.ChartManager.update();
                if (target === "history") window.HistoryManager.load();
                if (target === "log") window.HistoryManager.loadLog();
            });
        });
    }

    // Event listeners for new UI buttons
    function initEventListeners() {
        // Buttons are now mostly handled by inline onclick for simplicity in the layout,
        // but some specialized ones remain here.

        // History refresh
        var refreshHistoryBtn = document.getElementById("btn-refresh-history");
        if (refreshHistoryBtn) {
            refreshHistoryBtn.addEventListener("click", function () {
                window.HistoryManager.load();
            });
        }

        // Log refresh
        var refreshLogBtn = document.getElementById("btn-refresh-log");
        if (refreshLogBtn) {
            refreshLogBtn.addEventListener("click", function () {
                window.HistoryManager.loadLog();
            });
        }

        // Config editor
        var loadConfigBtn = document.getElementById("btn-load-config");
        if (loadConfigBtn) {
            loadConfigBtn.addEventListener("click", function () {
                window.ConfigEditor.load();
                if (window.OptionTree && typeof window.OptionTree.load === "function") {
                    window.OptionTree.load();
                }
            });
        }

        var saveConfigBtn = document.getElementById("btn-save-config");
        if (saveConfigBtn) {
            saveConfigBtn.addEventListener("click", function () {
                window.ConfigEditor.save();
            });
        }

        var cfgPathInput = document.getElementById("cfg-file-path");
        if (cfgPathInput) {
            cfgPathInput.addEventListener("change", function () {
                if (window.OptionTree && typeof window.OptionTree.load === "function") {
                    window.OptionTree.load();
                }
            });
        }
    }

    // Initialize everything
    function init() {
        initSidebar();
        window.ChartManager.init();
        window.SSEClient.connect();
        window.OptionTree.load();
        initEventListeners();
        
        // Initial state check
        window.UIStateManager.fetchCurrentState();

        // Periodic state refresh (every 5 seconds)
        setInterval(function() {
            window.UIStateManager.fetchCurrentState();
        }, 5000);
        
        window.logToConsole("Application initialized", "success");
    }

    // Wait for DOM to be ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
