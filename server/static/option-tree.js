/* ---------------------------------------------------------------------------
   Option Tree Renderer — handles dynamic form generation from schema
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    // The option schema from the server
    var optionSchema = [];
    // Current form state: { option_id: value }
    var formState = {};

    // Transient form state (in-memory only — server is source of truth)
    var FORM_STATE_KEY = "converter_form_state";

    function saveState() {
        updateFormState();
        try {
            sessionStorage.setItem(FORM_STATE_KEY, JSON.stringify(formState));
        } catch (e) {
            // Storage full or disabled — silently ignore
        }
    }

    function restoreState() {
        try {
            var raw = sessionStorage.getItem(FORM_STATE_KEY);
            return raw ? JSON.parse(raw) : {};
        } catch (e) {
            return {};
        }
    }

    function clearState() {
        try {
            sessionStorage.removeItem(FORM_STATE_KEY);
        } catch (e) {}
    }

    /**
     * Load the option schema from the server and render the form.
     */
    function loadOptionSchema() {
        clearState();  // Server is source of truth on page load
        var configPathInput = document.getElementById("cfg-file-path");
        var configPath = configPathInput ? configPathInput.value.trim() : "";
        var url = "/api/options/tree";
        if (configPath) {
            url += "?config=" + encodeURIComponent(configPath);
        }
        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (schema) {
                optionSchema = schema;
                renderOptionTree(optionSchema, document.getElementById("option-tree"));
                applyVisibility();
            })
            .catch(function (err) {
                document.getElementById("option-tree").innerHTML =
                    '<p class="placeholder">Failed to load options: ' + err + '</p>';
            });
    }

    /**
     * Render a list of option descriptors into a container.
     */
    function renderOptionTree(options, container) {
        if (!container) return;
        container.innerHTML = "";

        // Group options by group name
        var groups = {};
        options.forEach(function(opt) {
            var g = opt.group || "General";
            if (!groups[g]) groups[g] = [];
            groups[g].push(opt);
        });

        // Sort group names
        var groupNames = Object.keys(groups).sort();

        groupNames.forEach(function(gName) {
            var groupSection = document.createElement("div");
            groupSection.className = "option-group-section";
            
            var header = document.createElement("div");
            header.className = "option-group-header";
            header.textContent = gName;
            groupSection.appendChild(header);

            groups[gName].forEach(function (opt) {
                var el = buildOptionElement(opt);
                groupSection.appendChild(el);
            });

            container.appendChild(groupSection);
        });
    }

    /**
     * Build a DOM element for a single option.
     */
    function buildOptionElement(opt) {
        var wrapper = document.createElement("div");
        wrapper.className = "option-row";
        wrapper.dataset.optionId = opt.id;

        if (opt.visible_when) {
            wrapper.dataset.visibleWhen = JSON.stringify(opt.visible_when);
        }

        var labelEl = document.createElement("label");
        labelEl.setAttribute("for", "opt-" + opt.id);
        labelEl.className = "option-label";
        labelEl.textContent = opt.label;

        if (opt.help) {
            labelEl.title = opt.help;
        }

        var inputContainer = document.createElement("div");
        inputContainer.className = "option-input-wrap";

        var inputEl;
        switch (opt.type) {
            case "select":
                inputEl = renderSelect(opt);
                break;
            case "number":
                inputEl = renderNumber(opt);
                break;
            case "text":
                inputEl = renderText(opt);
                break;
            case "checkbox":
                inputEl = renderCheckbox(opt);
                break;
            default:
                inputEl = document.createElement("span");
                inputEl.textContent = "Unknown type: " + opt.type;
        }

        inputEl.id = "opt-" + opt.id;
        inputEl.name = opt.id;

        // Restore in-session edits from sessionStorage (cleared on page load)
        var saved = restoreState();
        if (saved.hasOwnProperty(opt.id)) {
            applyValue(inputEl, opt, saved[opt.id]);
        }

        inputContainer.appendChild(inputEl);

        // Inject BlockHelper for LoRA weighting
        if (opt.id === "tuning.block_weighting" && window.BlockHelper) {
            window.BlockHelper.inject(inputEl);
        }
        
        // Help text - put after input for checkboxes to appear below
        if (opt.help) {
            var helpEl = document.createElement("span");
            helpEl.className = "option-help";
            helpEl.textContent = opt.help;
            inputContainer.appendChild(helpEl);
        }
        
        wrapper.appendChild(labelEl);
        wrapper.appendChild(inputContainer);

        // Sub-options container
        if (opt.sub && opt.sub.length > 0) {
            var subContainer = document.createElement("div");
            subContainer.className = "option-sub";
            subContainer.dataset.parentId = opt.id;
            renderOptionTree(opt.sub, subContainer);
            wrapper.appendChild(subContainer);
        }

        // Bind change event
        inputEl.addEventListener("change", function () {
            updateFormState();
            applyVisibility();
            saveState();
        });

        // Initialize form state
        formState[opt.id] = getInputValue(inputEl, opt);

        return wrapper;
    }

    // Type renderers
    function renderSelect(opt) {
        var sel = document.createElement("select");
        (opt.choices || []).forEach(function (c) {
            var o = document.createElement("option");
            o.value = c.value;
            o.textContent = c.label;
            if (c.value === opt.default) o.selected = true;
            sel.appendChild(o);
        });
        return sel;
    }

    function renderNumber(opt) {
        var inp = document.createElement("input");
        inp.type = "number";
        inp.value = opt.default != null ? opt.default : "";
        if (opt.min != null) inp.min = opt.min;
        if (opt.max != null) inp.max = opt.max;
        if (opt.step != null) inp.step = opt.step;
        if (opt.placeholder) inp.placeholder = opt.placeholder;
        return inp;
    }

    function renderText(opt) {
        var inp = document.createElement("input");
        inp.type = "text";
        inp.value = opt.default != null ? opt.default : "";
        if (opt.placeholder) inp.placeholder = opt.placeholder;
        return inp;
    }

    function renderCheckbox(opt) {
        var inp = document.createElement("input");
        inp.type = "checkbox";
        inp.checked = !!opt.default;
        inp.value = "true";
        return inp;
    }

    // Apply a saved value to an input element
    function applyValue(el, opt, value) {
        if (opt.type === "checkbox") {
            el.checked = !!value;
        } else if (opt.type === "select") {
            el.value = String(value);
            var found = false;
            for (var i = 0; i < el.options.length; i++) {
                if (el.options[i].value === String(value)) { found = true; break; }
            }
            if (!found) el.selectedIndex = 0;
        } else {
            el.value = value;
        }
    }

    // Form state management
    function getInputValue(el, opt) {
        if (opt.type === "checkbox") return el.checked;
        if (opt.type === "number") return el.value !== "" ? parseFloat(el.value) : (opt.default != null ? opt.default : null);
        return el.value;
    }

    function updateFormState() {
        var container = document.getElementById("option-tree");
        var rows = container.querySelectorAll(".option-row");
        rows.forEach(function (row) {
            // Skip hidden rows so they don't overwrite visible ones with the same ID
            if (row.style.display === "none") return;

            var id = row.dataset.optionId;
            var input = row.querySelector("select, input");
            if (!input) return;
            var opt = findOption(optionSchema, id);
            if (opt) {
                formState[id] = getInputValue(input, opt);
            }
        });
    }

    function findOption(options, id) {
        for (var i = 0; i < options.length; i++) {
            if (options[i].id === id) return options[i];
            if (options[i].sub) {
                var found = findOption(options[i].sub, id);
                if (found) return found;
            }
        }
        return null;
    }

    // Visibility evaluation
    function applyVisibility() {
        updateFormState();

        var container = document.getElementById("option-tree");
        var rows = container.querySelectorAll(".option-row");

        // First pass: hide/show individual rows
        rows.forEach(function (row) {
            var cond = row.dataset.visibleWhen;
            if (!cond) {
                row.style.display = "";
                return;
            }

            var rules = JSON.parse(cond);
            var visible = evaluateVisibilityRules(rules);
            row.style.display = visible ? "" : "none";

            // Recursively hide sub-options if parent is hidden
            if (!visible) {
                var sub = row.querySelector(".option-sub");
                if (sub) {
                    var subRows = sub.querySelectorAll(".option-row");
                    subRows.forEach(function (sr) { sr.style.display = "none"; });
                }
            }
        });

        // Second pass: hide group sections if all rows within them are hidden
        var groupSections = container.querySelectorAll(".option-group-section");
        groupSections.forEach(function(section) {
            var visibleRows = section.querySelectorAll('.option-row:not([style*="display: none"])');
            section.style.display = (visibleRows.length > 0) ? "" : "none";
        });
        
        // Third pass: handle visible subs specifically
        var visibleSubs = container.querySelectorAll(".option-sub");
        visibleSubs.forEach(function (sub) {
            var parentRow = sub.closest(".option-row");
            if (parentRow && parentRow.style.display !== "none") {
                sub.style.display = "";
                sub.querySelectorAll(".option-row").forEach(function (childRow) {
                    var childCond = childRow.dataset.visibleWhen;
                    if (childCond) {
                        var childRules = JSON.parse(childCond);
                        childRow.style.display = evaluateVisibilityRules(childRules) ? "" : "none";
                    }
                });
            }
        });
    }

    function evaluateVisibilityRules(rules) {
        if (rules.__any__) {
            var anyList = rules.__any__;
            for (var ai = 0; ai < anyList.length; ai++) {
                if (formState[anyList[ai]]) return true;
            }
            return false;
        }

        if (rules.__none__) {
            var noneList = rules.__none__;
            for (var ni = 0; ni < noneList.length; ni++) {
                if (formState[noneList[ni]]) return false;
            }
            return true;
        }

        for (var key in rules) {
            if (!rules.hasOwnProperty(key)) continue;
            var expected = rules[key];
            var actual = formState[key];

            if (expected === "__truthy__") {
                if (!actual) return false;
                continue;
            }

            if (Array.isArray(expected)) {
                if (expected.indexOf(actual) === -1) return false;
            } else {
                if (actual != expected) return false;
            }
        }
        return true;
    }

    // Collect form data for submission
    function collectFormData() {
        updateFormState();
        var fd = new FormData();

        var configPathInput = document.getElementById("cfg-file-path");
        var configPath = configPathInput ? configPathInput.value.trim() : "convert-cfg.toml";
        fd.append("config", configPath);

        var container = document.getElementById("option-tree");
        var rows = container.querySelectorAll(".option-row");
        rows.forEach(function (row) {
            var id = row.dataset.optionId;
            var input = row.querySelector("select, input");
            if (!input) return;
            var opt = findOption(optionSchema, id);
            if (!opt) return;

            var isHidden = row.style.display === "none";

            // Skip hidden non-checkbox fields to avoid submitting defaults
            if (isHidden && opt.type !== "checkbox") return;

            var val = getInputValue(input, opt);

            // Skip fields with no real value (nullable fields left empty, e.g. an
            // unset tuning.gate_train_low). Submitting a coerced placeholder (the
            // old behavior: empty -> 0) would silently turn "gating disabled" into
            // "gating enabled with a bogus 0..0 range" on every single save, and
            // there'd be no way to get back to actually disabled via this form.
            if (val === null) return;

            // For checkboxes, always submit value (even hidden ones)
            // If checkbox is hidden and unchecked, submit "false"
            if (opt.type === "checkbox") {
                fd.append(id, val ? "true" : "false");
            } else {
                fd.append(id, val);
            }
        });

        return fd;
    }

    function saveToServer() {
        var btn = document.getElementById("btn-save-global");
        if (btn) btn.disabled = true;
        
        window.logToConsole("Saving configuration...", "info");
        
        var fd = collectFormData();
        fetch("/api/config", {
            method: "PUT",
            body: fd
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.error) {
                window.logToConsole("Save failed: " + data.error, "error");
            } else {
                window.logToConsole("Configuration saved successfully.", "success");
                // If we are in the config tab, we might want to refresh the raw editor too
                if (window.ConfigEditor && typeof window.ConfigEditor.load === "function") {
                    window.ConfigEditor.load();
                }
            }
        })
        .catch(function(err) {
            window.logToConsole("Network error during save: " + err, "error");
        })
        .finally(function() {
            if (btn) btn.disabled = false;
        });
    }

    // Public API
    window.OptionTree = {
        load: loadOptionSchema,
        saveState: saveState,
        collectFormData: collectFormData,
        saveToServer: saveToServer
    };
})();
