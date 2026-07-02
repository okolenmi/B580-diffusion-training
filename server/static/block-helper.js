/* ---------------------------------------------------------------------------
   Block Weighting Helper — generates LoRA block weighting string
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    var SDXL_BLOCKS = [
        "input_blocks.0", "input_blocks.1", "input_blocks.2", 
        "input_blocks.3", "input_blocks.4", "input_blocks.5",
        "input_blocks.6", "input_blocks.7", "input_blocks.8",
        "middle_block",
        "output_blocks.0", "output_blocks.1", "output_blocks.2",
        "output_blocks.3", "output_blocks.4", "output_blocks.5",
        "output_blocks.6", "output_blocks.7", "output_blocks.8",
        "time_embed", "label_emb", "out"
    ];

    function createHelper(targetInput) {
        var wrapper = document.createElement("div");
        wrapper.className = "block-helper";

        var header = document.createElement("div");
        header.className = "block-helper-header";
        header.innerHTML = '<span>Block Weighting Helper (SDXL)</span> <span class="toggle-icon">▼</span>';
        
        var body = document.createElement("div");
        body.className = "block-helper-body-wrapper";
        body.style.display = "none";

        var toolbar = document.createElement("div");
        toolbar.className = "block-helper-toolbar";
        
        var btnReset = document.createElement("button");
        btnReset.className = "btn btn-secondary btn-small";
        btnReset.textContent = "Reset All (1.0)";
        btnReset.onclick = function() {
            body.querySelectorAll("input").forEach(function(inp) { inp.value = 1.0; });
            updateTarget(targetInput, body);
        };

        var btnClear = document.createElement("button");
        btnClear.className = "btn btn-kill btn-small";
        btnClear.textContent = "Disable All (0.0)";
        btnClear.onclick = function() {
            body.querySelectorAll("input").forEach(function(inp) { inp.value = 0.0; });
            updateTarget(targetInput, body);
        };

        toolbar.appendChild(btnReset);
        toolbar.appendChild(btnClear);
        body.appendChild(toolbar);

        var grid = document.createElement("div");
        grid.className = "block-helper-grid";

        header.onclick = function() {
            var isHidden = body.style.display === "none";
            body.style.display = isHidden ? "block" : "none";
            header.querySelector(".toggle-icon").textContent = isHidden ? "▲" : "▼";
        };

        // Parse initial string
        var initialWeights = parseWeightString(targetInput.value);

        SDXL_BLOCKS.forEach(function(blockId) {
            var item = document.createElement("div");
            item.className = "block-item";
            
            var label = document.createElement("label");
            label.textContent = blockId;
            
            var input = document.createElement("input");
            input.type = "number";
            input.min = 0;
            input.max = 1;
            input.step = 0.1;
            input.value = initialWeights.hasOwnProperty(blockId) ? initialWeights[blockId] : 1.0;
            input.dataset.blockId = blockId;

            input.onchange = function() {
                updateTarget(targetInput, body);
            };

            item.appendChild(label);
            item.appendChild(input);
            grid.appendChild(item);
        });

        body.appendChild(grid);
        wrapper.appendChild(header);
        wrapper.appendChild(body);
        return wrapper;
    }

    function parseWeightString(str) {
        var weights = {};
        if (!str) return weights;
        str.split(",").forEach(function(part) {
            if (part.indexOf(":") !== -1) {
                var kv = part.split(":");
                var k = kv[0].trim();
                var v = parseFloat(kv[1].trim());
                if (!isNaN(v)) weights[k] = v;
            }
        });
        return weights;
    }

    function updateTarget(targetInput, body) {
        var parts = [];
        var inputs = body.querySelectorAll("input");
        inputs.forEach(function(inp) {
            var val = parseFloat(inp.value);
            if (isNaN(val)) val = 1.0;
            if (val !== 1.0) {
                parts.push(inp.dataset.blockId + ":" + val.toFixed(1));
            }
        });
        targetInput.value = parts.join(", ");
        
        // Trigger change event to sync with formState
        var event = new Event('change', { bubbles: true });
        targetInput.dispatchEvent(event);
    }

    window.BlockHelper = {
        inject: function(targetInput) {
            var helper = createHelper(targetInput);
            targetInput.parentNode.appendChild(helper);
        }
    };
})();
