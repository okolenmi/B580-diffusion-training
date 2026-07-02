/* ---------------------------------------------------------------------------
   Chart Manager — custom symlog scale: linear center, logarithmic edges
   Pure canvas implementation with hover tooltips
   --------------------------------------------------------------------------- */

(function () {
    "use strict";

    var lossData = [];  /* {step, loss, avg_loss} */
    var canvas, ctx;
    var animFrame = null;
    var dpr = 1;
    var W = 0, H = 0;  /* CSS dimensions */

    /* Layout */
    var margin = { top: 24, right: 20, bottom: 40, left: 80 };
    var plotL, plotR, plotT, plotB, plotW, plotH;

    /* Scale state */
    var linMin, linMax, fullMin, fullMax;
    var xMin, xMax;
    var lastMouse = null;
    var tooltipInfo = null;

    /* ───────────────────────────────────────────────────────────────────
       Helpers
       ─────────────────────────────────────────────────────────────────── */

    function niceNum(range, round) {
        if (range <= 0) return 1;
        var exp = Math.floor(Math.log10(range));
        var frac = range / Math.pow(10, exp);
        var nice;
        if (round) {
            nice = (frac < 1.5) ? 1 : (frac < 3) ? 2 : (frac < 7) ? 5 : 10;
        } else {
            nice = (frac <= 1) ? 1 : (frac <= 2) ? 2 : (frac <= 5) ? 5 : 10;
        }
        return nice * Math.pow(10, exp);
    }

    function formatNum(v) {
        if (v >= 1) return v.toFixed(4);
        if (v >= 0.01) return v.toFixed(5);
        return v.toExponential(2);
    }

    /* ───────────────────────────────────────────────────────────────────
       Symmetric-Log scale
       ─────────────────────────────────────────────────────────────────── */

    function symMap(value) {
        /* Guard against degenerate ranges */
        if (fullMax <= linMax) fullMax = linMax * 2;
        if (fullMin >= linMin) fullMin = linMin * 0.5;
        if (fullMin <= 0) fullMin = linMin * 0.5;
        if (linMax <= linMin) linMax = linMin * 1.01;

        if (value >= linMax) {
            var topH = plotH * 0.25;
            var t = Math.log(value / linMax) / Math.log(fullMax / linMax);
            t = Math.max(0, Math.min(1, t));
            return plotT + topH * (1 - t);
        } else if (value <= linMin) {
            var botH = plotH * 0.25;
            var t = Math.log(value / linMin) / Math.log(fullMin / linMin);
            t = Math.max(0, Math.min(1, t));
            /* t=0 (linMin) maps to 75% height, t=1 (fullMin) maps to 100% height */
            return (plotB - botH) + botH * t;
        } else {
            var centerH = plotH * 0.5;
            var frac = (value - linMin) / (linMax - linMin);
            frac = Math.max(0, Math.min(1, frac));
            return plotT + plotH * 0.25 + centerH * (1 - frac);
        }
    }

    function xPos(step) {
        if (xMax === xMin) return plotL + plotW / 2;
        return plotL + ((step - xMin) / (xMax - xMin)) * plotW;
    }

    /* ───────────────────────────────────────────────────────────────────
       Tick building (throttled to avoid overlap)
       ─────────────────────────────────────────────────────────────────── */

    function buildTicks() {
        var ticks = [];
        var maxTicks = 12;

        /* Linear band ticks — max ~6 */
        if (linMax > linMin) {
            var range = linMax - linMin;
            var step = niceNum(range / 4, true);
            if (step <= 0) step = range / 4;
            var start = Math.ceil(linMin / step) * step;
            for (var v = start; v <= linMax + step * 0.01; v += step) {
                if (v >= linMin - step * 0.01 && v <= linMax + step * 0.01) {
                    ticks.push(v);
                }
            }
        }

        /* Log ticks — one per decade per edge (2-4 total) */
        if (fullMin < linMin && linMin > 0 && fullMin > 0) {
            var loMin = Math.log10(fullMin);
            var loLMin = Math.log10(linMin);
            /* Only 1-2 tick values per decade band */
            for (var p = Math.floor(loMin); p <= Math.ceil(loLMin); p++) {
                var base = Math.pow(10, p);
                /* Only 2 and 5 for compactness */
                [2, 5].forEach(function (m) {
                    var val = m * base;
                    if (val >= fullMin * 0.99 && val < linMin * 0.99) {
                        ticks.push(val);
                    }
                });
            }
        }

        if (fullMax > linMax && linMax > 0) {
            var loLMax = Math.log10(linMax);
            var loMax = Math.log10(fullMax);
            for (var pp = Math.floor(loLMax); pp <= Math.ceil(loMax); pp++) {
                var b = Math.pow(10, pp);
                [2, 5].forEach(function (mm) {
                    var va = mm * b;
                    if (va > linMax * 1.01 && va <= fullMax * 1.01) {
                        ticks.push(va);
                    }
                });
            }
        }

        ticks.sort(function (a, b) { return a - b; });

        /* Deduplicate very close values */
        var unique = [];
        for (var i = 0; i < ticks.length; i++) {
            if (i === 0 || ticks[i] > unique[unique.length - 1] * 1.2) {
                unique.push(ticks[i]);
            }
        }

        /* Cap total */
        return unique.slice(0, maxTicks);
    }

    /* ───────────────────────────────────────────────────────────────────
       Drawing
       ─────────────────────────────────────────────────────────────────── */

    function draw() {
        if (!canvas || !ctx) return;

        // Use OffsetWidth/Height to get the container's layout size
        var rect = canvas.getBoundingClientRect();
        W = rect.width;
        H = rect.height;
        
        if (W <= 0 || H <= 0) return;

        canvas.width = W * dpr;
        canvas.height = H * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        plotL = margin.left;
        plotR = W - margin.right;
        plotT = margin.top;
        plotB = H - margin.bottom;
        plotW = plotR - plotL;
        plotH = plotB - plotT;

        ctx.clearRect(0, 0, W, H);

        var sliced = lossData.slice(-500);
        if (sliced.length === 0) {
            ctx.fillStyle = "#888899";
            ctx.font = "14px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("No data yet", W / 2, H / 2);
            tooltipInfo = null;
            return;
        }

        /* Compute scale */
        var avgVals = [], lossVals = [];
        for (var i = 0; i < sliced.length; i++) {
            if (sliced[i].avg_loss != null && sliced[i].avg_loss > 0) avgVals.push(sliced[i].avg_loss);
            if (sliced[i].loss != null && sliced[i].loss > 0) lossVals.push(sliced[i].loss);
        }

        if (avgVals.length >= 2) {
            var aMin = Infinity, aMax = 0;
            for (var j = 0; j < avgVals.length; j++) {
                if (avgVals[j] < aMin) aMin = avgVals[j];
                if (avgVals[j] > aMax) aMax = avgVals[j];
            }
            /* Add 10% padding to the linear range so the avg line stays centered */
            var aRange = aMax - aMin;
            if (aRange <= 0) aRange = aMin * 0.2;
            linMin = aMin - aRange * 0.1;
            linMax = aMax + aRange * 0.1;
            if (linMin <= 0) linMin = aMin * 0.5;

            fullMin = linMin * 0.5;
            fullMax = linMax * 2.0;
            if (lossVals.length > 0) {
                var lMax = Math.max.apply(null, lossVals);
                var lMin = Math.min.apply(null, lossVals);
                if (lMax > fullMax) fullMax = lMax * 1.1;
                if (lMin < fullMin && lMin > 0) fullMin = lMin * 0.5;
            }
            if (fullMax <= linMax) fullMax = linMax * 2;
            if (fullMin >= linMin) fullMin = linMin * 0.5;
            if (fullMin <= 0) fullMin = linMin * 0.5;
        } else if (avgVals.length === 1) {
            var v = avgVals[0];
            linMin = v * 0.9; linMax = v * 1.1;
            fullMin = v * 0.2; fullMax = v * 5;
        } else if (lossVals.length > 0) {
            fullMin = 0;
            fullMax = Math.max.apply(null, lossVals) * 1.2;
            linMin = fullMax * 0.01;
            linMax = fullMax * 0.99;
        } else {
            fullMin = 0; fullMax = 1; linMin = 0.1; linMax = 0.9;
        }

        xMin = sliced[0].step;
        xMax = sliced[sliced.length - 1].step;

        var ticks = buildTicks();

        /* ── Grid lines & Y labels ── */
        ctx.strokeStyle = "#2a2d3a";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#888899";
        ctx.font = "10px monospace";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";

        for (var t = 0; t < ticks.length; t++) {
            var py = symMap(ticks[t]);
            if (py >= plotT + 5 && py <= plotB - 5) {
                ctx.beginPath();
                ctx.moveTo(plotL, py);
                ctx.lineTo(plotR, py);
                ctx.stroke();
                ctx.fillText(formatNum(ticks[t]), plotL - 6, py);
            }
        }

        /* Linear band separator lines */
        ctx.strokeStyle = "#3a3d4a";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        var yLinTop = symMap(linMax);
        var yLinBot = symMap(linMin);
        if (yLinTop > plotT && yLinTop < plotB) {
            ctx.beginPath(); ctx.moveTo(plotL, yLinTop); ctx.lineTo(plotR, yLinTop); ctx.stroke();
        }
        if (yLinBot > plotT && yLinBot < plotB) {
            ctx.beginPath(); ctx.moveTo(plotL, yLinBot); ctx.lineTo(plotR, yLinBot); ctx.stroke();
        }
        ctx.setLineDash([]);

        /* ── X axis ── */
        ctx.strokeStyle = "#2a2d3a";
        ctx.beginPath();
        ctx.moveTo(plotL, plotB);
        ctx.lineTo(plotR, plotB);
        ctx.stroke();

        var xStep = niceNum((xMax - xMin) / 8, true);
        if (xStep <= 0) xStep = 1;
        var xStart = Math.ceil(xMin / xStep) * xStep;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        for (var xs = xStart; xs <= xMax + xStep * 0.01; xs += xStep) {
            var px = xPos(xs);
            if (px >= plotL && px <= plotR) {
                ctx.fillText(Math.round(xs).toString(), px, plotB + 6);
            }
        }

        /* ── Loss dots ── */
        for (var di = 0; di < sliced.length; di++) {
            if (sliced[di].loss != null) {
                var dx = xPos(sliced[di].step);
                var dy = symMap(sliced[di].loss);
                if (dy >= plotT - 3 && dy <= plotB + 3) {
                    ctx.fillStyle = "rgba(108,140,255,0.6)";
                    ctx.beginPath();
                    ctx.arc(dx, dy, 1.5, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }

        /* ── Avg line (skip null/zero values) ── */
        ctx.strokeStyle = "#4caf50";
        ctx.lineWidth = 2;
        ctx.beginPath();
        var firstPt = true;
        var lastPy = null;
        for (var ai = 0; ai < sliced.length; ai++) {
            if (sliced[ai].avg_loss == null || sliced[ai].avg_loss <= 0) {
                firstPt = true;
                lastPy = null;
                continue;
            }
            var ax = xPos(sliced[ai].step);
            var ay = symMap(sliced[ai].avg_loss);
            if (firstPt || lastPy == null) {
                ctx.moveTo(ax, ay);
                firstPt = false;
            } else {
                ctx.lineTo(ax, ay);
            }
            lastPy = ay;
        }
        ctx.stroke();

        /* ── Legend ── */
        ctx.fillStyle = "#6c8cff";
        ctx.beginPath();
        ctx.arc(plotL + 6, plotT - 8, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#888899";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("Loss", plotL + 14, plotT - 8);

        ctx.strokeStyle = "#4caf50";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(plotL, plotT + 8);
        ctx.lineTo(plotL + 12, plotT + 8);
        ctx.stroke();
        ctx.fillText("Avg", plotL + 16, plotT + 8);

        /* ── Tooltip ── */
        drawTooltip(sliced);
    }

    function drawTooltip(sliced) {
        if (!lastMouse || !tooltipInfo) return;

        var mx = lastMouse.x, my = lastMouse.y;
        var info = tooltipInfo;

        /* Tooltip box */
        ctx.font = "11px monospace";
        var lines = [
            "Step: " + info.step,
            "Loss: " + formatNum(info.loss),
        ];
        if (info.avg != null && info.avg > 0) lines.push("Avg:  " + formatNum(info.avg));

        var tw = 0;
        for (var i = 0; i < lines.length; i++) {
            var m = ctx.measureText(lines[i]);
            if (m.width > tw) tw = m.width;
        }
        var th = lines.length * 16 + 10;

        var tx = info.px + 12;
        var ty = info.lossY - th / 2;
        if (tx + tw + 16 > W) tx = info.px - tw - 20;
        if (ty < 0) ty = 4;
        if (ty + th > H) ty = H - th - 4;

        ctx.fillStyle = "rgba(20,20,40,0.92)";
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(tx, ty, tw + 16, th, 6);
        } else {
            ctx.rect(tx, ty, tw + 16, th);
        }
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#ccc";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        for (var li = 0; li < lines.length; li++) {
            ctx.fillText(lines[li], tx + 8, ty + 6 + li * 16);
        }

        /* Crosshair */
        ctx.strokeStyle = "rgba(200,200,200,0.3)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(info.px, plotT);
        ctx.lineTo(info.px, plotB);
        ctx.stroke();
        ctx.setLineDash([]);

        /* Highlight dot */
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(info.px, info.lossY, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#6c8cff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    /* ───────────────────────────────────────────────────────────────────
       Hover handling
       ─────────────────────────────────────────────────────────────────── */

    function findHoverPoint(sliced) {
        if (!lastMouse) return null;
        var mx = lastMouse.x;
        if (mx < plotL || mx > plotR) return null;

        /* Find nearest point by pixel x */
        var best = null, bestDist = Infinity;
        for (var i = 0; i < sliced.length; i++) {
            var px = xPos(sliced[i].step);
            var dist = Math.abs(px - mx);
            if (dist < bestDist) {
                bestDist = dist;
                best = { i: i, data: sliced[i], px: px };
            }
        }

        if (best && bestDist < 30 && best.data.loss != null) {
            var ly = symMap(best.data.loss);
            return {
                step: best.data.step,
                loss: best.data.loss,
                avg: best.data.avg_loss,
                px: best.px,
                lossY: ly,
            };
        }
        return null;
    }

    /* ───────────────────────────────────────────────────────────────────
       Public API
       ─────────────────────────────────────────────────────────────────── */

    function initChart() {
        canvas = document.getElementById("loss-chart");
        if (!canvas) return;
        ctx = canvas.getContext("2d");
        dpr = window.devicePixelRatio || 1;

        canvas.addEventListener("mousemove", function (e) {
            var rect = canvas.getBoundingClientRect();
            lastMouse = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            var sliced = lossData.slice(-500);
            tooltipInfo = findHoverPoint(sliced);
            requestDraw();
        });

        canvas.addEventListener("mouseleave", function () {
            lastMouse = null;
            tooltipInfo = null;
            requestDraw();
        });

        var ro = new ResizeObserver(function () { requestDraw(); });
        ro.observe(canvas);
    }

    function addDataPoint(step, loss, avgLoss) {
        var hasAvg = lossData.length >= 24;
        lossData.push({ step: step, loss: loss, avg_loss: hasAvg ? avgLoss : null });
        var seen = {};
        lossData = lossData.filter(function (d) {
            if (seen[d.step]) return false;
            seen[d.step] = true;
            return true;
        });
        requestDraw();
    }

    function requestDraw() {
        if (animFrame) return;
        animFrame = requestAnimationFrame(function () {
            draw();
            animFrame = null;
        });
    }

    function reset() {
        lossData = [];
        tooltipInfo = null;
        lastMouse = null;
        draw();
    }

    function update() {
        draw();
    }

    window.ChartManager = {
        init: initChart,
        update: update,
        addPoint: addDataPoint,
        reset: reset,
    };
})();
