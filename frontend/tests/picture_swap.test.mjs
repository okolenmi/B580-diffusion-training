/* PictureSwap: a burst of pictures costs one load of the newest, a failed load keeps the old picture. */
import test from "node:test";
import assert from "node:assert/strict";
import { PictureSwap } from "../js/board/picture_swap.js";

const gate = () => {
  const loads = [];
  const load = (url) => new Promise((resolve, reject) => loads.push({ url, resolve, reject }));
  return { loads, load };
};
const tick = () => new Promise((r) => setTimeout(r, 0));

test("pictures that arrive while one loads replace each other; only the newest is loaded next", async () => {
  const { loads, load } = gate();
  const img = { src: "" };
  const swap = new PictureSwap(img, { load });
  swap.show("a"); swap.show("b"); swap.show("c"); swap.show("d");
  assert.deepEqual(loads.map((l) => l.url), ["a"], "one load in flight");
  loads[0].resolve(); await tick();
  assert.equal(img.src, "a");
  assert.deepEqual(loads.map((l) => l.url), ["a", "d"], "b and c were never loaded");
  loads[1].resolve(); await tick();
  assert.equal(img.src, "d");
});

test("the same picture again is free, and a failed load keeps the old picture and the next one still comes", async () => {
  const { loads, load } = gate();
  const img = { src: "" };
  const swap = new PictureSwap(img, { load });
  swap.show("a"); loads[0].resolve(); await tick();
  swap.show("a");
  assert.equal(loads.length, 1, "already shown");
  swap.show("b"); loads[1].reject(new Error("broken")); await tick();
  assert.equal(img.src, "a", "the old picture stays");
  swap.show("c"); loads[2].resolve(); await tick();
  assert.equal(img.src, "c");
});

test("after destroy nothing is put on the element", async () => {
  const { loads, load } = gate();
  const img = { src: "" };
  const swap = new PictureSwap(img, { load });
  swap.show("a"); swap.destroy(); loads[0].resolve(); await tick();
  assert.equal(img.src, "");
  swap.show("b");
  assert.equal(loads.length, 1);
});
