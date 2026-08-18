import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the VibraLens maintenance workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>VibraLens — Bearing Life Intelligence<\/title>/i);
  assert.match(html, /Know what the/);
  assert.match(html, /Snapshot input/);
  assert.match(html, /Life estimate/);
  assert.match(html, /Run life estimate/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
  assert.match(html, /property="og:image"/i);
  assert.match(html, /http:\/\/localhost:3000\/og\.png/i);
});

test("ships product-specific source and social preview", async () => {
  const [page, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    access(new URL("../public/og.png", import.meta.url)),
  ]);

  assert.match(page, /NEXT_PUBLIC_VIBRALENS_API_URL/);
  assert.match(page, /planned_break_minutes/);
  assert.match(layout, /VibraLens — Bearing Life Intelligence/);
  assert.match(layout, /\/og\.png/);
});
