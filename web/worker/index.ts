// Serves the SPA's data files from the public R2 bucket, same-origin.
//
// Static assets (the Vite build in dist/) are handled by the assets binding
// before this code runs; wrangler.toml routes only /state.json and
// /history.jsonl here (`run_worker_first`). Anything else that reaches this
// handler is a path with no asset, which the ASSETS binding resolves to
// index.html per `not_found_handling`.
//
// Serving state through the binding instead of the bucket's r2.dev URL is the
// point of this Worker: the browser never makes a cross-origin request, so the
// bucket's CORS allowlist — which silently broke the site when the GitHub org
// was renamed — is no longer in the request path at all.

interface Env {
  ASSETS: Fetcher;
  STATE_BUCKET: R2Bucket;
}

// Object key → content type. Only these keys are ever read; the bucket also
// holds notify-state.json and health-state.json, which are the inference
// Worker's private bookkeeping and stay unreachable from here.
const PUBLIC_OBJECTS: Record<string, string> = {
  "/state.json": "application/json; charset=utf-8",
  "/history.jsonl": "application/x-ndjson; charset=utf-8",
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url);
    const contentType = PUBLIC_OBJECTS[pathname];
    if (contentType === undefined) {
      return env.ASSETS.fetch(request);
    }
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method Not Allowed", {
        status: 405,
        headers: { Allow: "GET, HEAD" },
      });
    }

    const object = await env.STATE_BUCKET.get(pathname.slice(1));
    if (object === null) {
      return new Response("Not Found", { status: 404 });
    }

    const headers = new Headers({
      "Content-Type": contentType,
      // The SPA polls every 60s and the Worker rewrites state.json every 15
      // minutes; a cached copy is exactly the stale reading the page is
      // trying not to show.
      "Cache-Control": "no-store",
      ETag: object.httpEtag,
    });
    return new Response(request.method === "HEAD" ? null : object.body, { headers });
  },
} satisfies ExportedHandler<Env>;
