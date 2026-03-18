import { render, route } from "rwsdk/router";
import { defineApp } from "rwsdk/worker";
import { Document } from "@/app/Document";
import HomePageShell from "@/pages/HomePageShell";
import PlayPageShell from "@/pages/PlayPageShell";
import CharacterPageShell from "@/pages/CharacterPageShell";
import ScenePageShell from "@/pages/ScenePageShell";
import VocabPageShell from "@/pages/VocabPageShell";
import ImprovListPageShell from "@/pages/ImprovListPageShell";
import ImprovSessionPageShell from "@/pages/ImprovSessionPageShell";

// ─── R2 data route ────────────────────────────────────────────────────────────
// Serves /data/* from the R2 bucket in production.
// In local dev, Vite serves from public/data/ so this route is never hit.

interface Env {
  DATA_BUCKET?: R2Bucket;
}

const ALLOWED_PREFIXES = ["bibles/", "beats/", "vocab/", "smoothed/", "improv/", "index.json", "improv-index.json"];

function serveFromR2(request: Request, env: Env): Response | Promise<Response> {
  const url = new URL(request.url);
  const key = url.pathname.replace(/^\/data\//, ""); // strip leading /data/

  if (!env.DATA_BUCKET) {
    // No R2 binding (local dev) — let Vite handle it
    return new Response("Not found", { status: 404 });
  }

  // Only serve from known data prefixes — prevents accidental exposure of
  // anything else that might end up in the bucket
  if (!ALLOWED_PREFIXES.some((p) => key === p || key.startsWith(p))) {
    return new Response("Forbidden", { status: 403 });
  }

  return env.DATA_BUCKET.get(key).then((object) => {
    if (!object) {
      return new Response(`Not found: ${key}`, { status: 404 });
    }

    const headers = new Headers();
    headers.set("Content-Type", "application/json");
    headers.set("Cache-Control", "public, max-age=3600");
    object.writeHttpMetadata(headers);

    return new Response(object.body as ReadableStream, { headers });
  });
}

// ─── App routes ───────────────────────────────────────────────────────────────

export default defineApp([
  // R2 data route — must come before render() so it intercepts /data/* requests
  route("/data/*", ({ request, env }) => serveFromR2(request, env as unknown as Env)),

  render(Document, [
    route("/", () => <HomePageShell />),
    route("/plays/:playId", ({ params }) => (
      <PlayPageShell playId={params.playId} />
    )),
    route("/plays/:playId/characters/:character", ({ params }) => (
      <CharacterPageShell playId={params.playId} character={params.character} />
    )),
    route("/plays/:playId/scenes/:act/:scene", ({ params }) => (
      <ScenePageShell playId={params.playId} act={params.act} scene={params.scene} />
    )),
    route("/vocab", () => <VocabPageShell />),
    route("/vocab/:playId", ({ params }) => <VocabPageShell playId={params.playId} />),
    route("/improv", () => <ImprovListPageShell />),
    route("/improv/:sessionId", ({ params }) => (
      <ImprovSessionPageShell sessionId={params.sessionId} />
    )),
  ]),
]);
