#!/usr/bin/env node
/**
 * sync-data.mjs
 *
 * Copies `*_bibles.json` files from `../data/bibles/` into `public/data/bibles/`
 * and generates `public/data/index.json` with play metadata.
 *
 * Run with: node scripts/sync-data.mjs
 * Or via: npm run sync
 */

import { readdir, readFile, copyFile, mkdir, writeFile } from "fs/promises";
import { existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolve paths relative to the viewer/ directory
const VIEWER_DIR = join(__dirname, "..");
const SOURCE_DIR = join(VIEWER_DIR, "..", "data", "bibles");
const DEST_DIR = join(VIEWER_DIR, "public", "data", "bibles");
const BEATS_SOURCE_DIR = join(VIEWER_DIR, "..", "data", "beats");
const BEATS_DEST_DIR = join(VIEWER_DIR, "public", "data", "beats");
const INDEX_PATH = join(VIEWER_DIR, "public", "data", "index.json");
const CONFIG_PY = join(VIEWER_DIR, "..", "config.py");

// ─── Helpers ──────────────────────────────────────────────────────────────────

function log(msg) {
  process.stdout.write(`  ${msg}\n`);
}

function ok(msg) {
  process.stdout.write(`✓ ${msg}\n`);
}

function warn(msg) {
  process.stderr.write(`⚠ ${msg}\n`);
}

/**
 * Parse the PLAYS dict from config.py using a simple regex approach.
 * Returns { playId: { title, author } }.
 */
async function parsePlaysFromConfig() {
  const plays = {};

  if (!existsSync(CONFIG_PY)) {
    warn(`config.py not found at ${CONFIG_PY} — will use play_id as title`);
    return plays;
  }

  const content = await readFile(CONFIG_PY, "utf-8");

  // Match: "cherry_orchard": { ... "title": "...", "author": "..." ... }
  // We use a multi-pass regex approach for robustness.
  const playBlockRe = /"([a-z_]+)"\s*:\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}/g;
  let blockMatch;

  while ((blockMatch = playBlockRe.exec(content)) !== null) {
    const playId = blockMatch[1];
    const block = blockMatch[2];

    const titleMatch = block.match(/"title"\s*:\s*"([^"]+)"/);
    const authorMatch = block.match(/"author"\s*:\s*"([^"]+)"/);

    if (titleMatch) {
      plays[playId] = {
        title: titleMatch[1],
        author: authorMatch ? authorMatch[1] : "Unknown",
      };
    }
  }

  return plays;
}

/**
 * Extract play metadata from a bibles JSON file.
 * Returns a PlayIndexEntry object.
 */
function extractMetadata(playData, configMeta) {
  const id = playData.id;
  const title = playData.title ?? configMeta?.title ?? id;
  const author = playData.author ?? configMeta?.author ?? "Unknown";
  const characters = playData.characters ?? [];
  const actCount = Array.isArray(playData.acts) ? playData.acts.length : 0;

  return { id, title, author, characters, actCount };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log("\n── UTA Viewer: Data Sync ──────────────────────────────────\n");

  // Ensure source dir exists
  if (!existsSync(SOURCE_DIR)) {
    warn(`Source directory not found: ${SOURCE_DIR}`);
    warn("Run the Python pipeline first to generate bibles JSON files.");
    process.exit(1);
  }

  // Ensure dest dirs exist
  await mkdir(DEST_DIR, { recursive: true });
  ok(`Destination: ${DEST_DIR}`);

  // Parse config.py for metadata
  log("Parsing config.py for play metadata...");
  const configMeta = await parsePlaysFromConfig();
  const configKeys = Object.keys(configMeta);
  log(`Found ${configKeys.length} plays in config.py: ${configKeys.join(", ") || "(none)"}`);

  // Scan source directory
  let files;
  try {
    files = await readdir(SOURCE_DIR);
  } catch (err) {
    warn(`Cannot read source directory: ${err.message}`);
    process.exit(1);
  }

  const bibleFiles = files.filter((f) => f.endsWith("_bibles.json"));

  if (bibleFiles.length === 0) {
    warn(`No *_bibles.json files found in ${SOURCE_DIR}`);
    warn("Run the pipeline to generate play data first.");
    // Write empty index
    await writeFile(INDEX_PATH, JSON.stringify({ plays: [] }, null, 2), "utf-8");
    ok("Wrote empty public/data/index.json");
    return;
  }

  log(`Found ${bibleFiles.length} bible file(s): ${bibleFiles.join(", ")}`);
  console.log();

  // Process each file
  const indexEntries = [];

  for (const filename of bibleFiles) {
    const srcPath = join(SOURCE_DIR, filename);
    const destPath = join(DEST_DIR, filename);

    // Copy file
    await copyFile(srcPath, destPath);
    ok(`Copied ${filename}`);

    // Parse for index
    try {
      const raw = await readFile(srcPath, "utf-8");
      const playData = JSON.parse(raw);
      const playId = playData.id ?? filename.replace("_bibles.json", "");
      const meta = extractMetadata(playData, configMeta[playId]);
      indexEntries.push(meta);
      log(
        `  → "${meta.title}" by ${meta.author} | ${meta.characters.length} characters, ${meta.actCount} act(s)`
      );
    } catch (parseErr) {
      warn(`  Could not parse ${filename}: ${parseErr.message}`);
    }
  }

  // Write index.json
  const index = { plays: indexEntries };
  await writeFile(INDEX_PATH, JSON.stringify(index, null, 2), "utf-8");
  console.log();
  ok(`Wrote public/data/index.json (${indexEntries.length} play${indexEntries.length !== 1 ? "s" : ""})`);

  // ── Beats ──────────────────────────────────────────────────────────────────
  if (existsSync(BEATS_SOURCE_DIR)) {
    await mkdir(BEATS_DEST_DIR, { recursive: true });
    const beatFiles = (await readdir(BEATS_SOURCE_DIR)).filter((f) => f.endsWith("_beats.json"));
    for (const filename of beatFiles) {
      await copyFile(join(BEATS_SOURCE_DIR, filename), join(BEATS_DEST_DIR, filename));
      ok(`Copied beats/${filename}`);
    }
  } else {
    warn(`No beats directory found at ${BEATS_SOURCE_DIR} — skipping`);
  }

  console.log("\n── Done ──────────────────────────────────────────────────\n");
  console.log("  Start the viewer with: npm run dev\n");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
