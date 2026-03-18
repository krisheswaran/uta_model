import type {
  Play,
  Beat,
  BeatState,
  Scene,
  CharacterBible,
  SceneBible,
  SmoothedPlay,
  SmoothedBeat,
} from "./types";

// ─── Data Fetching ────────────────────────────────────────────────────────────

export async function fetchPlayIndex(): Promise<{ plays: Array<{ id: string; title: string; author: string; characters: string[]; actCount: number }> }> {
  const res = await fetch("/data/index.json");
  if (!res.ok) throw new Error(`Failed to fetch play index: ${res.status}`);
  return res.json();
}

export async function fetchPlay(playId: string): Promise<Play> {
  const res = await fetch(`/data/bibles/${playId}_bibles.json`);
  if (!res.ok) throw new Error(`Failed to fetch play ${playId}: ${res.status}`);
  return res.json();
}

export async function fetchSmoothedPlay(playId: string): Promise<SmoothedPlay | null> {
  try {
    const res = await fetch(`/data/smoothed/${playId}.json`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ─── Utility: Beat Accessors ──────────────────────────────────────────────────

/** Flatten all beats across every act and scene of a play. */
export function getAllBeats(play: Play): Beat[] {
  return play.acts.flatMap((act) =>
    act.scenes.flatMap((scene) => scene.beats)
  );
}

/** Get all beats that involve a particular character (have a BeatState for them). */
export function getAllBeatsForCharacter(play: Play, character: string): Beat[] {
  return getAllBeats(play).filter((beat) =>
    beat.beat_states.some(
      (bs) => bs.character.toUpperCase() === character.toUpperCase()
    )
  );
}

/** Get the BeatState for a specific character within a beat (case-insensitive). */
export function getBeatStateForCharacter(
  beat: Beat,
  character: string
): BeatState | undefined {
  return beat.beat_states.find(
    (bs) => bs.character.toUpperCase() === character.toUpperCase()
  );
}

/** Get all scenes from a play (flattened). */
export function getAllScenesFromPlay(play: Play): Scene[] {
  return play.acts.flatMap((act) => act.scenes);
}

/** Get a specific scene by act + scene number. */
export function getScene(
  play: Play,
  act: number,
  scene: number
): Scene | undefined {
  return play.acts
    .find((a) => a.number === act)
    ?.scenes.find((s) => s.scene === scene);
}

/** Get scene bible for an act+scene. */
export function getSceneBible(
  play: Play,
  act: number,
  scene: number
): SceneBible | undefined {
  return play.scene_bibles.find((sb) => sb.act === act && sb.scene === scene);
}

/** Get character bible for a character (case-insensitive). */
export function getCharacterBible(
  play: Play,
  character: string
): CharacterBible | undefined {
  return play.character_bibles.find(
    (cb) => cb.character.toUpperCase() === character.toUpperCase()
  );
}

// ─── Affect Color Interpolation ───────────────────────────────────────────────

/** Map valence [-1,1] to a hex color string. */
export function valenceColor(valence: number): string {
  // Clamp
  const v = Math.max(-1, Math.min(1, valence));
  if (v >= 0) {
    // neutral (#49454F) → positive (#81c995)
    const t = v;
    const r = Math.round(0x49 + t * (0x81 - 0x49));
    const g = Math.round(0x45 + t * (0xc9 - 0x45));
    const b = Math.round(0x4f + t * (0x95 - 0x4f));
    return `rgb(${r},${g},${b})`;
  } else {
    // negative (#f28b82) → neutral (#49454F)
    const t = 1 + v; // 0 at v=-1, 1 at v=0
    const r = Math.round(0xf2 + t * (0x49 - 0xf2));
    const g = Math.round(0x8b + t * (0x45 - 0x8b));
    const b = Math.round(0x82 + t * (0x4f - 0x82));
    return `rgb(${r},${g},${b})`;
  }
}

/** Map a beat index [0..total-1] to a blue→orange color gradient. */
export function beatProgressColor(index: number, total: number): string {
  if (total <= 1) return "#8ab4f8";
  const t = index / (total - 1);
  // blue (#8ab4f8) → orange (#f9ab00)
  const r = Math.round(0x8a + t * (0xf9 - 0x8a));
  const g = Math.round(0xb4 + t * (0xab - 0xb4));
  const b = Math.round(0xf8 + t * (0x00 - 0xf8));
  return `rgb(${r},${g},${b})`;
}

// ─── Confidence Helpers ───────────────────────────────────────────────────────

export function confidenceClass(confidence: number): string {
  if (confidence >= 0.7) return "confidence-high";
  if (confidence >= 0.4) return "confidence-mid";
  return "confidence-low";
}

export function confidenceLabel(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

// ─── Scene Key ────────────────────────────────────────────────────────────────

export function sceneKey(act: number, scene: number): string {
  return `A${act}S${scene}`;
}

// ─── Factor Graph Helpers ────────────────────────────────────────────────────

export function getSmoothedBeatForCharacter(
  smoothedPlay: SmoothedPlay,
  character: string,
  beatId: string
): SmoothedBeat | undefined {
  const charData = smoothedPlay.characters[character] ?? smoothedPlay.characters[character.toUpperCase()];
  if (!charData) return undefined;
  return charData.beats.find(b => b.beat_id === beatId);
}
