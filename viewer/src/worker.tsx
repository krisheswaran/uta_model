import { render, route } from "rwsdk/router";
import { defineApp } from "rwsdk/worker";
import { Document } from "@/app/Document";
import HomePageShell from "@/pages/HomePageShell";
import PlayPageShell from "@/pages/PlayPageShell";
import CharacterPageShell from "@/pages/CharacterPageShell";
import ScenePageShell from "@/pages/ScenePageShell";

export default defineApp([
  render(Document, [
    route("/", () => <HomePageShell />),
    route("/plays/:playId", ({ params }) => (
      <PlayPageShell playId={params.playId} />
    )),
    route("/plays/:playId/characters/:character", ({ params }) => (
      <CharacterPageShell
        playId={params.playId}
        character={params.character}
      />
    )),
    route("/plays/:playId/scenes/:act/:scene", ({ params }) => (
      <ScenePageShell
        playId={params.playId}
        act={params.act}
        scene={params.scene}
      />
    )),
  ]),
]);
