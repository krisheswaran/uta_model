import CharacterView from "@/components/CharacterView";

interface Props {
  playId: string;
  character: string;
}

export default function CharacterPageShell({ playId, character }: Props) {
  return <CharacterView playId={playId} character={character} />;
}
