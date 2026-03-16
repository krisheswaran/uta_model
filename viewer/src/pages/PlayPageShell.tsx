import PlayView from "@/components/PlayView";

interface Props {
  playId: string;
}

export default function PlayPageShell({ playId }: Props) {
  return <PlayView playId={playId} />;
}
