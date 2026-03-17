import VocabView from "@/components/VocabView";

interface Props { playId?: string; }

export default function VocabPageShell({ playId }: Props) {
  return <VocabView playId={playId} />;
}
