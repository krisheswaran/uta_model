import ImprovSessionView from "@/components/ImprovSessionView";

interface Props { sessionId: string; }

export default function ImprovSessionPageShell({ sessionId }: Props) {
  return <ImprovSessionView sessionId={sessionId} />;
}
