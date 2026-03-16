import SceneView from "@/components/SceneView";

interface Props {
  playId: string;
  act: string;
  scene: string;
}

export default function ScenePageShell({ playId, act, scene }: Props) {
  return (
    <SceneView playId={playId} act={Number(act)} scene={Number(scene)} />
  );
}
