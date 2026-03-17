"""
Parser for TEI-XML encoded plays (Folger Digital Texts / DraCor format).

Folger TEI structure:
  <TEI>
    <text>
      <body>
        <div type="act" n="1">
          <div type="scene" n="1">
            <sp who="#ham">
              <speaker>HAMLET</speaker>
              <ab>To be or not to be...</ab>
            </sp>
            <stage>He draws his sword.</stage>
          </div>
        </div>
      </body>
    </text>
  </TEI>

Usage:
    from ingest.tei import parse_tei_play
    play = parse_tei_play("hamlet", xml_bytes)

Testing:
    After modifying this parser, run the regression test to verify all plays
    still parse correctly:

        conda run -n uta_model python scripts/regression_test_parsing.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Act, Beat, Play, Scene, Utterance

# TEI namespace map
_NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
}


def _text_content(el) -> str:
    """Recursively extract all text from an element, stripping markup."""
    return " ".join((el.text or "").split() +
                    [_text_content(child) for child in el] +
                    [(el.tail or "").split() and el.tail.split() or []])


def _make_id(*parts) -> str:
    return "_".join(str(p).lower().replace(" ", "_") for p in parts)


def parse_tei_play(
    play_id: str,
    xml_source: bytes | str,
    title: str = "",
    author: str = "",
) -> Play:
    """
    Parse a TEI-XML play file into a structured Play object.

    Supports both Folger (tei: namespace) and unnamespaced TEI.
    Returns a Play with Acts → Scenes → provisional Beats.
    """
    if isinstance(xml_source, str):
        xml_source = xml_source.encode()

    root = etree.fromstring(xml_source)

    # Detect namespace
    ns_prefix = "tei:" if root.tag.startswith("{http://www.tei-c.org/ns/1.0}") else ""
    ns = _NS if ns_prefix else {}

    def find_all(node, path: str):
        if ns:
            return node.findall(path.replace("/", "/tei:"), ns)
        return node.findall(path)

    def find_text(node, path: str) -> str:
        el = node.find(path.replace("/", "/tei:") if ns else path, ns if ns else None)
        if el is not None:
            return " ".join("".join(el.itertext()).split())
        return ""

    # Extract title / author from header if not provided
    if not title:
        title = find_text(root, ".//titleStmt/title") or play_id
    if not author:
        author = find_text(root, ".//titleStmt/author") or ""

    play = Play(id=play_id, title=title, author=author)
    all_characters: set[str] = set()
    utterance_index = 0

    body = root.find(".//tei:body", _NS)
    if body is None:
        body = root.find(".//body")
    if body is None:
        raise ValueError("Could not find <body> in TEI document")

    # Find act divs
    act_divs = (body.findall(".//tei:div[@type='act']", _NS) or
                body.findall(".//div[@type='act']"))

    if not act_divs:
        # Some TEI files have no act divisions — treat whole body as act 1
        act_divs_synthetic = [body]
        act_nums = [1]
    else:
        act_nums = []
        for adiv in act_divs:
            n = adiv.get("n", "")
            try:
                act_nums.append(int(n))
            except ValueError:
                act_nums.append(len(act_nums) + 1)

    for act_div, act_num in zip(act_divs if act_divs else [body], act_nums):
        act_obj = Act(
            id=_make_id(play_id, act_num),
            play_id=play_id,
            number=act_num,
        )

        scene_divs = (act_div.findall("tei:div[@type='scene']", _NS) or
                      act_div.findall("div[@type='scene']"))
        if not scene_divs:
            scene_divs = [act_div]

        for scene_idx, scene_div in enumerate(scene_divs, start=1):
            scene_num = scene_idx
            n = scene_div.get("n", "")
            if n:
                try:
                    scene_num = int(n)
                except ValueError:
                    pass

            utterances: list[Utterance] = []

            sp_elements = (scene_div.findall("tei:sp", _NS) or
                           scene_div.findall("sp"))
            for sp in sp_elements:
                speaker_el = sp.find("tei:speaker", _NS)
                if speaker_el is None:
                    speaker_el = sp.find("speaker")
                speaker = ""
                if speaker_el is not None:
                    speaker = " ".join("".join(speaker_el.itertext()).split()).upper()
                    speaker = speaker.rstrip(".")

                if not speaker:
                    who = sp.get("who", "")
                    speaker = who.lstrip("#").upper() if who else "UNKNOWN"

                # Gather all text from <ab>, <l>, <p> elements
                text_parts = []
                for tag in ("ab", "l", "p", "lg"):
                    for el in (sp.findall(f"tei:{tag}", _NS) or sp.findall(tag)):
                        t = " ".join("".join(el.itertext()).split())
                        if t:
                            text_parts.append(t)

                text = " / ".join(text_parts) if text_parts else ""
                if not text:
                    # fallback: grab all text content except speaker element
                    bits = []
                    for child in sp:
                        tag_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                        if tag_local != "speaker" and tag_local != "stage":
                            t = " ".join("".join(child.itertext()).split())
                            if t:
                                bits.append(t)
                    text = " ".join(bits)

                if not text:
                    continue

                # Stage direction for this speech
                stage_el = sp.find("tei:stage", _NS)
                if stage_el is None:
                    stage_el = sp.find("stage")
                stage_text = " ".join("".join(stage_el.itertext()).split()) if stage_el is not None else None

                uid = _make_id(play_id, act_num, scene_num, f"u{utterance_index}")
                utt = Utterance(
                    id=uid,
                    play_id=play_id,
                    act=act_num,
                    scene=scene_num,
                    index=utterance_index,
                    speaker=speaker,
                    text=text,
                    stage_direction=stage_text,
                )
                utterances.append(utt)
                all_characters.add(speaker)
                utterance_index += 1

            # Provisional: one beat per scene
            beat = Beat(
                id=_make_id(play_id, act_num, scene_num, "b1"),
                play_id=play_id,
                act=act_num,
                scene=scene_num,
                index=1,
                utterances=utterances,
                characters_present=sorted({u.speaker for u in utterances}),
            )
            scene_obj = Scene(
                id=_make_id(play_id, act_num, scene_num),
                play_id=play_id,
                act=act_num,
                scene=scene_num,
                beats=[beat],
            )
            act_obj.scenes.append(scene_obj)

        play.acts.append(act_obj)

    play.characters = sorted(all_characters)
    return play
