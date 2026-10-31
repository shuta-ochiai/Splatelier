# App animations

The two animations the app plays are rendered by the scripts here. The videos themselves are
checked in under `App/Splatelier/`, so you only need this to change them.

| Video | Script | Used by |
|---|---|---|
| `App/Splatelier/launch.mp4` (4.1 s, 1080×1920) | `render_launch.py` | `LaunchVideoView`, played after the OS launch screen; a tap skips it once the app is ready |
| `App/Splatelier/Guides/guide_*.mp4` (6–8 s loops, 1080×1080) | `render_guides.py` | `GuideClipView`, capture tips on the Capture screen: fixed-order sets of clips, a set picked at random, a tap pauses, a swipe to the left goes to the next clip and a swipe to the right to the previous one |

## Rebuild

```
pip install numpy pillow     # and have ffmpeg installed
scripts/media/build_media.sh [launch|guides|all]
```

`build_media.sh` fetches the fonts (`fetch_fonts.sh`, into the git-ignored `scripts/media/fonts/`),
renders the frames into a temporary directory and encodes them (H.264, no audio, BT.709). If you
add or remove a clip, also run `cd App && xcodegen generate`.

Each renderer also writes single frames for a quick look:

```
python3 scripts/media/render_launch.py /tmp/out --only 240          # frame 240 of the launch video
python3 scripts/media/render_guides.py /tmp/out slow --only 2.5     # the guide frame at 2.5 s
```

## Launch video

- The background is `#161B46`, the same color as `LaunchBackground` in the asset catalog, so the
  OS launch screen turns into the video without a visible jump. Change both together.
- `--slow` stretches time (default 1.4, which makes the brush slower), `--no-title` leaves the
  "Splatelier" wordmark out. The wordmark's size and position are `TITLE_SIZE` and `TITLE_CY`.

## Capture guides

Eight scenes: `slow`, `turn`, `angles`, `pivot`, `loop`, `light`, `subjects`, `cool` (the last one is about the phone, not the shot: close other apps, unplug the charger, take the case off, so training does not slow down from heat). The English title and
captions are drawn into the frames (the app has no Japanese UI, so they are not overlaid by
SwiftUI). To add a scene, add a `scene_*` function and register it in `SCENES`, put its name into
a set in `GuideSequence.sets` (`App/Splatelier/GuideClipView.swift`) and add it to the loop in `build_media.sh`.

## Fonts

Instrument Serif (the logo's typeface), Lora and Noto Sans JP, all SIL Open Font License 1.1. The
license texts are fetched next to the fonts. They are credited in the main README.
