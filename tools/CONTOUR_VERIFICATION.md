## Contour verification (ADB screenshots)

These steps verify the contour effect is **visible, natural**, and **never bleeds onto the background** near the ear/hairline.

### Preconditions

- A device/emulator is connected and authorized.
- The app is running and a makeup style with contour enabled is selected.

### 1) Confirm device connection

```bash
adb devices
```

### 2) Capture required screenshots

From the repo root:

```bash
tools/capture_contour_screens.sh 1 0.0 straight_on
tools/capture_contour_screens.sh 1 0.0 slight_left_turn
tools/capture_contour_screens.sh 1 0.0 slight_right_turn
tools/capture_contour_screens.sh 1 0.0 different_lighting
```

All images are saved into `captures_visible/`.

### 3) What to validate (pass/fail)

- **No background bleed**: contour must not darken background near the ear/hairline.
- **Placement**: cheekbones, jawline, nose, temples should be present but subtle.
- **Finish**: should preserve skin texture (no flat opaque “paint”).
- **Stability**: on slight head turns, contour follows without jitter/flicker.

### 4) How to tune (make it more/less obvious)

Most-impact controls (in order):

- **Global strength**: `MakeupStyle.contourAlpha` and `MakeupStyle.highlightAlpha`
  - Mapped in `ContourRenderer.computeStrengths()` (see `baseContour/baseHighlight` clamps).
  - If contour is too subtle overall, raise `contourAlpha` in the style first.

- **Per-region intensity defaults**: `ContourParams.cheekOpacity/jawOpacity/noseOpacity/foreheadOpacity`
  - Defaults (requirements): cheek=0.45, jaw=0.25, nose=0.35, forehead=0.20
  - If cheeks look like “blush”, lower `cheekOpacity` slightly and/or increase `coolTone`.

- **Stability vs lag**: `ContourParams.temporalSmoothing`
  - Higher (e.g. 0.70): steadier, more lag on fast motion
  - Lower (e.g. 0.45): more responsive, more jitter

- **Mask “thickness” / feather**: `app/src/main/res/raw/contour_mask_fragment.glsl`
  - Cheek ribbon thickness: `coreHalfWidth*` / `blendHalfWidth*` and `contourOffsetPx`
  - Nose visibility: nose block `alpha` and `halfW`
  - Jaw subtlety: jaw block `halfW`, `blurPx`, and final `contourBlend += ...`

Practical workflow:

- Use the in-app **Masks** debug button (cycles contour/highlight/beard masks) to confirm:
  - cheeks are strongest under cheekbone and stop before mouth corners
  - jaw mask stays under jaw and fades into neck
  - nose mask is two slim rails, not a blob
  - exclusions keep eyes/lips/brows/nostrils clean

