# WEBCAMS.md - Regional Webcam Context

Cameras considered for "is the mountain out?", and why the live one is live.
`mountain.toml` `[webcam]` is the source of truth for what the pipeline
actually fetches; this file is the reasoning behind it.

## Primary Target: Mount Rainier

### KING 5 Queen Anne Tower Camera — **PRIMARY since 2026-09-24**

- **Image URL:** https://cdn.tegna-media.com/king/weather/queenanne.jpg
- **Location:** KING 5's tower on Queen Anne hill, near Kerry Park
  (~47.629, -122.360), looking southeast over downtown Seattle.
- **Framing:** the classic Kerry Park postcard. **Rainier sits above and just
  left of the downtown towers**, roughly **80x45 px** in the 1920x1080 frame,
  bearing ~152°, 96 km out. Small, but it lands inside the center 224x224 crop
  the model sees (near its left edge — see the drift warning below).
- **Format:** plain still JPEG, 1920x1080, refreshes about every 4 minutes.
  No auth, no referer check, no tokens in the URL.
- **Burn-in:** a black strip across the **bottom ~25 px** carrying
  `9:49:37 AM - 9/23/2026 - Queen Anne Tower Camera - KING 5` in white. It
  changes every frame and is the highest-contrast thing in the picture, so it
  is cropped before inference (`[webcam] crop_bottom_px`, `collect/frame.py`).
  Captures are archived uncropped.
- **Status:** **ACTIVE.**

> [!WARNING]
> **This is a PTZ broadcast camera on someone else's CDN, not a fixed research
> instrument.** The station can pan, tilt or zoom it for a live shot and leave
> it somewhere new, and nothing will announce that. Rainier sits near the left
> edge of the crop the model sees, so a modest pan right can put the mountain
> out of frame while the pipeline keeps happily predicting.
>
> **Re-verify the framing periodically** (monthly, and after any run of
> implausible predictions):
>
> ```sh
> curl -s -o /tmp/queenanne.jpg https://cdn.tegna-media.com/king/weather/queenanne.jpg
> # look at it: towers centered, Space Needle right, Rainier above/left of the towers
> ```
>
> If the pointing has moved, the labels and the checkpoint are as invalid as
> they were at the UW→KING 5 cut. Say so in `CHECKPOINTS.md` and start again.

Freshness is also checked automatically: three consecutive ticks returning
byte-identical frames marks the feed dead and `state.json` carries
`status: "stale"` instead of a prediction (`collect/freshness.py`).

### UW Atmospheric Sciences Webcam 2 — **DEAD**

- **Image URL:** https://atmos.uw.edu/data/images/webcam2_latest.jpg
- **Was:** 7th floor of the ATG building, UW Seattle campus, pointing southeast
  toward Rainier. This was the primary camera from the start of the project
  until 2026-09; every capture and every label in `data/` before that date came
  from it.
- **Status:** **DEAD — 404 since ~2026-09-15.** The URL 302s to
  `a.atmos.washington.edu/data/images/webcam2_latest.jpg`, which also 404s, and
  the camera is **gone from UW's server listing**. This is not the intermittent
  2026-08 outage (when the file went away and came back); the entry no longer
  exists upstream.
- **Consequence:** the checkpoint trained on this framing does not transfer.
  See `CHECKPOINTS.md` and `TRAINING.md`.

### UW Red Square — **NO VIDEO**

- **Image URL:** https://www.washington.edu/cambots/camera1_l.jpg
- **Status:** **NO VIDEO.** The URL still returns HTTP 200 and
  `Content-Type: image/jpeg`, but the payload is an 8 KB, 704x480 **black
  frame with the words "NO VIDEO" burned into the corner** (verified
  2026-09-24: mean luminance 1.1/255). A healthy-looking 200 with nothing
  behind it — precisely the failure the freshness check exists for, since
  those bytes never change. Rejected on quality anyway: lower resolution and
  less favorable framing than webcam 2 even when it worked.

### KING 5 Tacoma Camera — considered, not chosen

- **Image URL:** https://cdn.tegna-media.com/king/weather/tacoma.jpg
- **Why it was tempting:** Tacoma is ~40 km closer to the mountain, so Rainier
  is much larger in frame and a partial/full call would be easier.
- **Why not:** it is a different airmass from KSEA, which is the METAR station
  the model's weather features come from and the one this project has years of
  paired observations for — switching the view *and* the weather feature's
  provenance at once makes two uncontrolled changes. Queen Anne keeps the
  Seattle sightline the project has always answered ("is the mountain out
  *from Seattle*") and keeps KSEA honest. Keep this URL as the fallback if the
  Queen Anne camera is repointed or retired; adopting it means re-labelling
  anyway.

### National Park Service (NPS) - Paradise

- **Mountain View:** http://www.nps.gov/webcams-mora/mountain.jpg
- **Tatoosh Range:** http://www.nps.gov/webcams-mora/tatoosh.jpg
- **Camp Muir:** https://www.nps.gov/webcams-mora/muir.jpg
- **Status:** not used. These sit *on* the mountain, which answers a different
  question (what is the weather at Paradise) than the one this project asks
  (can you see Rainier from Seattle).

## Other Regional Webcams

- **KING 5 Columbia Center:** https://cdn.tegna-media.com/king/weather/columbia.jpg
  — downtown rooftop, another Tegna still. Not evaluated in depth.
- **Space Needle:** 360-degree panorama.
- **Crystal Mountain:** http://skicrystal.com/The-Mountain/about/Webcams
- **Mission Ridge:** http://www.missionridge.com/webcams

## Technical Sources

- [UW Atmospheric Sciences Webcam Index](https://a.atmos.washington.edu/data/webcams.html)
  (webcam 2 no longer listed)
- KING 5 stills are served from `cdn.tegna-media.com/king/weather/<name>.jpg`.
  They are unauthenticated and cache-friendly; be polite — one fetch per tick.
