# Coding Instructions for Video Annotation

## Overview

You are annotating short-form TikTok videos about running shoes. For each video, you will provide labels for several visual and stylistic features.

**Watch the entire video before annotating.** Pay attention to the overall feel, not just individual moments.

---

## Feature Definitions

### 1. Perspective (POV)

**What to look for:** Who is the camera representing?

| Code | Definition | Visual Cues |
|------|------------|-------------|
| **1st Person** | Camera shows the viewer's perspective | Hands visible holding products; camera moves as if you're there; no face of camera operator visible; shaky/dynamic movement |
| **2nd Person** | Subject directly addresses the viewer | Person looking at camera; speaking/talking to you; making eye contact; feels like a conversation |
| **3rd Person** | Camera is an objective observer | Subject not looking at camera; watching people interact; documentary-style; observing events |

**Decision Rules:**
- If hands are prominently visible (e.g., unboxing, holding shoes) → **1st Person**
- If someone is looking at the camera and talking → **2nd Person**
- If we're watching people talk to each other or do activities without addressing us → **3rd Person**

---

### 2. Camera Distance

**What to look for:** How close is the main subject to the camera?

| Code | Definition | Visual Cues |
|------|------------|-------------|
| **Close (Personal)** | Intimate framing, head-and-shoulders or closer | Face fills most of frame; can see facial details clearly; feels personal |
| **Mid/Wide (Social/Public)** | Subject at conversational distance or farther | Full body visible; multiple people in frame; background is prominent; feels more distant |

**Decision Rules:**
- If you can see facial details clearly and face takes up >30% of frame → **Close**
- If you can see full body or the background is prominent → **Mid/Wide**
- When in doubt about borderline cases, consider: does this feel intimate or distant?

---

### 3. Gaze Direction

**What to look for:** Where is the main subject looking?

| Code | Definition | Visual Cues |
|------|------------|-------------|
| **At Camera** | Subject's eyes are directed at the camera/viewer | Making eye contact with you; direct gaze; feels like they're looking at you |
| **Away from Camera** | Subject looking elsewhere | Looking at product; looking at another person; eyes directed off-screen |
| **No Face Visible** | Cannot determine gaze | No face in frame; face obscured; only hands/products visible |

**Decision Rules:**
- If the subject makes eye contact with the camera most of the time → **At Camera**
- If their face is visible but they're looking at something else → **Away**
- If there's no face or it's too small/blurry to tell → **No Face Visible**
- **Multiple people:** Focus on the dominant/main speaker. If unclear, look at who takes up more space.
- **Near-gaze:** If gaze is almost at camera but slightly off (common for teleprompter), code as **At Camera**

---

### 4. Editing Pace

**What to look for:** How frequently do scene changes (cuts) occur?

| Code | Definition | Feel |
|------|------------|------|
| **Slow** | Few cuts; shots last several seconds | Relaxed, unhurried; like a conversation or detailed explanation |
| **Moderate** | Regular cuts; shots last 1-3 seconds | Normal pacing; neither rushed nor slow |
| **Fast** | Frequent cuts; shots under 1 second | Energetic, dynamic; music-video style; lots of quick changes |

**Decision Rules:**
- Count rough number of scene changes
- Consider the overall energy/feel
- Brand ads tend to be faster; reviews tend to be slower

---

### 5. Visual Density

**What to look for:** How "busy" is the visual content?

| Code | Definition | What Makes It Dense |
|------|------------|---------------------|
| **Minimal** | Clean, simple visuals | Plain background; single subject; little text; few elements |
| **Moderate** | Some visual complexity | Some text overlays; a few objects; moderate activity |
| **High** | Busy, complex visuals | Multiple text overlays; many objects; rapid changes; lots happening on screen |

**Things that add density:**
- Text overlays
- Multiple products
- Effects/filters
- Split screens
- Fast motion
- Busy backgrounds
- Multiple people

---

### 6. Gesture/Hands

**What to look for:** Are hands visible and what are they doing?

| Code | Definition |
|------|------------|
| **Hands Visible** | Can see hands; holding product, gesturing casually |
| **Hands Not Visible** | Hands not in frame |
| **Pointing/Gesturing** | Active pointing, demonstrating, or emphatic gestures |

---

## General Guidelines

1. **Watch the whole video** before making decisions
2. **Consider the dominant impression** - what's true for most of the video?
3. **When uncertain**, make your best judgment and mark as "difficult" if needed
4. **Be consistent** - apply the same standards across all videos
5. **Take breaks** - annotation fatigue affects quality

## Keyboard Shortcuts (in the tool)

- `1-3`: Quick select for current feature
- `Space`: Play/pause video
- `Enter`: Save and go to next video
- `N`: Skip to next video without saving
- `?`: Show these instructions

---

## Examples

### Example 1: Influencer Review
- **Perspective:** 2nd person (talking to camera)
- **Distance:** Close (face fills frame)
- **Gaze:** At camera
- **Pace:** Slow (few cuts, talking)
- **Density:** Moderate (some text overlays)
- **Gesture:** Hands visible (holding shoes)

### Example 2: Brand Montage
- **Perspective:** 3rd person (observational)
- **Distance:** Mid/Wide (full body shots)
- **Gaze:** Away (subjects running, not looking at camera)
- **Pace:** Fast (quick cuts, music-driven)
- **Density:** High (lots of motion, text)
- **Gesture:** Hands not visible

### Example 3: POV Unboxing
- **Perspective:** 1st person (your hands opening box)
- **Distance:** Close (product close-up)
- **Gaze:** No face visible
- **Pace:** Moderate
- **Density:** Minimal (just the product)
- **Gesture:** Hands visible

---

## Questions?

If you encounter edge cases or have questions about specific videos, use the "Notes" field to document your reasoning or flag for discussion.
