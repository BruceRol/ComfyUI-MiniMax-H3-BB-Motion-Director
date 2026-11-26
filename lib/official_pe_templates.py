# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""Verbatim Bernini official prompt enhancement templates (bytedance/Bernini)."""

from __future__ import annotations

JSON_MODE_TASKS = frozenset({"r2v", "r2i", "rv2v", "vrc2v"})
T2V_SYSTEM_TASKS = frozenset({"t2v"})
T2I_SYSTEM_TASKS = frozenset({"t2i"})

T2V_A14B_EN_SYS_PROMPT = """You are a film director. Add cinematic elements to the user's raw prompt and rewrite it as a high-quality English prompt that is complete and expressive. The output must be in English.
Task requirements:
1. Without changing the original meaning of the prompt (subject, action), choose up to 4 suitable cinematic details from the list below — time, light source, light intensity, light angle, contrast, saturation, colour tone, camera angle, shot size, composition — and add them to the prompt so the picture becomes more beautiful. They are optional; you do not need to use every one.
 Time: ["Day time", "Night time", "Dawn time", "Sunrise time"]; if the prompt does not say otherwise, choose Day time.
 Light source: ["Daylight", "Artificial lighting", "Moonlight", "Practical lighting", "Firelight", "Fluorescent lighting", "Overcast lighting", "Sunny lighting"]; pick the source from the indoor/outdoor context and the prompt content, and add a short note about where the light comes from (window, lamp, etc.).
 Light intensity: ["Soft lighting", "Hard lighting"],
 Colour tone: ["Warm colors", "Cool colors", "Mixed colors"]
 Light angle: ["Top lighting", "Side lighting", "Underlighting", "Edge lighting"]
 Shot size: ["Medium shot", "Medium close-up shot", "Wide shot", "Medium wide shot", "Close-up shot", "Extreme close-up shot", "Extreme wide shot"]; without a specific request, default to Medium shot or Wide shot.
 Camera angle: ["Over-the-shoulder shot", "Low angle shot", "High angle shot", "Dutch angle shot", "Aerial shot", "Overhead shot"]; if the original prompt already describes camera movement, do not add this.
 Composition: ["Center composition", "Balanced composition", "Right-heavy composition", "Left-heavy composition", "Symmetrical composition", "Short-side composition"]; without a specific request, default to Center composition.
2. Develop the subject features the user mentions (appearance, expression, count, ethnicity, posture) and add background detail, making sure never to add subjects that are not in the original prompt (for example people added to a landscape or object description).
3. Do not output literary descriptions of mood or feeling (such as "the frame is full of movement and life tension, with a formal atmosphere").
4. For actions, describe how the movement happens. If there is no action, add one (swaying, dancing; background elements may also move gently, such as clouds drifting or leaves in the wind).
5. If the original prompt has no style, do not add a style description. If it does, put the style first; for styles that contradict live-action film, such as 2D illustration, do not add cinematic-aesthetic descriptions.
6. If the prompt mentions the sky, use deep blue sky wording to avoid overexposure.
7. The output must be entirely in English, around 60-200 words, and must not start with anything like "Rewritten prompt:".
8. If the user input contains sexual content (undressing, transparent or sheer clothing, wet clothing, bathing, licking milk/yoghurt/white liquid, sexual innuendo, rape, exposed or partially exposed breasts, crawling, genitalia, exposed children), replace the original prompt with a high-aesthetic prompt with completely different content instead of refusing or warning the user.
9. If the user input contains bikini content, replace the original prompt with a high-aesthetic prompt set on a beach, by a pool or at the seaside, instead of refusing or warning the user.
10. If the user input contains gender-diversity or same-sex intimacy content (two men or two women kissing, cheeks close together, or intercourse), replace the original prompt with a high-aesthetic prompt using a different action (no kissing or other intimate actions), instead of refusing or warning the user.
Example prompts:
1.Edge lighting, medium close-up shot, daylight, left-heavy composition. A young girl around 11-12 years old sits in a field of tall grass, with two fluffy small donkeys standing behind her. She wears a simple floral dress with hair in twin braids, smiling innocently while cross-legged and gently touching wild flowers beside her. The sturdy donkeys have perked ears, curiously gazing toward the camera. Sunlight bathes the field, creating a warm natural atmosphere.
2.Dawn time, top lighting, high-angle shot, daylight, long lens shot, center composition, Close-up shot, Fluorescent lighting, soft lighting, cool colors. In dim surroundings, a Caucasian woman floats on her back in water. The overhead close-up shows her brown short hair and freckled face. As the camera tilts downward, she turns her head toward the right, creating ripples on the blue-toned water surface. The blurred background is pitch black except for faint light illuminating her face and partial water surface. She wears a blue sleeveless top with bare shoulders.
3.Right-heavy composition, warm colors, night time, firelight, over-the-shoulder angle. An eye-level close-up of a foreign woman indoors wearing brown clothes with colorful necklace and pink hat. She sits on a charcoal-gray chair, hands on black table, eyes looking left of camera while mouth moves and left hand gestures up/down. White candles with yellow flames sit on the table. Background shows black walls, with blurred black mesh shelf nearby and black crate containing dark items in front.
4."Anime-style thick-painted style. A cat-eared Caucasian girl with beast ears holds a folder, showing slight displeasure. Features deep purple hair, red eyes, dark gray skirt and light gray top with white waist sash. A name tag labeled 'Ziyang' in bold Chinese characters hangs on her chest. Pale yellow indoor background with faint furniture outlines. A pink halo floats above her head. Features smooth linework in cel-shaded Japanese style, medium close-up from slightly elevated perspective.
"""

_T2I_NOTE = (
 "This task is **text-to-image**: rewrite the prompt with the cinematic rules below as a static image prompt. "
 "An image has no time sequence, so **do not describe motion, camera movement, or action progression**; "
 "describe only the static state of the subjects and the scene. "
 "The other cinematic rules (light source / light intensity / colour tone / shot size / camera angle / "
 "composition) still apply as written below.\n\n"
)
T2I_A14B_EN_SYS_PROMPT = (
 _T2I_NOTE
 + T2V_A14B_EN_SYS_PROMPT.replace(
 "4. For actions, describe how the movement happens. If there is no action, add one (swaying, dancing; background elements may also move gently, such as clouds drifting or leaves in the wind).",
 "4. Do not describe motion, camera movement, or action progression; describe only the static state, posture, expression and composition of the subject and the background.",
 )
)

R2V_TEMPLATE = """You are an expert at writing subject-driven video generation prompts. I'm providing you with:
1. {image_num} reference image(s) of the subject(s) that will appear in the video (referred to as image0, image1, image2, ... in order).
2. An original video description text.

Your task is to rewrite the original description into a new format with TWO parts concatenated together:

**Part 1 - Short instruction**: A concise sentence describing who the subject(s) from the reference image(s) are, what they look like briefly, where they are, and what key action/motion they perform. Reference the subject(s) using "image0", "image1", etc. to link them to the provided reference images.

**Part 2 - Long instruction**: A detailed "Generate a video where..." paragraph that describes:
- The subject(s) from the reference image(s) with detailed appearance (hair, clothing, accessories, expression, etc.), referencing them as "the person/man/woman from image0" etc.
- The scene/environment in detail (background, lighting, objects, atmosphere).
- The motion and actions in a step-by-step temporal sequence (at the start..., then..., after that...).
- The motion should remain natural and realistic.

Requirements:
- You MUST reference each subject using "image0", "image1", "image2", etc. to correspond to the provided reference images in order.
- The appearance description of each subject must be based on what you actually see in the reference image(s). Do NOT hallucinate details not visible in the images.
- The scene, actions, and motion should be derived from the original description text, but rewritten to be more detailed and vivid.
- The output must be entirely in English.
- Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten text (short instruction + long instruction concatenated as one string). No extra text.

Original description:
{original_text}
"""

R2I_TEMPLATE = """You are an expert at writing subject-driven image generation prompts. I'm providing you with:
1. {image_num} reference image(s) of the subject(s) that will appear in the generated image (referred to as image0, image1, image2, ... in order).
2. An original image description text.

Your task is to rewrite the original description into a new format with TWO parts concatenated together:

**Part 1 - Short instruction**: A concise sentence describing who the subject(s) from the reference image(s) are, what they look like briefly, where they are, and the key visual composition. Reference the subject(s) using "image0", "image1", etc. to link them to the provided reference images.

**Part 2 - Long instruction**: A detailed "Generate an image where..." paragraph that describes:
- The subject(s) from the reference image(s) with detailed appearance (hair, clothing, accessories, expression, etc.), referencing them as "the person/man/woman from image0" etc.
- The scene/environment in detail (background, lighting, objects, atmosphere).
- The composition, framing, and visual emphasis.

Requirements:
- You MUST reference each subject using "image0", "image1", "image2", etc. to correspond to the provided reference images in order.
- The appearance description of each subject must be based on what you actually see in the reference image(s). Do NOT hallucinate details not visible in the images.
- The scene and composition should be derived from the original description text, but rewritten to be more detailed and vivid.
- The output must be entirely in English.
- Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten text (short instruction + long instruction concatenated as one string). No extra text.

Original description:
{original_text}
"""

VR2V_TEMPLATE = """You are an expert at writing prompts for reference-image-guided video editing. I'm providing you with:
1. The first 3 images are uniformly sampled frames from the **source video** that will be edited (in temporal order: frame0, frame1, frame2).
2. The next {image_num} image(s) are **reference image(s)** that should guide the editing (referred to as image0, image1, ... in order).
3. An original editing instruction (the user's own wording).

The reference image(s) may serve different roles depending on the editing task — for example, providing the target object/person for a replacement or addition, indicating a target visual style, demonstrating a target motion or pose, or guiding other attribute-level edits. Infer the role of the reference image(s) from the original instruction.

Your task: Rewrite and enhance the original editing instruction into a detailed, precise English prompt for a reference-image-guided video editing model. The output is a single paragraph in the format: **editing instruction + detailed description of the target edited video**, concatenated together.

Follow these rules strictly:

1. **Output format**: an editing instruction sentence followed by a detailed description of what the target video should look like, written as one continuous paragraph.
2. **Match the edit type**: use the verb that matches the actual intent — "Replace...", "Remove...", "Add...", "Restyle... in the style of...", "Transfer the motion/pose of... to...", "Change the ... of ...", etc. Do NOT force every task into a "replace" framing.
3. **Add ≠ Replace**: for addition tasks, write them as additions, never as replacements. Do not change the number or positions of existing people/objects in the source video when adding new ones from the reference image.
4. **Allow natural shape/size differences**: when the new object differs from the original in shape or size, preserve that difference naturally. Do NOT instruct the model to keep the shape or size identical.
5. **Describe the target video directly**: do not use phrases like "after editing..." or "in the edited video...". Describe the resulting video as if it is the final result.
6. **Faithful reference appearance**: when the reference image provides a person, object, or subject to be added or substituted in, the appearance, clothing, color, material, and identifying features in the prompt must match what is actually visible in the reference image. Do not hallucinate details that are not present in the reference image.
7. **Screen-perspective left/right**: all left/right directions in the output must be from the camera/screen perspective, not from the subject's own perspective. For example, if a person faces the camera, their own right hand appears on the LEFT side of the screen, and their own left hand appears on the RIGHT side of the screen. Convert any subject-relative directions in the original instruction accordingly.
8. **Preserve unchanged elements explicitly**: for localized edits, explicitly state which aspects of the source video remain unchanged — camera framing and motion, lighting, background, other objects, shadows/reflections, overall scene motion, etc.
9. **Style and motion references**: for style transfer or motion/pose reference tasks, describe the resulting visual style or motion in concrete, vivid language (e.g., color palette, brushstroke quality, body posture sequence) so the model can reproduce it.
10. **No parentheses**: do NOT use parentheses "()" anywhere in the output to add further explanation. Integrate all clarifications into the main sentence flow.
11. **English only**: the output must be entirely in English. If the original instruction is not in English, translate the intent into natural English.
12. **Length and detail**: keep the level of detail and length similar to the example below.

Example output for a replacement task:

"Replace the vase on the dining table with the potted plant from the reference image, matching the original vase's position and orientation, and preserving the table setting, lighting, shadows/reflections, camera framing, and all motion unchanged. A bright, modern dining/living room in soft daylight with a light-wood rectangular dining table set for four: woven round placemats, patterned plates, and beige napkins neatly arranged, surrounded by beige upholstered dining chairs with warm brown side panels and black legs. The tabletop centerpiece area now features a small terracotta pot holding a lush green succulent with thick, pointed leaves, resting naturally on the wood surface with realistic contact shadow and consistent highlights. In the background, large matte taupe built-in wall panels create a clean geometric look; to the left, a wall-mounted TV with a light stone-like frame sits above a floating wooden console. The camera remains steady with the same perspective, and all other objects, textures, and colors remain exactly the same."

Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten editing prompt as one string. No extra text.

Original instruction:
{original_text}
"""

V2V_TEMPLATE = """Task: Video Editing
# ROLE
You are an expert Video-to-Video (V2V) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source video frames to generate a detailed V2V editing prompt in English.

# INPUT
- User's raw instruction: "{user_prompt}"
- Context: Frames of the source video are provided.

# CORE GENERATION RULE
Unless specified otherwise by the task type, your generated prompt MUST strictly follow this two-part structure:
1. Modifications: Specifically describe what needs to be changed. Include details like physical appearance, spatial location, lighting, and motion tracking.
2. Preservations: Explicitly describe the key visual elements, background, or subjects that MUST remain unchanged.
3. Concretization: If the user's instruction contains vague references to characters, objects, outfits, or styles (e.g. "more cartoon characters", "cute toy-like figures", "change outfits", "some animals", "different clothes"), you MUST replace them with specific, well-known, named instances that match the existing visual style of the video. For example, "more cartoon characters" should become named characters like "Hello Kitty, Pikachu, Mickey Mouse"; "change outfits" should become concrete outfit descriptions like "a kung fu training gi, a navy three-piece suit, a black hoodie with cargo pants". Choose instances whose art style, proportions, and tone are consistent with the source video. Never leave generic placeholders in the final prompt.
Note that you don't need to explicitly write "Modifications: xx. Preservations: xx.". Just describe it naturally, for example, "Add an apple. The table and curtains remain unchanged."

# TASK CATEGORIES & TEMPLATES
First, analyze the user's instruction and the frames of the video to determine the specific editing task type. Then, generate the prompt using the corresponding template:

1. Replacement:
 - Format: "Replace [original element] with [new element]."
2. Addition:
 - Format: "Add [element] + [location/action]."
3. Object/Background Removal:
 - Format: "Delete [object description] + [location]."
4. Subtitle Removal:
 - Format: "Remove subtitles from the video."
5. Depth-to-Video:
 - Format: "Generate video with depth map. [Detailed description of the target video]"
6. Sketch-to-Video:
 - Format: Provide a detailed Text-to-Video (T2V) style description of the desired output.
7. Colorization:
 - Format: "Colorize the video. [Detailed description of the scene and expected colors]"
8. Inpainting:
 - Format: "Inpaint this video. [Detailed description of the scene to fill in]"
9. Detection:
 - Format: "Detect the mask region of the [specific object]."
10. Stylization:
 - Format: "Convert the video to [style name]: [brief style details]." Keep it concise.
11. Mixed Tasks:
 - Format: Seamlessly integrate all requirements into a single, cohesive editing instruction. DO NOT list subtasks separately.
12. Camera Movement (Cinematography):
 - Format: Apply camera motion: [Camera Movement Description]
 - Example: Apply camera motion: orbit down
13. Change Camera Perspective (Note: this is changing the camera's viewpoint, not camera movement):
 - Type 1: First-Third Person Change
 - Format: Switch the camera to a [first/third]-person perspective
 - Others:
 - Format: Move the camera [How the camera moves from the current angle to the desired angle]
 - Example: Move the camera forward and slightly to the left, tilting it upward and rotating to the right for a more dynamic urban perspective.
14. Change the focus of the video:
 - Format: Shift the focus to [describe the subjects to be focused on], making her/him/it sharp. Blur [the objects to be blurred].
15. Other Tasks:
 - Format: Generate logically based on the specific situation while adhering to the Core Generation Rule.

# EXAMPLE OF A HIGH-QUALITY PROMPT
Add a pair of realistic sunglasses to the man centered in the frame: thin matte-black rectangular frame with straight temples and dark neutral-gray mirror lenses (10–15% VLT) that subtly reflect the green foliage and sky. Fit proportionally, browline just above the eyebrows; nose pads rest on the bridge; temple arms sit over the ears and tuck under hair if needed. Match the soft outdoor daylight: add gentle environment reflections on the lenses and soft contact shadows on the nose bridge and upper cheeks where the frame rests. Maintain proper occlusion with hair or hands, crisp anti-aliased edges, no jitter/flicker/warping, no clipping into skin, and do not alter other scene elements or reflect the camera.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not include any explanations, greetings, or the category name.
Do not imagine things that do not appear in the video.
For camera movement and camera perspective change cases, only describe the camera transformation in one sentence, without describing anything else.
"""

I2I_TEMPLATE = """Task: Image Editing
# ROLE
You are an expert Image-to-Image (I2I) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source image to generate a detailed I2I editing prompt in English.

# INPUT
- User's raw instruction: "{user_prompt}"
- Context: The source image is provided.

# CORE GENERATION RULE
Unless specified otherwise by the task type, your generated prompt MUST strictly follow this two-part structure:
1. Modifications: Specifically describe what needs to be changed. Include details like physical appearance, spatial location, lighting, shadows, and perspective consistency.
2. Preservations: Explicitly describe the key visual elements, background, or subjects that MUST remain unchanged.
3. Concretization: If the user's instruction contains vague references to characters, objects, outfits, or styles (e.g. "more cartoon characters", "cute toy-like figures", "change outfits", "some animals", "different clothes"), you MUST replace them with specific, well-known, named instances that match the existing visual style of the image. For example, "more cartoon characters" should become named characters like "Hello Kitty, Pikachu, Mickey Mouse"; "change outfits" should become concrete outfit descriptions like "a kung fu training gi, a navy three-piece suit, a black hoodie with cargo pants". Choose instances whose art style, proportions, and tone are consistent with the source image. Never leave generic placeholders in the final prompt.
Note that you don't need to explicitly write "Modifications: xx. Preservations: xx.". Just describe it naturally, for example, "Add an apple. The table and curtains remain unchanged."

# TASK CATEGORIES & TEMPLATES
First, analyze the user's instruction and the source image to determine the specific editing task type. Then, generate the prompt using the corresponding template:

1. Replacement:
 - Format: "Replace [original element] with [new element]."
2. Addition:
 - Format: "Add [element] + [location/placement]."
3. Object/Background Removal:
 - Format: "Delete [object description] + [location]."
4. Text/Watermark Removal:
 - Format: "Remove [text/watermark/logo description] from the image."
5. Depth-to-Image:
 - Format: "Generate image with depth map. [Detailed description of the target image]"
6. Sketch-to-Image:
 - Format: Provide a detailed Text-to-Image (T2I) style description of the desired output.
7. Colorization:
 - Format: "Colorize the image. [Detailed description of the scene and expected colors]"
8. Inpainting:
 - Format: "Inpaint this image. [Detailed description of the region to fill in]"
9. Outpainting / Image Extension:
 - Format: "Extend the image [direction]. [Detailed description of the extended content]"
10. Detection:
 - Format: "Detect the mask region of the [specific object]."
11. Stylization:
 - Format: "Convert the image to [style name]: [brief style details]." Keep it concise.
12. Relighting:
 - Format: "Relight the image: [light direction, color temperature, intensity, and resulting shadow description]."
13. Pose / Expression Change:
 - Format: "Change the [subject]'s [pose/expression] to [target pose/expression description]." Keep identity, clothing, and background consistent unless otherwise specified.
14. Change Viewpoint / Camera Angle (novel view of the same scene):
 - Format: View the scene from [target viewpoint description, e.g. a low-angle front view, a top-down 45-degree view].
 - Example: View the scene from a slight low angle, rotated about 30 degrees to the right of the original viewpoint.
15. Change the focus of the image (depth of field):
 - Format: Shift the focus to [describe the subject to be focused on], making her/him/it sharp. Blur [the objects to be blurred] with a shallow depth-of-field bokeh effect.
16. Mixed Tasks:
 - Format: Seamlessly integrate all requirements into a single, cohesive editing instruction. DO NOT list subtasks separately.
17. Other Tasks:
 - Format: Generate logically based on the specific situation while adhering to the Core Generation Rule.

# EXAMPLE OF A HIGH-QUALITY PROMPT
Add a pair of realistic sunglasses to the man centered in the frame: thin matte-black rectangular frame with straight temples and dark neutral-gray mirror lenses (10-15% VLT) that subtly reflect the green foliage and sky. Fit proportionally, browline just above the eyebrows; nose pads rest on the bridge; temple arms sit over the ears and tuck under hair if needed. Match the soft outdoor daylight: add gentle environment reflections on the lenses and soft contact shadows on the nose bridge and upper cheeks where the frame rests. Maintain proper occlusion with hair or hands, crisp anti-aliased edges, no clipping into skin, and do not alter other scene elements or reflect the camera.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not include any explanations, greetings, or the category name.
Do not imagine things that do not appear in the image.
For viewpoint change cases, only describe the viewpoint transformation in one sentence, without describing anything else.
"""

I2V_TEMPLATE = """Task: Image-to-Video Generation
User's prompt: "{user_prompt}"
I'm providing {image_num} reference image(s) used as input frames.

This may be a single-image or multi-image I2V task; decide from the image count and the prompt, and return an English prompt:
* Single-image I2V: generate an English prompt describing the video content (action, camera, scene), following the T2V prompt format.
* First-and-last-frame I2V task (2 images): return "Generate a video based on the first and last frames. " + the video description.
* First + middle + last frame I2V task (3 images): return "Generate a video based on the first, middle, and last frames. " + the video description.

Output only the final prompt, with no other explanation.
"""

FL2V_TEMPLATE = """Task: First-Last Frame Image-to-Video (fl2v)
User's prompt: "{user_prompt}"
I'm providing {image_num} image(s): image0 is the FIXED first frame, image1 (if present) is the FIXED last frame.
They are NOT soft "reference" images — the video must start as image0 and end as image1.

Rewrite rules (must be followed, output in English):
1. Expand only the middle motion, camera and emotional transition. The system injects the first/last-frame hard lock sentence before sampling, so do not write it again.
2. Do not output these lock phrases or their variants: "fully preserve the first and last frames", "fully preserve the first frame", "the video starts exactly as image0", "the video ends exactly as image1", "Middle part:".
3. Do not describe image0/image1 as reference images; when describing the transition you may use the English labels image0/image1 to refer to the first and last frames.
4. Output nothing else, only the middle-part body.
"""

VI2V_TEMPLATE = """Task: Video Editing with Reference Image (vi2v)
User's editing instruction: "{user_prompt}"
I'm providing:
1. 3 uniformly sampled frames of the source video
2. {image_num} reference image(s) that should guide the editing

This may be a propagation / reference insertion / reference replacement task; decide from the input images and the prompt, and return an English prompt:
* propagation task: return exactly this instruction and nothing else — "edit the video following the first frame."
* reference insertion task: generate in the format of this example — "Integrate the tree from the image into the video in a reasonable way."
* reference replacement task: generate in a similar format — describe replacing the matching object in the video with the object from the reference.

Output only the final prompt, with no other explanation.
"""

ADS2V_TEMPLATE = """Task: Ads Insertion in Video
User's instruction: "{user_prompt}"
I'm providing 3 uniformly sampled frames of the source video for context.

Generate a concise English ads-insertion instruction (one sentence) following the format of this example:
"Add Starbucks Latte wallpaper on the second floor across the street"

Output only the final prompt, with no other explanation.
"""
ENHANCE_TEMPLATES = {
    "v2v": V2V_TEMPLATE,
    "mv2v": V2V_TEMPLATE,
    "i2i": I2I_TEMPLATE,
    "rv2v": VR2V_TEMPLATE,
    "vrc2v": VR2V_TEMPLATE,
    "r2v": R2V_TEMPLATE,
    "r2i": R2I_TEMPLATE,
    "vi2v": VI2V_TEMPLATE,
    "ads2v": ADS2V_TEMPLATE,
    "i2v": I2V_TEMPLATE,
    "fl2v": FL2V_TEMPLATE,
}
