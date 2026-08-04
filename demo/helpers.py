"""
helpers.py
----------
All helper functions for the Co-DETR HOI demo:

Inference
    find_interaction_branch(query_head)
    call_interaction(branch, tok_a, tok_b)
    run_inference(model, test_pipeline, img_path, device,
                  class_names, score_thr, nms_iou)

Drawing
    compute_style(img_shape)                   — per-image drawing params
    draw_ui(vis, detections, hf_inters, fs_inters, style,
            verbose_labels=False)               — top-level renderer

Internal helpers (not re-exported, but available if you need them):
    rounded_rect, rounded_rect_translucent, blend_box, put_text_crisp,
    box_iou, place_det_label, resolve_coincident_boxes
"""

import cv2
import numpy as np
import torch

from mmdet.core      import bbox_cxcywh_to_xyxy
from mmcv.ops        import batched_nms
from mmcv.parallel   import collate, scatter

import configs as C


# ═════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════
def find_interaction_branch(query_head, verbose=True):
    """Locate the interaction_head submodule inside query_head."""
    cands = {n: m for n, m in query_head.named_modules()
             if 'interaction_head' in n.lower() and n != ''}
    if not cands:
        raise AttributeError("interaction_head not found in query_head")
    name = sorted(cands, key=len)[0]
    if verbose:
        print(f"[INFO] Using interaction head: query_head.{name}")
    return cands[name]


def call_interaction(branch, tok_a, tok_b):
    """
    Returns (interacts: bool, prob: float).
    Decision rule matches training: logits.argmax(dim=1) == 1 means interaction.
    prob = softmax[1] (for display only).
    """
    with torch.no_grad():
        x      = torch.cat([tok_a, tok_b], dim=0).unsqueeze(0)
        logits = branch.mlp(x)
        pred   = logits.argmax(dim=1).item()
        prob   = torch.softmax(logits, dim=-1)[0, 1].item()
    return pred == 1, prob


def run_inference(model, test_pipeline, img_path, device,
                  class_names, score_thr=0.3, nms_iou=0.5):
    """
    Preprocess -> forward -> topk+NMS while keeping exact query indices.

    Returns:
        detections : list of dicts
            {box, score, class_id, class_name, query_idx, center}
            box is xyxy in original-image pixel coords
        embeddings : Tensor [Q, D] from the last decoder layer
    """
    data = test_pipeline(dict(img_info=dict(filename=img_path), img_prefix=''))
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [device])[0]

    img_tensor = data['img']
    if isinstance(img_tensor, list):
        img_tensor = img_tensor[0]

    img_metas = data['img_metas']
    if isinstance(img_metas, list) and isinstance(img_metas[0], list):
        img_metas = img_metas[0]

    for meta in img_metas:
        meta['batch_input_shape'] = tuple(img_tensor.shape[-2:])

    with torch.no_grad():
        feats    = model.extract_feat(img_tensor)
        outs, hs = model.query_head(feats, img_metas, return_hs=True)

    outputs_classes, outputs_coords = outs[0], outs[1]
    cls_logits = outputs_classes[-1][0]
    dec_boxes  = outputs_coords[-1][0]
    embeddings = hs[-1][0]

    scores_all  = cls_logits.sigmoid()
    max_per_img = model.query_head.test_cfg.get('max_per_img', 300)
    flat_scores, flat_idx = scores_all.view(-1).topk(max_per_img)
    flat_labels = flat_idx % model.query_head.num_classes
    flat_qidx   = flat_idx // model.query_head.num_classes

    img_shape = img_metas[0]['img_shape']
    H, W      = img_shape[:2]
    factor    = dec_boxes.new_tensor([W, H, W, H])
    topk_boxes = bbox_cxcywh_to_xyxy(dec_boxes[flat_qidx]) * factor

    nms_cfg = dict(type='soft_nms', iou_threshold=nms_iou, min_score=score_thr)
    _, keep = batched_nms(topk_boxes, flat_scores, flat_labels, nms_cfg)

    sf = np.array(img_metas[0].get('scale_factor', 1.0), dtype=np.float32).ravel()
    if sf.size == 1:
        sf = np.tile(sf, 4)
    elif sf.size == 2:
        sf = np.array([sf[0], sf[1], sf[0], sf[1]])

    detections = []
    for k in keep.tolist():
        det_score = flat_scores[k].item()
        if det_score < score_thr:
            continue
        cls_id    = flat_labels[k].item()
        query_idx = flat_qidx[k].item()
        box_px    = topk_boxes[k].cpu().numpy()
        box_orig  = box_px / sf
        cx = int((box_orig[0] + box_orig[2]) / 2)
        cy = int((box_orig[1] + box_orig[3]) / 2)
        detections.append(dict(
            box       =box_orig,
            score     =det_score,
            class_id  =cls_id,
            class_name=class_names[cls_id],
            query_idx =query_idx,
            center    =(cx, cy),
        ))

    return detections, embeddings


# ═════════════════════════════════════════════════════════════
# DRAWING PRIMITIVES
# ═════════════════════════════════════════════════════════════
def rounded_rect(img, pt1, pt2, color, radius, thickness=-1):
    """Filled (thickness=-1) or outlined rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(1, int(radius))
    r = min(r, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness, cv2.LINE_AA)


def rounded_rect_translucent(img, pt1, pt2, color, radius, alpha,
                             border_color=None, border_thickness=0):
    """Filled rounded rectangle alpha-blended into img, optional border on top."""
    x1, y1 = pt1
    x2, y2 = pt2
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(img.shape[1], x2); y2c = min(img.shape[0], y2)
    if x2c <= x1c or y2c <= y1c:
        return
    roi     = img[y1c:y2c, x1c:x2c]
    overlay = roi.copy()
    rp1 = (x1 - x1c, y1 - y1c)
    rp2 = (x2 - x1c, y2 - y1c)
    rounded_rect(overlay, rp1, rp2, color, radius, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    if border_color is not None and border_thickness > 0:
        rounded_rect(img, pt1, pt2, border_color, radius, border_thickness)


def blend_box(img, pt1, pt2, color, alpha=0.15):
    """Translucent solid-color fill inside an xyxy box."""
    x1, y1 = pt1
    x2, y2 = pt2
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2); y2 = min(img.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def put_text_crisp(img, text, org, font, scale, color, thickness):
    """Text with a thin dark outline so it stays legible on any background."""
    outline_th = thickness + max(1, int(round(thickness * 0.8)))
    cv2.putText(img, text, org, font, scale, (0, 0, 0),
                outline_th, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color,
                thickness, cv2.LINE_AA)


def box_iou(b1, b2):
    """Standard IoU between two xyxy boxes."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ═════════════════════════════════════════════════════════════
# STYLE (resolution-aware drawing parameters)
# ═════════════════════════════════════════════════════════════
def compute_style(img_shape):
    """
    img_shape : (H, W) or (H, W, C)
    Returns a dict of drawing parameters scaled for this image.

    Uses the LONGER edge (not the diagonal), which tracks perceived image
    size more accurately across different aspect ratios.
    """
    H, W = img_shape[:2]
    edge = float(max(H, W))
    raw  = edge / C.REF_EDGE
    s    = raw ** C.SCALE_EXPONENT
    s    = float(np.clip(s, C.SCALE_MIN, C.SCALE_MAX))

    def _max1(v):
        return max(1, int(round(v)))

    return dict(
        scale         = s,
        img_h         = H,
        img_w         = W,
        # Boxes
        box_thickness = _max1(C.BOX_THICKNESS_REF * s),
        dot_radius    = _max1(C.DOT_RADIUS_REF * s),
        # Links
        link_halo     = _max1(C.LINK_HALO_REF * s),
        link_core     = _max1(C.LINK_CORE_REF * s),
        # Detection label
        det_font      = cv2.FONT_HERSHEY_DUPLEX,
        det_font_sc   = C.DET_FONT_SCALE_REF * s,
        det_font_th   = _max1(C.DET_FONT_THICK_REF * s),
        det_pad_x     = _max1(C.DET_PAD_X_REF * s),
        det_pad_y     = _max1(C.DET_PAD_Y_REF * s),
        # Link probability badge
        lnk_font      = cv2.FONT_HERSHEY_DUPLEX,
        lnk_font_sc   = C.LNK_FONT_SCALE_REF * s,
        lnk_font_th   = _max1(C.LNK_FONT_THICK_REF * s),
        lnk_pad_x     = _max1(C.LNK_PAD_X_REF * s),
        lnk_pad_y     = _max1(C.LNK_PAD_Y_REF * s),
        # Thresholds scaled with image
        min_box_side  = C.MIN_BOX_SIDE_FOR_LABEL * s,
        min_link_len  = C.MIN_LINK_LEN_FOR_BADGE * s,
    )


# ═════════════════════════════════════════════════════════════
# LAYOUT (label placement + coincident-box handling)
# ═════════════════════════════════════════════════════════════
def place_det_label(img_w, img_h, box, label_w, label_h, style,
                    prefer='auto'):
    """
    prefer:
      'auto'           — outside-above, or inside-top if the box is very
                         large, or below if above would clip.
      'outside_above'  — force outside-above (fall back to auto if clipped).
      'inside_top'     — force inside-top, aligned to the box's top edge.
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    can_above = y1 - label_h >= 0
    can_below = y2 + label_h < img_h
    big_box = (label_w <= 0.55 * box_w) and (label_h <= 0.25 * box_h) \
              and box_h > label_h * 3

    if prefer == 'inside_top' and box_w >= label_w and box_h >= label_h:
        bg_pt1 = (x1, y1)
        bg_pt2 = (x1 + label_w, y1 + label_h)
    elif prefer == 'outside_above' and can_above:
        bg_pt1 = (x1, y1 - label_h)
        bg_pt2 = (x1 + label_w, y1)
    elif big_box or not can_above:
        if box_w >= label_w and box_h >= label_h * 1.5:
            bg_pt1 = (x1, y1)
            bg_pt2 = (x1 + label_w, y1 + label_h)
        elif can_above:
            bg_pt1 = (x1, y1 - label_h)
            bg_pt2 = (x1 + label_w, y1)
        elif can_below:
            bg_pt1 = (x1, y2)
            bg_pt2 = (x1 + label_w, y2 + label_h)
        else:
            bg_pt1 = (x1, y1)
            bg_pt2 = (x1 + label_w, y1 + label_h)
    else:
        # Default: outside-above, corner-aligned
        bg_pt1 = (x1, y1 - label_h)
        bg_pt2 = (x1 + label_w, y1)

    # Clamp horizontally
    if bg_pt2[0] > img_w:
        shift = bg_pt2[0] - img_w
        bg_pt1 = (bg_pt1[0] - shift, bg_pt1[1])
        bg_pt2 = (bg_pt2[0] - shift, bg_pt2[1])
    if bg_pt1[0] < 0:
        shift = -bg_pt1[0]
        bg_pt1 = (bg_pt1[0] + shift, bg_pt1[1])
        bg_pt2 = (bg_pt2[0] + shift, bg_pt2[1])

    txt_pt = (bg_pt1[0] + style['det_pad_x'],
              bg_pt2[1] - style['det_pad_y'])
    return bg_pt1, bg_pt2, txt_pt


def resolve_coincident_boxes(detections):
    """
    Tag first/second-object detection pairs that occupy nearly the same box.

    Populates each detection with:
        draw_box          — coords used for rendering (unchanged here)
        label_prefer      — 'auto' | 'outside_above' | 'inside_top'
        coincident_outer  — True for the secondobject of a coincident pair
        coincident_inner  — True for the firstobject of a coincident pair
    """
    for d in detections:
        d['draw_box']         = d['box'].copy()
        d['label_prefer']     = 'auto'
        d['coincident_inner'] = False
        d['coincident_outer'] = False

    firsts  = [d for d in detections if d['class_id'] == 1]
    seconds = [d for d in detections if d['class_id'] == 2]

    for f in firsts:
        for s in seconds:
            if box_iou(f['box'], s['box']) >= C.COINCIDENT_IOU:
                s['label_prefer']     = 'outside_above'
                f['label_prefer']     = 'inside_top'
                s['coincident_outer'] = True
                f['coincident_inner'] = True


# ═════════════════════════════════════════════════════════════
# RENDERER (top-level)
# ═════════════════════════════════════════════════════════════
def _det_label_text(d):
    """Class (+ optional track id) and score for box labels."""
    tid = d.get('track_id', None)
    if tid is not None:
        return f"{d['class_name']}#{int(tid)} {d['score']:.2f}"
    return f"{d['class_name']} {d['score']:.2f}"


def draw_ui(vis, detections, hf_inters, fs_inters, style,
            verbose_labels=False):
    """
    Render the full visualisation in place on `vis`.

    detections : list of dicts (from run_inference)
    hf_inters  : list of (hand_det, first_det, prob) tuples
    fs_inters  : list of (first_det, second_det, prob) tuples
    style      : dict from compute_style
    verbose_labels :
        False (default) — smart mode: hide labels on tiny boxes, badges on
                          short links, and anywhere the text would overflow.
        True            — show every label and every badge regardless of size.
    """
    s = style
    H, W = vis.shape[:2]
    smart = not verbose_labels

    # Pre-compute which boxes will have their label hidden (so we can also
    # hide badges on links that attach to unlabeled boxes).
    hidden_label_ids = set()
    if smart:
        for d in detections:
            x1, y1, x2, y2 = d['box'].astype(int)
            if min(x2 - x1, y2 - y1) < s['min_box_side']:
                hidden_label_ids.add(id(d))
                continue
            txt = f"{d['class_name']} {d['score']:.2f}"
            (tw, _), _ = cv2.getTextSize(txt, s['det_font'],
                                         s['det_font_sc'],
                                         s['det_font_th'])
            if tw + 2 * s['det_pad_x'] > (x2 - x1):
                hidden_label_ids.add(id(d))

    # ── one link (line + optional badge) ─────────────────────
    def draw_link(img, d1, d2, color, prob):
        c1 = np.array(d1['center'], dtype=np.float32)
        c2 = np.array(d2['center'], dtype=np.float32)
        link_len = float(np.linalg.norm(c2 - c1))

        c1i = tuple(c1.astype(int))
        c2i = tuple(c2.astype(int))
        cv2.line(img, c1i, c2i, C.COL_LINE_HALO,
                 s['link_halo'], cv2.LINE_AA)
        cv2.line(img, c1i, c2i, color,
                 s['link_core'], cv2.LINE_AA)

        label = f"{prob:.2f}"
        (tw, th), bl = cv2.getTextSize(label, s['lnk_font'],
                                       s['lnk_font_sc'], s['lnk_font_th'])
        badge_w = tw + 2 * s['lnk_pad_x']

        if smart:
            if link_len < s['min_link_len']:
                return
            if badge_w > link_len:
                return
            if id(d1) in hidden_label_ids or id(d2) in hidden_label_ids:
                return

        mx = (c1i[0] + c2i[0]) // 2
        my = (c1i[1] + c2i[1]) // 2
        px, py = s['lnk_pad_x'], s['lnk_pad_y']
        bg_pt1 = (mx - tw // 2 - px, my - th // 2 - py)
        bg_pt2 = (mx + tw // 2 + px, my + th // 2 + py + bl // 2)
        radius = max(2, int(4 * s['scale']))
        rounded_rect_translucent(
            img, bg_pt1, bg_pt2, (25, 25, 25), radius,
            C.BADGE_BG_ALPHA,
            border_color=color,
            border_thickness=max(1, int(1.3 * s['scale'])),
        )
        put_text_crisp(img, label, (mx - tw // 2, my + th // 2),
                       s['lnk_font'], s['lnk_font_sc'],
                       (255, 255, 255), s['lnk_font_th'])

    # ── links first, so boxes/labels draw on top ─────────────
    for h, f, prob in hf_inters:
        draw_link(vis, h, f, C.COL_LINE_HF, prob)
    for f, so, prob in fs_inters:
        draw_link(vis, f, so, C.COL_LINE_FS, prob)

    # ── tag coincident F/S pairs ─────────────────────────────
    resolve_coincident_boxes(detections)

    # ── boxes + labels ───────────────────────────────────────
    # Secondobjects first so firstobjects visually sit "inside" them.
    draw_order = sorted(detections, key=lambda d: -d['class_id'])
    for d in draw_order:
        x1, y1, x2, y2 = d['draw_box'].astype(int)
        col = C.CLASS_COLORS[d['class_id']]

        # Skip translucent fill on the inner of a coincident pair.
        if not d.get('coincident_inner', False):
            blend_box(vis, (x1, y1), (x2, y2), col, alpha=C.BOX_FILL_ALPHA)

        # Inner of a coincident pair: thinner outline so the outer shows.
        if d.get('coincident_inner', False):
            outline_th = max(1, s['box_thickness'] - 2)
        else:
            outline_th = s['box_thickness']
        cv2.rectangle(vis, (x1, y1), (x2, y2), col,
                      outline_th, cv2.LINE_AA)

        # Center dot
        cv2.circle(vis, d['center'], s['dot_radius'],
                col, -1, cv2.LINE_AA)
        cv2.circle(vis, d['center'], s['dot_radius'],
                (255, 255, 255), max(1, int(1.25 * s['scale'])), cv2.LINE_AA)

        # Label (includes track_id when the video tracker assigned one)
        txt = _det_label_text(d)
        (tw, th), bl = cv2.getTextSize(txt, s['det_font'],
                                       s['det_font_sc'], s['det_font_th'])
        lbl_w = tw + 2 * s['det_pad_x']
        lbl_h = th + 2 * s['det_pad_y'] + bl

        if smart:
            short_side = min(x2 - x1, y2 - y1)
            if short_side < s['min_box_side']:
                continue
            if lbl_w > (x2 - x1):
                continue

        bg_pt1, bg_pt2, txt_pt = place_det_label(
            W, H, (x1, y1, x2, y2), lbl_w, lbl_h, s,
            prefer=d.get('label_prefer', 'auto'),
        )
        radius = max(2, int(3 * s['scale']))
        rounded_rect_translucent(
            vis, bg_pt1, bg_pt2, col, radius, C.LABEL_BG_ALPHA
        )
        put_text_crisp(vis, txt, txt_pt,
                       s['det_font'], s['det_font_sc'],
                       (255, 255, 255), s['det_font_th'])
