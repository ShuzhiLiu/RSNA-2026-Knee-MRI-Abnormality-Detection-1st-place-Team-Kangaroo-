"""Final annotation request helpers extracted from the consumed source."""
import base64
import csv
import json
import math
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
PLANES = ('Sagittal', 'Coronal', 'Axial')
LABELS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
REPORT_GUIDES = {'ACL': 'ACL/LCA/VKB/anterior cruciate ligament injury, sprain, partial tear or complete rupture.', 'MCL': 'MCL/LCM/medial collateral ligament injury, sprain, partial tear or complete rupture.', 'Medial Meniscus': 'A tear of the medial/inner meniscus; intrasubstance degeneration without a surface tear is not enough.', 'Lateral Meniscus': 'A tear of the lateral/outer meniscus; intrasubstance degeneration without a surface tear is not enough.', 'Medial OA': 'Osteoarthritis/chondrosis in the medial tibiofemoral compartment, including definite compartmental cartilage loss.', 'Lateral OA': 'Osteoarthritis/chondrosis in the lateral tibiofemoral compartment, including definite compartmental cartilage loss.', 'PF OA': 'Patellofemoral osteoarthritis/chondrosis involving patellar or trochlear cartilage.', 'Effusion': 'Knee joint effusion; do not count a Baker cyst as joint effusion.', 'Synovitis': 'Synovitis or definite synovial thickening/proliferation; simple effusion alone is not explicit synovitis.', "Baker's": 'Baker/popliteal cyst, including a ruptured Baker cyst.', 'Contusion': 'Traumatic bone contusion/bone bruise; do not count generic degenerative marrow edema.', 'Fracture': 'Acute or subacute fracture around the knee; marrow edema or contusion alone is not an explicit fracture.'}
DISEASE_GUIDES = {'ACL': 'Determine whether there is anterior cruciate ligament injury (sprain,\npartial tear, or complete tear), not merely whether a normal ACL is visible on one\nslice. Trace the ligament through the intercondylar notch on sagittal images and\ncross-check coronal/axial images. Positive evidence includes abnormal intrasubstance\nfluid signal, thickening, fiber discontinuity, lax or abnormal orientation, or an\nempty expected course. Pivot-shift bone bruises or anterior tibial translation are\nsupportive but must not alone establish ACL injury. Postoperative appearance or\npoor visualization should increase uncertainty rather than force a positive call.', 'MCL': 'Determine whether there is medial collateral ligament injury (sprain,\npartial tear, or complete tear). Follow superficial and deep MCL fibers from the\nmedial femoral epicondyle to the proximal medial tibia, chiefly on coronal and axial\nfluid-sensitive images. Periligamentous edema with ligament thickening supports a\nlow-grade injury; partial fiber disruption supports a partial tear; complete\ndiscontinuity/retraction supports a full tear. Do not label isolated nonspecific\nmedial soft-tissue edema as MCL injury when the ligament is intact.', 'Medial Meniscus': 'Determine whether the medial meniscus has a tear. Inspect\nanterior horn, body, posterior horn, and roots on sagittal and coronal images.\nPositive evidence includes linear/complex abnormal signal reaching an articular\nsurface on convincing slices, truncation or missing tissue, displaced fragment,\nbucket-handle morphology, radial/root defect, or tear-associated extrusion. Pure\nintrasubstance degeneration that does not reach a surface is not sufficient.\nDistinguish the medial meniscus from adjacent ligaments and partial-volume effects.', 'Lateral Meniscus': 'Determine whether the lateral meniscus has a tear. Inspect\nanterior horn, body, posterior horn, and roots on sagittal and coronal images.\nPositive evidence includes abnormal signal contacting an articular surface,\nmorphologic truncation, radial/root defect, displaced fragment, or convincing\ntear-associated extrusion. Do not call isolated intrasubstance degeneration a tear.\nAvoid false positives from the popliteal hiatus, meniscofemoral ligaments, vascular\nstructures, and partial-volume effects around the posterior horn.', 'Medial OA': 'Determine whether osteoarthritis is present in the medial\ntibiofemoral compartment specifically. Integrate medial femoral condyle and medial\ntibial plateau cartilage loss, marginal osteophytes, subchondral sclerosis/cysts,\nchronic subchondral marrow change, joint-space/meniscal degeneration and extrusion.\nA single traumatic contusion or isolated meniscal tear is not by itself medial OA.\nDo not transfer abnormalities from the lateral or patellofemoral compartment to\nthis label.', 'Lateral OA': 'Determine whether osteoarthritis is present in the lateral\ntibiofemoral compartment specifically. Integrate lateral femoral condyle and lateral\ntibial plateau cartilage loss, marginal osteophytes, subchondral sclerosis/cysts,\nchronic subchondral marrow change, and chronic meniscal degeneration/extrusion.\nDo not infer lateral OA from isolated acute marrow edema, a meniscal tear alone, or\nosteoarthritis confined to the medial or patellofemoral compartment.', 'PF OA': 'Determine whether osteoarthritis is present in the patellofemoral\ncompartment. Examine patellar facets and trochlear cartilage, best on axial and\nsagittal images. Positive evidence is a coherent degenerative pattern such as\ncartilage thinning/full-thickness loss, marginal osteophytes, and subchondral cystic,\nsclerotic, or chronic marrow changes. Do not label isolated transient edema or\ntibiofemoral-only disease as patellofemoral OA.', 'Effusion': 'Determine whether there is a clinically meaningful knee joint\neffusion. Inspect the suprapatellar pouch and medial/lateral joint recesses on\nfluid-sensitive sagittal and axial images. Definite excess fluid that expands the\nrecesses supports a positive label. A trace physiologic amount of fluid should not\nbe confidently positive. Do not confuse a Baker cyst or other extra-articular fluid\ncollection with intra-articular effusion.', 'Synovitis': 'Determine whether synovitis is present. Look for synovial\nthickening, nodular/villous proliferation, inflammatory tissue in the suprapatellar,\nparapatellar or intercondylar recesses, and abnormal inflammatory change around\nHoffa fat. Simple fluid alone establishes effusion, not synovitis. Because\nnon-contrast MRI may not reliably separate synovium from fluid, use intermediate\nprobability when thickened synovial tissue is not convincingly demonstrated.', "Baker's": 'Determine whether a Baker (popliteal) cyst is present. Seek a\nfluid-signal collection at the posteromedial knee centered between the medial head\nof gastrocnemius and semimembranosus tendon, preferably with the characteristic neck\nor communication with the joint. Assess axial and sagittal images. Do not call an\narbitrary posterior ganglion, vascular structure, or diffuse edema a Baker cyst.\nRupture or dissection can coexist with the characteristic origin.', 'Contusion': 'Determine whether there is an acute traumatic bone contusion in\nany knee bone (distal femur, proximal tibia, patella, or fibular head). Search all\nthree planes for geographic/patchy marrow edema-like high signal on fluid-sensitive\nimages without a definite fracture line or cortical disruption. Distinguish acute\ncontusion patterns from chronic osteoarthritic subchondral change, osteonecrosis,\nred marrow, artifact, and edema caused by a visible fracture.', 'Fracture': 'Determine whether any acute or subacute fracture is present in\nthe imaged knee, including distal femur, proximal tibia/tibial plateau, patella, or\nproximal fibula. Require convincing structural evidence such as a low-signal\nfracture line, cortical interruption, trabecular disruption, articular step-off,\ndepression, or displaced fragment, often with surrounding marrow edema. Bone marrow\nedema alone without a fracture line or structural disruption is a contusion, not a\nconfident fracture.'}

def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values

def load_study_rows(train_csv: Path, scope: str) -> list[dict[str, str]]:
    with train_csv.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    labelled = lambda row: all((row.get(label, '').strip() for label in LABELS))
    if scope == 'labelled':
        return [row for row in rows if labelled(row)]
    if scope == 'unlabelled':
        return [row for row in rows if not labelled(row)]
    return rows

@lru_cache(maxsize=512)
def image_data_url(path_text: str) -> str:
    path = Path(path_text)
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'

def http_post_json(url: str, body: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def binary_choice_probability(response: dict[str, Any], negative: str, positive: str) -> tuple[float, dict[str, Any]]:
    choice = response['choices'][0]
    content = (choice['message'].get('content') or '').strip()
    positions = (choice.get('logprobs') or {}).get('content') or []
    if not positions:
        raise ValueError('Choice response did not contain token logprobs')
    candidates = positions[0].get('top_logprobs') or []
    masses = {positive: 0.0, negative: 0.0}
    retained = []
    for candidate in candidates:
        token = str(candidate.get('token') or '').strip()
        logprob = float(candidate['logprob'])
        if not token or logprob < -100:
            continue
        for label in masses:
            if label.startswith(token):
                masses[label] += math.exp(logprob)
                retained.append({'token': token, 'logprob': logprob, 'branch': label})
                break
    denominator = masses[positive] + masses[negative]
    if denominator <= 0:
        raise ValueError(f'No {positive}/{negative} prefixes in top logprobs: {candidates[:8]!r}')
    probability = masses[positive] / denominator
    return (probability, {'choice': content, 'branch_mass': masses, 'prefix_logprobs': retained})

def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    positive = y_true == 1
    negative = y_true == 0
    n_positive, n_negative = (int(positive.sum()), int(negative.sum()))
    if not n_positive or not n_negative:
        return math.nan
    order = np.argsort(y_score, kind='mergesort')
    sorted_score = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))
