# AeroMem

AeroMem studies how a small aerial robot can use observations from an earlier sortie to revisit a specific object after its view has been blocked or the object has been moved.

## Approach

The system stores persistent object identities with 3D boxes, sparse observed surface points, representative crops, visual embeddings, and relations to nearby landmarks. On revisit, it projects stored surface points into current RGB-D frames. Foreground obstruction calls for a new viewpoint while preserving the track. Clear lines of sight and repeated close-range visual absence justify a relocation search. A vision-language model ranks furniture-scale sub-zones, and close views are used to check the original identity before updating the track.

The CPU fast path updates occupancy, checks line of sight, retrieves memory candidates and plans local paths. Keyframe perception and event-triggered semantic reasoning operate at a lower rate.

## Selected results

The held-out ReplicaCAD benchmark contains 20 layouts and 60 paired instances. The main study comprises 480 rendered episodes. The table reports benchmark results for the comparison adapters implemented under the shared aerial protocol. All task-level values are in [results.csv](results.csv).

| Method | Success rate (%) | SPL |
| --- | ---: | ---: |
| VLFM-NoMem | 33.33 | 0.262 |
| ConceptGraphs-Static | 35.00 | 0.274 |
| HOV-SG-Frustum | 31.67 | 0.245 |
| 3D-Mem-VLM | 40.00 | 0.296 |
| SG-Nav-Online | 36.67 | 0.279 |
| Active-NBV-3DSG | 41.67 | 0.271 |
| AeroMem | 68.33 | 0.454 |

AeroMem completed 16/20 static revisits, 14/20 occlusion trials and 11/20 relocation trials. On Jetson AGX Orin in 30 W mode, single-core CPU replay matched all 500 archived decision outputs, with mean 10.188 ms and P95 16.960 ms per update. Two remotely piloted Sunray300 runs provided onboard RGB-D observations, and a separate handheld recording supplied the occlusion state check.

## Run the CPU example

This repository contains a reference subset for camera geometry, object memory, depth visibility and voxel planning. The example uses synthetic depth arrays to exercise the visibility rule. Full simulator assets, model weights, raw telemetry and flight-control tooling are outside the scope of the example.

```bash
python -m pip install -r requirements.txt
python demo_visibility.py
```

Expected states are `occluded`, `line_of_sight`, and `unknown`. A clear line of sight indicates visibility of the stored location. Instance presence requires a visual match.
