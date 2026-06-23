# Sample Manufacturing-aware Defect Explanation Reports

This file contains representative structured explanation examples generated from Stage 6.5.

## grid

### Example 1

- Image: `datasets/MVTecAD/grid/test/thread/010.png`
- True defect: `thread`
- Predicted defect: `thread`
- Top-2 predictions: `thread|metal_contamination`
- Candidate region: `(96, 11)-(147, 119)`
- Candidate anomaly score: `mean=0.9006, max=1.0000`
- Defect family: `deformation`
- Manufacturing processes: molding, surface forming, pattern alignment, visual inspection
- Possible causes: assembly error, mechanical stress, pressing defect, handling damage

**Explanation:** The image is predicted as defect type `thread` on category `grid`. The product is described as a regular texture surface with grid-like patterned surface. Visual evidence for this prediction includes local shape distortion, bent structure, or abnormal thread-like geometry; related attributes are bent part, shape distortion, misalignment, structural change. According to the anomaly localization result, the top PatchCore candidate region is located at (96, 11)-(147, 119), with area 2594. Relevant manufacturing or inspection stages include molding, surface forming, pattern alignment, visual inspection. Possible causes include assembly error, mechanical stress, pressing defect, handling damage. The inspection focus for this category includes broken grid structure, foreign material, local pattern interruption.

### Example 2

- Image: `datasets/MVTecAD/grid/test/thread/006.png`
- True defect: `thread`
- Predicted defect: `thread`
- Top-2 predictions: `thread|bent`
- Candidate region: `(74, 6)-(103, 51)`
- Candidate anomaly score: `mean=0.8890, max=1.0000`
- Defect family: `deformation`
- Manufacturing processes: molding, surface forming, pattern alignment, visual inspection
- Possible causes: assembly error, mechanical stress, pressing defect, handling damage

**Explanation:** The image is predicted as defect type `thread` on category `grid`. The product is described as a regular texture surface with grid-like patterned surface. Visual evidence for this prediction includes local shape distortion, bent structure, or abnormal thread-like geometry; related attributes are bent part, shape distortion, misalignment, structural change. According to the anomaly localization result, the top PatchCore candidate region is located at (74, 6)-(103, 51), with area 929. Relevant manufacturing or inspection stages include molding, surface forming, pattern alignment, visual inspection. Possible causes include assembly error, mechanical stress, pressing defect, handling damage. The inspection focus for this category includes broken grid structure, foreign material, local pattern interruption.

### Example 3

- Image: `datasets/MVTecAD/grid/test/thread/003.png`
- True defect: `thread`
- Predicted defect: `thread`
- Top-2 predictions: `thread|metal_contamination`
- Candidate region: `(94, 102)-(117, 129)`
- Candidate anomaly score: `mean=0.8873, max=1.0000`
- Defect family: `deformation`
- Manufacturing processes: molding, surface forming, pattern alignment, visual inspection
- Possible causes: assembly error, mechanical stress, pressing defect, handling damage

**Explanation:** The image is predicted as defect type `thread` on category `grid`. The product is described as a regular texture surface with grid-like patterned surface. Visual evidence for this prediction includes local shape distortion, bent structure, or abnormal thread-like geometry; related attributes are bent part, shape distortion, misalignment, structural change. According to the anomaly localization result, the top PatchCore candidate region is located at (94, 102)-(117, 129), with area 471. Relevant manufacturing or inspection stages include molding, surface forming, pattern alignment, visual inspection. Possible causes include assembly error, mechanical stress, pressing defect, handling damage. The inspection focus for this category includes broken grid structure, foreign material, local pattern interruption.

## leather

### Example 1

- Image: `datasets/MVTecAD/leather/test/glue/018.png`
- True defect: `glue`
- Predicted defect: `glue`
- Top-2 predictions: `glue|cut`
- Candidate region: `(170, 24)-(187, 42)`
- Candidate anomaly score: `mean=0.9489, max=1.0000`
- Defect family: `contamination`
- Manufacturing processes: cutting, pressing, dyeing, surface finishing
- Possible causes: dust, oil, residue, insufficient cleaning

**Explanation:** The image is predicted as defect type `glue` on category `leather`. The product is described as a textured surface material with leather-like surface texture. Visual evidence for this prediction includes foreign material, residue, or unexpected blob-like region; related attributes are foreign material, local stain, unexpected blob, color inconsistency. According to the anomaly localization result, the top PatchCore candidate region is located at (170, 24)-(187, 42), with area 276. Relevant manufacturing or inspection stages include cutting, pressing, dyeing, surface finishing. Possible causes include dust, oil, residue, insufficient cleaning. The inspection focus for this category includes cut, fold, glue, color defect, surface damage.

### Example 2

- Image: `datasets/MVTecAD/leather/test/glue/014.png`
- True defect: `glue`
- Predicted defect: `glue`
- Top-2 predictions: `glue|color`
- Candidate region: `(53, 30)-(69, 47)`
- Candidate anomaly score: `mean=0.9477, max=1.0000`
- Defect family: `contamination`
- Manufacturing processes: cutting, pressing, dyeing, surface finishing
- Possible causes: dust, oil, residue, insufficient cleaning

**Explanation:** The image is predicted as defect type `glue` on category `leather`. The product is described as a textured surface material with leather-like surface texture. Visual evidence for this prediction includes foreign material, residue, or unexpected blob-like region; related attributes are foreign material, local stain, unexpected blob, color inconsistency. According to the anomaly localization result, the top PatchCore candidate region is located at (53, 30)-(69, 47), with area 243. Relevant manufacturing or inspection stages include cutting, pressing, dyeing, surface finishing. Possible causes include dust, oil, residue, insufficient cleaning. The inspection focus for this category includes cut, fold, glue, color defect, surface damage.

### Example 3

- Image: `datasets/MVTecAD/leather/test/glue/002.png`
- True defect: `glue`
- Predicted defect: `glue`
- Top-2 predictions: `glue|poke`
- Candidate region: `(134, 124)-(151, 141)`
- Candidate anomaly score: `mean=0.9479, max=1.0000`
- Defect family: `contamination`
- Manufacturing processes: cutting, pressing, dyeing, surface finishing
- Possible causes: dust, oil, residue, insufficient cleaning

**Explanation:** The image is predicted as defect type `glue` on category `leather`. The product is described as a textured surface material with leather-like surface texture. Visual evidence for this prediction includes foreign material, residue, or unexpected blob-like region; related attributes are foreign material, local stain, unexpected blob, color inconsistency. According to the anomaly localization result, the top PatchCore candidate region is located at (134, 124)-(151, 141), with area 263. Relevant manufacturing or inspection stages include cutting, pressing, dyeing, surface finishing. Possible causes include dust, oil, residue, insufficient cleaning. The inspection focus for this category includes cut, fold, glue, color defect, surface damage.

## screw

### Example 1

- Image: `datasets/MVTecAD/screw/test/scratch_head/013.png`
- True defect: `scratch_head`
- Predicted defect: `scratch_head`
- Top-2 predictions: `scratch_head|scratch_neck`
- Candidate region: `(63, 182)-(85, 203)`
- Candidate anomaly score: `mean=0.9210, max=1.0000`
- Defect family: `scratch`
- Manufacturing processes: metal forming, threading, surface treatment, mechanical handling
- Possible causes: friction, handling damage, transportation damage, surface processing defect

**Explanation:** The image is predicted as defect type `scratch_head` on category `screw`. The product is described as a small metal fastener with metal object with thread and head structure. Visual evidence for this prediction includes thin line-like surface damage or abrasion around the highlighted anomaly crop; related attributes are thin, linear, surface damage, high local contrast. According to the anomaly localization result, the top PatchCore candidate region is located at (63, 182)-(85, 203), with area 389. Relevant manufacturing or inspection stages include metal forming, threading, surface treatment, mechanical handling. Possible causes include friction, handling damage, transportation damage, surface processing defect. The inspection focus for this category includes thread damage, scratch on head, neck defect, small structural defect.

### Example 2

- Image: `datasets/MVTecAD/screw/test/scratch_head/006.png`
- True defect: `scratch_head`
- Predicted defect: `scratch_head`
- Top-2 predictions: `scratch_head|scratch_neck`
- Candidate region: `(196, 62)-(212, 82)`
- Candidate anomaly score: `mean=0.9201, max=1.0000`
- Defect family: `scratch`
- Manufacturing processes: metal forming, threading, surface treatment, mechanical handling
- Possible causes: friction, handling damage, transportation damage, surface processing defect

**Explanation:** The image is predicted as defect type `scratch_head` on category `screw`. The product is described as a small metal fastener with metal object with thread and head structure. Visual evidence for this prediction includes thin line-like surface damage or abrasion around the highlighted anomaly crop; related attributes are thin, linear, surface damage, high local contrast. According to the anomaly localization result, the top PatchCore candidate region is located at (196, 62)-(212, 82), with area 274. Relevant manufacturing or inspection stages include metal forming, threading, surface treatment, mechanical handling. Possible causes include friction, handling damage, transportation damage, surface processing defect. The inspection focus for this category includes thread damage, scratch on head, neck defect, small structural defect.

### Example 3

- Image: `datasets/MVTecAD/screw/test/thread_side/007.png`
- True defect: `thread_side`
- Predicted defect: `thread_side`
- Top-2 predictions: `thread_side|thread_top`
- Candidate region: `(124, 117)-(147, 134)`
- Candidate anomaly score: `mean=0.9203, max=1.0000`
- Defect family: `deformation`
- Manufacturing processes: metal forming, threading, surface treatment, mechanical handling
- Possible causes: assembly error, mechanical stress, pressing defect, handling damage

**Explanation:** The image is predicted as defect type `thread_side` on category `screw`. The product is described as a small metal fastener with metal object with thread and head structure. Visual evidence for this prediction includes local shape distortion, bent structure, or abnormal thread-like geometry; related attributes are bent part, shape distortion, misalignment, structural change. According to the anomaly localization result, the top PatchCore candidate region is located at (124, 117)-(147, 134), with area 311. Relevant manufacturing or inspection stages include metal forming, threading, surface treatment, mechanical handling. Possible causes include assembly error, mechanical stress, pressing defect, handling damage. The inspection focus for this category includes thread damage, scratch on head, neck defect, small structural defect.

## wood

### Example 1

- Image: `datasets/MVTecAD/wood/test/hole/000.png`
- True defect: `hole`
- Predicted defect: `hole`
- Top-2 predictions: `hole|scratch`
- Candidate region: `(47, 51)-(81, 92)`
- Candidate anomaly score: `mean=0.8308, max=1.0000`
- Defect family: `crack`
- Manufacturing processes: cutting, polishing, coating, surface inspection
- Possible causes: material stress, impact, thermal stress, molding defect

**Explanation:** The image is predicted as defect type `hole` on category `wood`. The product is described as a natural textured material with wood surface with natural grain. Visual evidence for this prediction includes broken, missing, or hollow local structure around the detected anomaly region; related attributes are irregular line, broken structure, sharp boundary. According to the anomaly localization result, the top PatchCore candidate region is located at (47, 51)-(81, 92), with area 1152. Relevant manufacturing or inspection stages include cutting, polishing, coating, surface inspection. Possible causes include material stress, impact, thermal stress, molding defect. The inspection focus for this category includes scratch, hole, liquid stain, color defect.

### Example 2

- Image: `datasets/MVTecAD/wood/test/hole/009.png`
- True defect: `hole`
- Predicted defect: `hole`
- Top-2 predictions: `hole|scratch`
- Candidate region: `(20, 38)-(51, 74)`
- Candidate anomaly score: `mean=0.8250, max=1.0000`
- Defect family: `crack`
- Manufacturing processes: cutting, polishing, coating, surface inspection
- Possible causes: material stress, impact, thermal stress, molding defect

**Explanation:** The image is predicted as defect type `hole` on category `wood`. The product is described as a natural textured material with wood surface with natural grain. Visual evidence for this prediction includes broken, missing, or hollow local structure around the detected anomaly region; related attributes are irregular line, broken structure, sharp boundary. According to the anomaly localization result, the top PatchCore candidate region is located at (20, 38)-(51, 74), with area 920. Relevant manufacturing or inspection stages include cutting, polishing, coating, surface inspection. Possible causes include material stress, impact, thermal stress, molding defect. The inspection focus for this category includes scratch, hole, liquid stain, color defect.

### Example 3

- Image: `datasets/MVTecAD/wood/test/hole/008.png`
- True defect: `hole`
- Predicted defect: `hole`
- Top-2 predictions: `hole|scratch`
- Candidate region: `(32, 57)-(66, 93)`
- Candidate anomaly score: `mean=0.8207, max=1.0000`
- Defect family: `crack`
- Manufacturing processes: cutting, polishing, coating, surface inspection
- Possible causes: material stress, impact, thermal stress, molding defect

**Explanation:** The image is predicted as defect type `hole` on category `wood`. The product is described as a natural textured material with wood surface with natural grain. Visual evidence for this prediction includes broken, missing, or hollow local structure around the detected anomaly region; related attributes are irregular line, broken structure, sharp boundary. According to the anomaly localization result, the top PatchCore candidate region is located at (32, 57)-(66, 93), with area 1001. Relevant manufacturing or inspection stages include cutting, polishing, coating, surface inspection. Possible causes include material stress, impact, thermal stress, molding defect. The inspection focus for this category includes scratch, hole, liquid stain, color defect.
