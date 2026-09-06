# Concept naming protocol

## Single-sample workflow

For an input image x, NFS-CBM predicts class ŷ from concept activations z. The class-specific contribution of concept k is R(ŷ, k) = W(ŷ, k) zₖ, as defined in Eq. 6 of the manuscript. Select k* = arg maxₖ R(ŷ, k). This selection uses contribution to the predicted class, not activation alone.

Perturb zₖ* and keep all other concept dimensions fixed. CDG produces the original reconstruction under z and the perturbed reconstruction under z′. The naming step compares these two images. The term perturbation does not prescribe its direction.

`name_concepts.py` starts with the exported image pair and concept identifier. It does not run NFS-CBM inference, select the dimension or generate images. Those preceding stages require the trained model implementation and checkpoint.

## Dataset-level naming in the manuscript

Section III-G also describes a complementary protocol: for each fixed concept k, select the M test samples with the highest activation for that concept, then assess recurring effects across their intervention pairs. This differs from selecting the highest-contribution concept for a single prediction. The supplied manuscript uses positive concept increases for these pairs. The script accepts a manifest of preselected pairs for this protocol.

## Semantic assessment

The [archived appendix prompt](appendix_prompt.txt) reproduces Supplementary Material Section II with normalized line breaks. The [execution prompt](prompt_template.txt) changes the positive-increase wording to a direction-neutral perturbation and removes the requirement that an attribute become more visible. The execution prompt asks GPT-5.4 to identify the attribute or object part that changes visibly and to ignore incidental artifacts. Dataset context identifies the object domain without prescribing a concept name. Predicted-class metadata is appended when supplied, following the input description in the main manuscript.

For one pair, the request uses the execution prompt and explicit image labels. For multiple pairs, it additionally uses [multisample_instruction.txt](multisample_instruction.txt). Every pair belongs to the same concept. GPT-5.4 compares A against B within each pair, summarizes recurring changes and records disagreements in `uncertainty`. It produces one structured description per concept in one API request. The implementation does not cluster names or vote over separately generated labels.

## Recorded output

The schema preserves all eleven appendix fields. `confidence` accepts `high`, `medium` and `low`; the appendix shows `high` as an example and explicitly calls for lower confidence when the evidence is unclear. The release validates field types and the concept identifier before saving a completed result.

The request archive preserves full prompt text and checksums of both source files and normalized submitted images. It records the model snapshot and inference settings. The response archive preserves the returned JSON, model identifier, usage and request identifiers. Cached results require an exact request checksum match.

## Interpretation

A semantic name is a concise description of observed intervention effects. It does not establish a ground-truth attribute annotation or guarantee disentanglement. A single-pair result describes the supplied sample; multiple samples support inspection of recurring effects. Ambiguous and inconsistent results remain visible through the confidence and uncertainty fields.

## Workflow assets

![Concept naming workflow](workflow.png)

The workflow uses one sample from Fig. 4a of the supplied manuscript. The original and perturbed reconstructions correspond to concept 82. The “Red plumage” label follows that figure; it is not a new API result. The raw input photograph was not supplied, so its diagram node reuses the original reconstruction as a schematic illustration. No reconstruction-fidelity claim follows from this reuse. The contribution bars are schematic and do not report measured scores.

The diagram stacks the original and perturbed reconstructions vertically. Two separate horizontal arrows enter GPT-5.4, matching the two image inputs in one script request. The concept dictionary maps concept identifiers to names; `names.csv` and `result.json` preserve the generated records, including confidence and evidence.

The [PowerPoint](workflow.pptx) contains editable text, shapes, a concept dictionary table and connectors. The [PDF](workflow.pdf) provides a static version for academic documents.
