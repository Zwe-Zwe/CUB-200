// COS30082 Assignment 1 report. Build:  python report/build.py   (-> report/report.pdf)
// Edit the values below before submitting.
#let student-name = "Zwe Htet Zaw"
#let student-id = "102783659"
#let youtube = "https://youtu.be/-hWIFqhMIh4"

#import "data/tables.typ": *
#set document(title: "COS30082 Assignment 1: Bird Species Classification", author: student-name)
#set page(paper: "a4", margin: (x: 1.9cm, top: 1.6cm, bottom: 1.6cm),
          numbering: "1", number-align: center)
#set text(font: "Carlito", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.55em, spacing: 0.75em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1em, below: 0.55em, text(size: 13pt, it))
#show heading.where(level: 2): it => block(above: 0.85em, below: 0.4em, text(size: 11pt, it))
#show link: it => underline(offset: 1.5pt, stroke: 0.4pt, it)
#set table(stroke: 0.5pt + luma(170), inset: (x: 4pt, y: 2.6pt))
#show table: set text(size: 9pt)
#show table.cell.where(y: 0): set text(weight: "bold")
#show figure.caption: set text(size: 9pt)
#set figure(gap: 0.45em)
#show figure: set block(above: 0.8em, below: 0.8em)

// ------------------------------------------------------------------------------------------- cover + contents
#set page(numbering: none)
#v(5cm)
#align(center)[
  #text(size: 12pt)[COS30082 Applied Machine Learning] \
  #v(0.3em)
  #text(size: 20pt, weight: "bold")[Assignment 1: Bird Species Classification] \
  #v(0.2em)
  #text(size: 12pt)[CUB-200, fine-tuning ResNet-50 and ConvNeXt-Tiny]
  #v(2.5cm)
  #text(size: 12pt)[*Name:* #student-name \ *Student ID:* #student-id]
]
#v(2.5cm)
#show table.cell.where(y: 0): set text(weight: "regular")   // cover table has no header row
#table(columns: (auto, 1fr), stroke: none, inset: (x: 4pt, y: 4pt),
  [*Video presentation (YouTube):*], [#link(youtube)],
  [*Hugging Face (trained model):*], [#link("https://huggingface.co/zwe-zwe/CUB-200")],
  [*Web application:*], [#link("https://cub200-birds.streamlit.app/")],
  [*GitHub repository:*], [#link("https://github.com/Zwe-Zwe/CUB-200")],
)
#pagebreak()
#show table.cell.where(y: 0): set text(weight: "bold")
#outline(title: "Contents", depth: 2, indent: 1.2em)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)

= Methodology

== Dataset and how I split it
The dataset (CUB-200; Welinder et al. 2010) has 4,829 training images and 1,204 test images of 200 bird species.
That is only about 24 training images per species (between 16 and 31), so I expected overfitting to be the main
problem from the start. While loading the data I also found 91 file names that broke the naming pattern and
gave wrong species names, so I fixed the parsing and added a check.

With so few images, it was tempting to use the test set for validation, but then every tuning decision would be
fitted to the test set and the final score would be too optimistic. So I kept the test set aside until the end and
split the training images 80/20 into a training set (3,863 images)
and a validation set (966 images), stratified so that every species appears in both. I used validation accuracy for
every decision: when to stop training, which checkpoint to keep, and whether a change to the setup was worth
keeping. Once the setup was final, I retrained it on all 4,829 training images and evaluated it on the test set once.

I report the two metrics required by the brief. Top-1 accuracy is the share of test images whose highest-scoring
class is the correct one. Average accuracy per class is the accuracy for each species, averaged over the 200
species, so every species counts equally. I also report Top-5 accuracy (the correct species is among the five most
likely) for the final models. When two results were close, I used McNemar's test (McNemar 1947). It looks at the
images that one model gets right and the other gets wrong, and checks whether the difference could be due to chance.

== Preprocessing and data augmentation
For evaluation, each image is resized (shorter side 256 pixels for a 224-pixel input, or 512 for 448), centre-cropped
and normalised with the ImageNet mean and standard deviation, which is what the pretrained models expect. For training
I compared two augmentation settings (Table 1). Augmentation shows the model a slightly different version of each
image in every epoch, which makes it harder to simply memorise the training images.

#figure(
  table(columns: (auto, 1.25fr, 1fr), align: left,
    [Setting], [Transformations], [Reason],
    [Basic], [Random crop covering 60--100% of the image, horizontal flip, rotation up to 15°, colour jitter,
      random erasing],
    [Varies position, size, angle and lighting while keeping most of the bird visible. Erasing imitates birds
      partly hidden by branches.],
    [Strong], [Random crop covering 25--100%, horizontal flip, TrivialAugmentWide,
      random erasing],
    [Much more variation, for a model that still overfits. Small crops can cut off the parts that identify a
      species.],
  ),
  caption: [The two augmentation settings. Flipping is safe because birds look the same mirrored.],
)

== Models
I chose ResNet-50 because it is the classic CNN design covered in the lectures, and ConvNeXt-Tiny because it is a
recent CNN of almost the same size. Both are pretrained on ImageNet-1k, so neither has an advantage from larger
pretraining data, and the comparison shows how much a modern design helps.

- *ResNet-50* (He et al. 2016), 23.9 million parameters. It is built from residual blocks, where a shortcut
  connection adds a block's input to its output. This makes very deep networks much easier to train. It uses
  BatchNorm after each convolution.
- *ConvNeXt-Tiny* (Liu et al. 2022), 28.0 million parameters. It is a ResNet updated with ideas from vision
  transformers: larger 7×7 convolutions, LayerNorm instead of BatchNorm, and the GELU activation.

For both models I replaced the original 1,000-class ImageNet output layer with a dropout layer and a new linear
layer for the 200 bird species.

== Loss function and training setup
I used cross-entropy loss with label smoothing of 0.1 (Szegedy et al. 2016). Instead of asking the model to give
the correct class 100% probability, the target becomes 90% for the correct class, with the remaining 10% spread over
the other classes. Many species in this dataset look almost identical, so this stops the model from becoming
overconfident.

The optimiser is AdamW (Loshchilov & Hutter 2019). The setting that mattered most was the learning rate. The
pretrained layers (the "backbone") get a learning rate ten times smaller than the new output layer (the "head"). The
useful ImageNet features are then adjusted only gently, while the new layer learns quickly.
The learning rate warms up during the first epoch and then follows a cosine curve down to zero.

== How I tuned the models
I started from a simple baseline and changed one thing at a time. I kept a change only if validation accuracy
improved, and reverted it otherwise. Both models went through the same steps (Table 4), so they got the same amount
of tuning. Table 2 lists the final settings.

#figure(
  table(columns: (1.35fr, 1fr, 1fr), align: (left, center, center),
    [Setting], [ResNet-50], [ConvNeXt-Tiny],
    [Image size, epochs, batch size], table.cell(colspan: 2)[30 epochs at 224 px (batch 32), then 10 epochs at
      448 px (batch 16)],
    [Learning rate, backbone / head], table.cell(colspan: 2)[0.0001 / 0.001, then 0.00002 / 0.0002 at 448 px],
    [Optimiser and schedule], table.cell(colspan: 2)[AdamW, 1-epoch warm-up, cosine decay],
    [Weight decay / label smoothing], table.cell(colspan: 2)[0.05 / 0.1],
    [Dropout / stochastic depth], table.cell(colspan: 2)[0.3 / 0.1],
    [Augmentation], [Strong], [Basic],
    [Final training data], table.cell(colspan: 2)[All 4,829 training images (train + validation)],
  ),
  caption: [Final settings. The only difference between the two models is the augmentation, chosen on validation.],
)

= Results and Discussion

== Overall results
#figure(
  table(columns: (2.2fr, 0.75fr, 0.7fr, 0.7fr, 0.8fr, 0.85fr, 0.7fr),
    align: (left, right, right, right, right, right, right),
    [Model], [Params], [Train], [Val], [Test Top-1], [Avg/class], [Top-5],
    [ResNet-50, baseline (L0)], [23.9 M], [95.4], [65.9], [66.2], [65.4], [88.2],
    [ConvNeXt-Tiny, baseline (L0)], [28.0 M], [98.7], [75.6], [74.0], [73.6], [92.2],
    [ResNet-50, tuned (L6)], [23.9 M], [100.0], [84.6], [83.0], [82.4], [96.2],
    [ConvNeXt-Tiny, tuned (L6)], [28.0 M], [99.9], [89.4], [88.0], [87.6], [97.3],
    [ResNet-50, final], [23.9 M], [99.9], [--], [86.46], [86.12], [96.76],
    [*ConvNeXt-Tiny, final*], [28.0 M], [99.9], [--], [*90.20*], [*89.94*], [*98.01*],
  ),
  caption: [Accuracy (%). Final models were trained on train + validation, so they have no validation score.],
)

As Table 3 shows, my best model, the final ConvNeXt-Tiny, reaches 90.20% Top-1 and 89.94% average per-class accuracy on the test set.
The two metrics are very close for both models, so the accuracy is spread evenly across species and does not come
from a few easy ones. The final ConvNeXt-Tiny classifies 125 of the 200 species perfectly, and no species falls
below 25%.

== Tuning step by step
#figure(image("figures/ladder.png", width: 84%),
  caption: [Test Top-1 accuracy after each tuning step. The hollow point (L4) was reverted for ConvNeXt-Tiny.],
)

#let up(x) = x
#let dn(x) = x
#figure(
  table(columns: (auto, 2.2fr, 0.95fr, 0.85fr, 0.95fr, 0.85fr), align: (left, left, right, left, right, left),
    [Step], [Change], [ResNet-50 val], [Decision], [ConvNeXt val], [Decision],
    [L0], [Baseline: only train the new output layer], [65.9], [--], [75.6], [--],
    [L1], [Train all layers, one learning rate (0.001)], [60.7 #dn[(−5.3)]], [Worse], [3.9 #dn[(−71.6)]], [Worse],
    [L2], [Separate learning rates, warm-up + cosine], [80.0 #up[(+14.1)]], [Kept], [86.7 #up[(+11.1)]], [Kept],
    [L3], [Regularisation], [81.4 #up[(+1.4)]], [Kept], [86.9 #up[(+0.2)]], [Kept],
    [L4], [Strong augmentation], [82.5 #up[(+1.1)]], [Kept], [86.5 #dn[(−0.3)]], [Reverted],
    [L5], [40 epochs instead of 20], [82.9 #up[(+0.4)]], [Kept], [87.5 #up[(+0.6)]], [Kept],
    [L6], [Extra 10 epochs at 448 px], [84.6 #up[(+1.7)]], [Kept], [89.4 #up[(+2.0)]], [Kept],
    [L8], [Model-specific optimiser settings], [84.5 #dn[(−0.1)]], [Not used], [89.8 #up[(+0.3)]], [Not used],
  ),
  caption: [Validation accuracy (%) behind each decision. L7 is the L6 setup retrained on all training images.],
)

*L0, baseline* (Figure 1 shows every step). I froze the pretrained network and trained only the new output layer. This gave 66.2% (ResNet-50)
and 74.0% (ConvNeXt-Tiny) on the test set. The ImageNet features are useful, but they were never trained to tell
bird species apart.

*L1, training all layers.* I expected unfreezing everything to help, but with one learning rate of 0.001 it made
things worse: ConvNeXt-Tiny collapsed to 4.4% in the first epoch and ResNet-50 dropped to 59.8%. A learning rate that suits a new layer is far
too large for pretrained weights and wipes out what they had learned, which is known as catastrophic forgetting.

*L2, separate learning rates.* Using 0.0001 for the backbone and 0.001 for the head, with warm-up and cosine decay,
fixed this and gave the biggest improvement in the project (77.5% and 86.1%). However, training accuracy was now
99.9%, so both models were overfitting.

*L3, regularisation.* I added label smoothing, dropout of 0.3 before the output layer, stochastic depth of 0.1
(randomly skipping residual blocks during training) and a higher weight decay of 0.05. ResNet-50
improved by 3 points and ConvNeXt-Tiny stayed the same, so I kept these for both.

*L4, stronger augmentation.* This helped ResNet-50 but hurt ConvNeXt-Tiny, which also lost 2.4 points on the test
set. Crops as small as a quarter of the image often cut off the features that separate two species. From this step
on, the two models use different augmentation.

*L5, longer training.* 40 epochs instead of 20 gave a small gain for both models.

*L6, higher resolution.* I trained for 30 epochs at 224 pixels, then 10 more at 448 pixels with a lower learning
rate (and a batch size of 16, as 32 images at 448 pixels did not fit in my laptop GPU's 8 GB). Validation accuracy rose by about 2 points for both models. The details that separate similar species, such as
eye rings or wing bars, are only a few pixels wide at 224 pixels. I also tried test-time augmentation (averaging the
prediction with that of a mirrored image), but it made no difference on validation, so I left it out.

*L7, final training.* I retrained the L6 setup on all 4,829 training images. This added 2.2 points for ConvNeXt-Tiny
and 3.5 points for ResNet-50 on the test set.

*L8, a fairness check.* The shared setup might suit one model better than the other, so I also tried each model's
usual best practice: SGD with momentum for ResNet-50 (at three learning rates) and layer-wise learning-rate decay for
ConvNeXt-Tiny. Neither beat L6 on validation, so I kept the L6 setup for both.

Table 5 adds up what each kept change contributed on the test set. Getting the learning rates right gave more than
half of the total improvement for both models.

#figure(
  table(columns: (2.4fr, 1fr, 1fr), align: (left, right, right),
    [Change], [ResNet-50], [ConvNeXt-Tiny],
    [Separate learning rates + warm-up and cosine (L0 → L2)], [+11.3], [+12.1],
    [Regularisation (L3)], [+3.0], [0.0],
    [Strong augmentation (L4)], [+0.3], [not kept],
    [Longer training (L5)], [+1.0], [−0.2],
    [Higher resolution, 448 px (L6)], [+1.2], [+2.1],
    [Training on all training images (L7)], [+3.5], [+2.2],
    [*Total, baseline → final*], [*+20.3*], [*+16.2*],
  ),
  caption: [Change in test Top-1 accuracy (percentage points) from each kept step. Decisions were made on
    validation, so small steps can be slightly negative on test (ConvNeXt-Tiny L5).],
)

== Dealing with overfitting
Overfitting was the main challenge, as the brief predicted. Both models reach almost 100% training accuracy, which
is expected when a model with 24 to 28 million parameters sees only about 19 images per species. I could not remove
the gap completely, but I made it smaller while validation accuracy went up. From L2 to L6, the gap between training
and validation accuracy fell from 19.8 to 15.4 points for ResNet-50 and from 13.2 to 10.4 points for ConvNeXt-Tiny.

Figure 2 shows this in the training curves. Without regularisation (a), the model fits the training images within
four epochs and validation accuracy stops improving at about 86%. With the final setup (b), validation accuracy keeps
climbing and jumps when the image size doubles. With strong augmentation (c), ResNet-50 only reaches about 90% on the
augmented training images, so it has a much harder time memorising them.

#figure(image("figures/curves.png", width: 100%),
  caption: [Training accuracy (on augmented images) and validation accuracy per epoch. The dashed line marks the
    switch from 224 to 448 pixels.],
)

I stopped adding regularisation once it started to hurt, as strong augmentation did for ConvNeXt-Tiny.

== Comparing the models and choosing the best one
ConvNeXt-Tiny was better than ResNet-50 at every step except L1, where it collapsed. On the test set it gets 75 images right that
ResNet-50 gets wrong, while the opposite happens for only 30 images. McNemar's test gives p = 0.00001, so this
difference is very unlikely to be chance. Both models had the same pretraining data, a similar size and the same
tuning, and the model-specific settings in L8 did not change the result. I therefore think the difference comes from
the architecture itself, which matches Liu et al. (2022), who found that ConvNeXt outperforms ResNets of similar
size.

The two models also behaved differently during tuning. ConvNeXt-Tiny's ImageNet features were better for birds from
the start: its frozen baseline was 7.8 points ahead of ResNet-50's. It also learned faster (its best L2 epoch was 10
against 16 for ResNet-50) and overfitted less, with a train-validation gap of 13.2 points at L2 against 19.8 for
ResNet-50. That explains why regularisation helped ResNet-50 by 3 points but did nothing for ConvNeXt-Tiny, and why
only ResNet-50 benefited from strong augmentation. This fits the architectures: ConvNeXt's 7×7 filters see more of
the bird at once than ResNet's 3×3 filters, which helps when two species differ by a pattern across the wing or head. ConvNeXt-Tiny is better on 54 species and worse
on only 20, and its Top-5 accuracy of 98.01% means that even when it is wrong, the right answer is almost always among
its top candidates.

I chose *ConvNeXt-Tiny* as my final model: 90.20% Top-1, 89.94% average per-class and 98.01% Top-5 accuracy, and it
powers the web app. For comparison, a much larger ConvNeXt-Base pretrained on ImageNet-22k reached
93.69% in an early test without any tuning, so larger pretraining data looks like the clearest way to improve further.

== Where the model still makes mistakes
Most of the final model's 118 mistakes are between species that look almost the same (Figure 3), such as
Boat-tailed Grackle and Brewer's Blackbird, two cormorants, or Fish Crow and American Crow, which even bird watchers
usually tell apart only by their call. ResNet-50 struggles with the same species, which suggests these errors come
from the data rather than from one particular model. The model's confidence is also useful: on average it is 72%
confident when it is right and 29% when it is wrong, so the web app shows a warning when confidence is below 30%.

#figure(image("figures/misclassified_examples.png", width: 63%),
  caption: [The most common mistakes of the final model. Top: the misclassified test image. Bottom: a training image
    of the species it was mistaken for.],
)

== Limitations and future work
All results come from a single training run per setting, so repeating the runs with different random seeds would
show how much the numbers vary. With 1,204 test images, the Top-1 accuracy itself is only precise to about ±1.7
points (Appendix A.3).
I also checked the data for leakage: 4 test images are exact copies of training images, but each copy has a
different label in the two files, so both models count them as mistakes (Appendix A.5). Without them, ConvNeXt-Tiny
would score 90.50%. With only 4 to 8 test images per species, the per-species accuracies are rough. In
future work I would try combining the two models, using models that focus on specific bird parts, and using a model
pretrained on more data.

= Conclusion
I fine-tuned two pretrained CNNs for 200-class bird classification and improved them step by step. Changing one
setting at a time, and checking every change on a validation set instead of the test set, raised
ConvNeXt-Tiny from a 74.0% baseline to 90.20% Top-1 (89.94% average per class) and ResNet-50 from 66.2% to 86.46%.

The biggest lesson for me was how much the learning rate matters. With one learning rate for the whole network,
fine-tuning destroyed the pretrained features, while a small rate for the backbone and a larger one for the new
layer gave more than half of the total improvement. Overfitting could not be avoided with about 24 images per
species, but label smoothing, dropout, stochastic depth, weight decay and augmentation kept it under control, as long
as the amount of augmentation suited the model. Training at 448 pixels helped both models see the fine details, and
retraining on the validation images at the end added another 2 to 3.5 points.

ConvNeXt-Tiny is my best model: it is significantly more accurate than ResNet-50 under the same conditions, and a
complete re-run reproduced every result exactly (Appendix A.9).

#pagebreak()
#heading(numbering: none)[References]
#[
#set par(hanging-indent: 1.2em, spacing: 0.55em, justify: false)
#text(size: 9.5pt)[
He, K, Zhang, X, Ren, S & Sun, J 2016, 'Deep residual learning for image recognition', in _Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition_, pp. 770--778.

Liu, Z, Mao, H, Wu, C-Y, Feichtenhofer, C, Darrell, T & Xie, S 2022, 'A ConvNet for the 2020s', in _Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 11976--11986.

Loshchilov, I & Hutter, F 2019, 'Decoupled weight decay regularization', in _International Conference on Learning
Representations_.

McNemar, Q 1947, 'Note on the sampling error of the difference between correlated proportions or percentages',
_Psychometrika_, vol. 12, no. 2, pp. 153--157.

Szegedy, C, Vanhoucke, V, Ioffe, S, Shlens, J & Wojna, Z 2016, 'Rethinking the Inception architecture for computer
vision', in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, pp. 2818--2826.

Welinder, P, Branson, S, Mita, T, Wah, C, Schroff, F, Belongie, S & Perona, P 2010, _Caltech-UCSD Birds 200_,
Technical Report CNS-TR-2010-001, California Institute of Technology, Pasadena.
]
]

// =========================================================================================== APPENDIX
#pagebreak()
#set page(numbering: (..n) => "A" + str(n.pos().first()))
#counter(page).update(1)
#set heading(numbering: (..n) => "A." + n.pos().slice(1).map(str).join("."))
#show raw.where(block: true): it => block(width: 100%, inset: 6pt, stroke: 0.5pt + luma(170),
  text(size: 7.6pt, it))

#heading(numbering: none)[Appendix: supporting evidence]
#set heading(outlined: false)
#text(size: 9pt)[(not part of the 4-page report)]

This appendix gives the full data behind the report. Every table is generated from the saved results by the scripts
in the `report/` folder of the code repository, so the numbers match the training logs exactly.

== All tuning runs
#figure(
  table(columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, left, right, right, right, right, right),
    [Step], [Model], [Config file], [Train], [Val], [Test Top-1], [Avg/class], [Best epoch],
    ..all-runs),
  caption: [Every run of the tuning ladder (accuracy in %, no test-time augmentation). Train is the accuracy on the
    training images without augmentation. "Best epoch" is the checkpoint kept (chosen on validation; the final runs
    keep the last epoch).],
)

== Statistical tests for each decision
McNemar's exact test compares two models on the same images. "Only new / only old" counts the validation images
that only the new or only the old setting classifies correctly. Decisions used the validation columns only; the test
columns are shown afterwards as a check.

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, center, center, right, center, right),
    [Model], [Comparison], [Val acc.], [Only new / only old], [Val p], [Test acc.], [Test p],
    ..decisions),
  caption: [McNemar tests for every ladder decision.],
)

The large steps (L1, L2) are clearly significant. The later steps (L3 to L6) each improved validation accuracy by
only 0.2 to 2 points, which is not significant on its own with 966 validation images. Taken together, however, they
are significant on validation for both models (last two rows), so the gain from L2 to L6 is real.

== Precision of the final results
#figure(
  table(columns: (auto, auto, auto, auto), align: (left, right, center, center),
    [Final model], [Test Top-1], [95% interval, Top-1], [95% interval, avg/class],
    ..cis),
  caption: [Bootstrap 95% confidence intervals (2,000 resamples of the 1,204 test images).],
)

On the test set, ConvNeXt-Tiny is right on #final-mcnemar.c images where ResNet-50 is wrong, and ResNet-50 on
#final-mcnemar.r images where ConvNeXt-Tiny is wrong (McNemar p = #final-mcnemar.p). The intervals of the two
models do not overlap.

== Per-species results
#figure(image("figures/per_class_hist.png", width: 70%),
  caption: [Number of species in each accuracy range on the test set. With 4 to 8 test images per species, the
    possible accuracies below 100% are coarse (for example 7/8 = 87.5%).],
)

#grid(columns: (1.25fr, 1fr), gutter: 10pt,
  figure(
    table(columns: (auto, auto, auto, auto), align: (left, right, left, right),
      [ConvNeXt-Tiny], [Acc. (n)], [ResNet-50], [Acc. (n)],
      ..worst),
    caption: [The 10 hardest species for each final model (test accuracy %, number of test images).],
  ),
  figure(
    table(columns: (auto, auto, auto), align: (left, left, right),
      [True species], [Predicted as], [Count],
      ..confusions),
    caption: [Most frequent confusions of the final ConvNeXt-Tiny.],
  ),
)

== Data checks
#figure(
  table(columns: (auto, auto), align: (left, left),
    [Check], [Result],
    [Images per species (training)], [16 to 31, mean 24.1],
    [Images per species (test)], [4 to 8],
    [Train / validation split], [3,863 / 966, stratified by species, random seed 42],
    [Labels], [Each label maps to exactly one species name (checked in code)],
    [Exact duplicates between Train and Test (MD5 hash)], [4 images (listed below)],
    [Near duplicates (256-bit difference hash, ≤ 10 bits)], [no further images],
  ),
  caption: [Checks on the data.],
)

#figure(
  table(columns: (auto, auto, auto, auto), align: left,
    [Test image], [Label in test.txt], [Identical training image (its label)], [Prediction of both models],
    [0197.jpg], [Purple Finch], [American_Goldfinch_0013 (American Goldfinch)], [American Goldfinch],
    [0273.jpg], [American Goldfinch], [Indigo_Bunting_0006 (Indigo Bunting)], [Indigo Bunting],
    [0918.jpg], [Red-eyed Vireo], [Tennessee_Warbler_0012 (Tennessee Warbler)], [Tennessee Warbler],
    [0976.jpg], [Cape May Warbler], [Tennessee_Warbler_0020 (Tennessee Warbler)], [Tennessee Warbler],
  ),
  caption: [The 4 test images that also appear in the training set. The labels disagree, and both models predict
    the training label, so these images lower the test score (90.20% with them, 90.50% without).],
)

== Learning-rate schedule
#figure(image("figures/lr_schedule.png", width: 62%),
  caption: [Backbone learning rate of the final ConvNeXt-Tiny recipe, as logged during training: 1-epoch warm-up
    and cosine decay over 30 epochs at 224 px, then a new, 5× smaller cosine cycle for the 10 epochs at 448 px. The
    head always uses a 10× larger learning rate.],
)

== Other things I tried
*Test-time augmentation (TTA).* For every run I also evaluated a flip TTA: the prediction on the image is averaged
with the prediction on its mirror image. The decision was made once, on validation, for the final recipe (L6): TTA
gave exactly the same validation accuracy for both models, so it was left off because it doubles the prediction time.
TTA helped ResNet-50 at some earlier steps, but by L6 the gain had gone. The test columns are only a check; for the
final ConvNeXt-Tiny, TTA changes nothing (90.20% either way).

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto), align: (left, left, right, right, right, right),
    [Step], [Model], [Val, no TTA], [Val, TTA], [Test, no TTA], [Test, TTA],
    ..tta),
  caption: [Top-1 accuracy (%) with and without flip TTA for every run. Final runs have no validation score.],
)

*Early exploration before the ladder.* Before designing the step-by-step ladder, I tried two fixed recipes on four
pretrained models: recipe A (224 px, the same settings as ladder step L3) and recipe B (15 epochs at 224 px, then 5 at
448 px, with CutMix on half of the training batches). The two larger-pretraining models (ImageNet-21k / 22k) showed
how much more pretraining data helps, and ConvNeXt-Base became the reference line in Figure 1. CutMix was not used in
the ladder: the ladder's L6 setting without CutMix reached higher validation accuracy for both models (89.4% vs 87.2%
for ConvNeXt-Tiny, 84.6% vs 81.1% for ResNet-50). Recipe B also trained for fewer epochs, so this comparison does not
isolate CutMix by itself.

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto, auto), align: (left, left, right, left, right, right, right),
    [Model], [Pretraining], [Params], [Recipe], [Val Top-1], [Test Top-1], [Avg/class],
    ..early),
  caption: [Early runs with two fixed recipes (accuracy in %, no TTA).],
)

== Web application
#figure(image("figures/webapp.jpg", width: 85%),
  caption: [The deployed web app (#link("https://cub200-birds.streamlit.app/")[cub200-birds.streamlit.app]) with
    one of the example test images. The model runs on the server CPU; the weights are downloaded from the Hugging
    Face model repository.],
)

== Key code
The full, commented code is in the repository (`birdcls/` package). These are the parts that implement the
decisions described in the report.

*Stratified train / validation split* (`birdcls/data.py`):
```python
# Stratified split so every class appears in both train and validation
train, val = train_test_split(full, test_size=0.2, stratify=full.label, random_state=42)
```

*New output layer* (`birdcls/models.py`), shown for ConvNeXt-Tiny from torchvision:
```python
m = models.convnext_tiny(weights="IMAGENET1K_V1", stochastic_depth_prob=drop_path)
# classifier = [LayerNorm2d, Flatten, Linear(768 -> 1000)]; replace the last layer
in_features = m.classifier[2].in_features
m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
```

*Separate learning rates for backbone and head, and the warm-up + cosine schedule* (`birdcls/engine.py`):
```python
def make_optimizer(cfg, backbone, head, lr_backbone, lr_head):
    groups = [{"params": head, "lr": lr_head}]                 # new layer: large LR
    if not cfg.freeze_backbone:
        groups.insert(0, {"params": backbone, "lr": lr_backbone})   # pretrained layers: 10x smaller LR
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

def warmup_cosine(optimizer, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:                     # linear warm-up
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))   # cosine decay to 0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
```

*Training step with label smoothing and mixed precision* (`birdcls/engine.py`, simplified):
```python
criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)    # 0.1
for phase in cfg.phases:                     # 224 px phase, then 448 px phase (progressive resizing)
    optimizer = make_optimizer(cfg, backbone, head, phase.lr_backbone, phase.lr_head)
    scheduler = warmup_cosine(optimizer, phase.epochs * steps, phase.warmup_epochs * steps)
    for epoch in range(phase.epochs):
        model.train()                        # enables dropout and stochastic depth
        for imgs, labels in train_loader:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader)   # stopping + checkpoint use val_acc
```

*The two required metrics* (`birdcls/metrics.py`):
```python
top1 = float((y_true == y_pred).mean())                    # Top-1 accuracy
per_class = [(y_pred[y_true == c] == c).mean() for c in range(num_classes)]
ave = float(np.nanmean(per_class))                         # average accuracy per class
```

*McNemar's exact test* (`report/appendix_stats.py`):
```python
a_only = ((a == y) & (b != y)).sum()       # images only model A gets right
b_only = ((a != y) & (b == y)).sum()       # images only model B gets right
p = binomtest(a_only, a_only + b_only, 0.5).pvalue
```

*Final ConvNeXt-Tiny configuration* (`configs/tl_L6_hires.yaml`), trained with `--full-train`:
```yaml
optimizer: adamw
lr_schedule: cosine
augment: basic
label_smoothing: 0.1
weight_decay: 0.05
dropout: 0.3
drop_path: 0.1
patience: 10
phases:
  - {img_size: 224, batch_size: 32, epochs: 30, lr_backbone: 1.0e-4, lr_head: 1.0e-3, warmup_epochs: 1}
  - {img_size: 448, batch_size: 16, epochs: 10, lr_backbone: 2.0e-5, lr_head: 2.0e-4, warmup_epochs: 0}
```

== Reproducibility
Software: Python 3.12, PyTorch 2.14 (CUDA 13.0), torchvision 0.29, timm 1.0.30. Hardware: one NVIDIA RTX 4060
laptop GPU (8 GB). Random seed 42 for the split and training.

To check that the results are not a one-off, I retrained all 21 runs of the ladder (both models, L0 to L8) from the
beginning with the same code and settings. Every run reproduced exactly: the same validation and test accuracy, the
same best epoch, and the same prediction for each of the 1,204 test images. Training the final model and evaluating
it:
```bash
python -m birdcls.train --model convnext_tiny --config tl_L6_hires --full-train
python -m birdcls.evaluate --runs convnext_tiny__tl_L6_hires__full
python -m birdcls.evaluate --analyse convnext_tiny__tl_L6_hires__full
```
