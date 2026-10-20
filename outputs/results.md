| run | model | config | tta | params_M | eval_size | best_epoch | train_acc | val_top1 | gap | test_top1 | test_avg_per_class | train_minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_base.fb_in22k__A_standard | convnext_base.fb_in22k | A_standard | False | 87.8 | 224 | 11 | 0.9992 | 0.9317 | 0.0675 | 0.9319 | 0.9296 | 10.6507 |
| convnext_base.fb_in22k__A_standard +TTA | convnext_base.fb_in22k | A_standard | True | 87.8 | 224 | 11 | 0.9992 | 0.9337 | 0.0655 | 0.9302 | 0.9288 | 10.6507 |
| convnext_base.fb_in22k__B_finegrained | convnext_base.fb_in22k | B_finegrained | False | 87.8 | 448 | 18 | 0.999 | 0.9462 | 0.0528 | 0.9369 | 0.9355 | 19.631 |
| convnext_base.fb_in22k__B_finegrained +TTA | convnext_base.fb_in22k | B_finegrained | True | 87.8 | 448 | 18 | 0.999 | 0.9441 | 0.0549 | 0.936 | 0.9347 | 19.631 |
| convnext_tiny__A_standard | convnext_tiny | A_standard | False | 28.0 | 224 | 17 | 0.999 | 0.8685 | 0.1304 | 0.8613 | 0.8571 | 6.9346 |
| convnext_tiny__A_standard +TTA | convnext_tiny | A_standard | True | 28.0 | 224 | 17 | 0.999 | 0.8665 | 0.1325 | 0.8688 | 0.8668 | 6.9346 |
| convnext_tiny__B_finegrained | convnext_tiny | B_finegrained | False | 28.0 | 448 | 19 | 0.9984 | 0.8716 | 0.1268 | 0.8663 | 0.8639 | 10.1748 |
| convnext_tiny__B_finegrained +TTA | convnext_tiny | B_finegrained | True | 28.0 | 448 | 19 | 0.9984 | 0.8644 | 0.1341 | 0.8704 | 0.8686 | 10.1748 |
| resnet50.tv2_in1k__A_standard | resnet50.tv2_in1k | A_standard | False | 23.9 | 224 | 20 | 0.999 | 0.8137 | 0.1853 | 0.8048 | 0.7987 | 4.8187 |
| resnet50.tv2_in1k__A_standard +TTA | resnet50.tv2_in1k | A_standard | True | 23.9 | 224 | 20 | 0.999 | 0.823 | 0.176 | 0.8164 | 0.8122 | 4.8187 |
| resnet50.tv2_in1k__B_finegrained | resnet50.tv2_in1k | B_finegrained | False | 23.9 | 448 | 20 | 0.9915 | 0.8106 | 0.1809 | 0.799 | 0.7924 | 7.9526 |
| resnet50.tv2_in1k__B_finegrained +TTA | resnet50.tv2_in1k | B_finegrained | True | 23.9 | 448 | 20 | 0.9915 | 0.8085 | 0.183 | 0.804 | 0.7976 | 7.9526 |
| resnetv2_50x1_bit.goog_in21k__A_standard | resnetv2_50x1_bit.goog_in21k | A_standard | False | 23.9 | 224 | 17 | 0.9987 | 0.8623 | 0.1364 | 0.8571 | 0.8541 | 7.4355 |
| resnetv2_50x1_bit.goog_in21k__A_standard +TTA | resnetv2_50x1_bit.goog_in21k | A_standard | True | 23.9 | 224 | 17 | 0.9987 | 0.8592 | 0.1395 | 0.8555 | 0.8529 | 7.4355 |
| resnetv2_50x1_bit.goog_in21k__B_finegrained | resnetv2_50x1_bit.goog_in21k | B_finegrained | False | 23.9 | 448 | 19 | 0.9966 | 0.8789 | 0.1178 | 0.8671 | 0.8649 | 15.5721 |
| resnetv2_50x1_bit.goog_in21k__B_finegrained +TTA | resnetv2_50x1_bit.goog_in21k | B_finegrained | True | 23.9 | 448 | 19 | 0.9966 | 0.882 | 0.1146 | 0.863 | 0.8611 | 15.5721 |
| scratch_vgg__scratch_base | scratch_vgg | scratch_base | False | 8.9 | 224 | 43 | 0.9858 | 0.0569 | 0.9288 | 0.049 | 0.0478 | 8.0953 |
| scratch_vgg__scratch_base +TTA | scratch_vgg | scratch_base | True | 8.9 | 224 | 43 | 0.9858 | 0.059 | 0.9268 | 0.0498 | 0.0486 | 8.0953 |
| scratch_vgg_bn__scratch_base | scratch_vgg_bn | scratch_base | False | 8.9 | 224 | 38 | 0.9997 | 0.1511 | 0.8486 | 0.1437 | 0.1381 | 9.0927 |
| scratch_vgg_bn__scratch_base +TTA | scratch_vgg_bn | scratch_base | True | 8.9 | 224 | 38 | 0.9997 | 0.1439 | 0.8558 | 0.1628 | 0.1567 | 9.0927 |
| scratch_vgg_bn__scratch_aug | scratch_vgg_bn | scratch_aug | False | 8.9 | 224 | 44 | 0.7712 | 0.2588 | 0.5124 | 0.2542 | 0.2462 | 9.0796 |
| scratch_vgg_bn__scratch_aug +TTA | scratch_vgg_bn | scratch_aug | True | 8.9 | 224 | 44 | 0.7712 | 0.2567 | 0.5144 | 0.2616 | 0.2525 | 9.0796 |
| scratch_vgg_gap__scratch_aug | scratch_vgg_gap | scratch_aug | False | 2.4 | 224 | 45 | 0.5918 | 0.3023 | 0.2895 | 0.3081 | 0.3004 | 8.7375 |
| scratch_vgg_gap__scratch_aug +TTA | scratch_vgg_gap | scratch_aug | True | 2.4 | 224 | 45 | 0.5918 | 0.3085 | 0.2833 | 0.3048 | 0.2961 | 8.7375 |
| scratch_vgg_gap__scratch_cosine | scratch_vgg_gap | scratch_cosine | False | 2.4 | 224 | 143 | 0.9982 | 0.4689 | 0.5292 | 0.4626 | 0.4531 | 26.2442 |
| scratch_vgg_gap__scratch_cosine +TTA | scratch_vgg_gap | scratch_cosine | True | 2.4 | 224 | 143 | 0.9982 | 0.4658 | 0.5323 | 0.4734 | 0.462 | 26.2442 |
| scratch_vgg_gap__scratch_wd | scratch_vgg_gap | scratch_wd | False | 2.4 | 224 | 139 | 0.9992 | 0.4803 | 0.5189 | 0.4809 | 0.4742 | 27.7635 |
| scratch_vgg_gap__scratch_wd +TTA | scratch_vgg_gap | scratch_wd | True | 2.4 | 224 | 139 | 0.9992 | 0.4803 | 0.5189 | 0.4792 | 0.4716 | 27.7635 |
| scratch_vgg_gap__scratch_strongaug | scratch_vgg_gap | scratch_strongaug | False | 2.4 | 224 | 135 | 0.9671 | 0.499 | 0.4682 | 0.4826 | 0.469 | 26.19 |
| scratch_vgg_gap__scratch_strongaug +TTA | scratch_vgg_gap | scratch_strongaug | True | 2.4 | 224 | 135 | 0.9671 | 0.4928 | 0.4744 | 0.4942 | 0.4809 | 26.19 |
| scratch_resnet18__scratch_strongaug | scratch_resnet18 | scratch_strongaug | False | 11.3 | 224 | 149 | 0.9317 | 0.4161 | 0.5155 | 0.4136 | 0.4064 | 19.8413 |
| scratch_resnet18__scratch_strongaug +TTA | scratch_resnet18 | scratch_strongaug | True | 11.3 | 224 | 149 | 0.9317 | 0.4089 | 0.5228 | 0.4161 | 0.4082 | 19.8413 |
| scratch_vgg_gap__scratch_strongaug__full | scratch_vgg_gap | scratch_strongaug | False | 2.4 | 224 | 150 | 0.9863 | nan | nan | 0.5864 | 0.576 | 27.4336 |
| scratch_vgg_gap__scratch_strongaug__full +TTA | scratch_vgg_gap | scratch_strongaug | True | 2.4 | 224 | 150 | 0.9863 | nan | nan | 0.5914 | 0.5816 | 27.4336 |
| resnet50.tv2_in1k__tl_L0_probe | resnet50.tv2_in1k | tl_L0_probe | False | 23.9 | 224 | 20 | 0.9542 | 0.6594 | 0.2948 | 0.662 | 0.6543 | 2.6791 |
| resnet50.tv2_in1k__tl_L0_probe +TTA | resnet50.tv2_in1k | tl_L0_probe | True | 23.9 | 224 | 20 | 0.9542 | 0.6843 | 0.2699 | 0.6711 | 0.6622 | 2.6791 |
| convnext_tiny__tl_L0_probe | convnext_tiny | tl_L0_probe | False | 28.0 | 224 | 18 | 0.9871 | 0.7557 | 0.2314 | 0.74 | 0.7357 | 2.7381 |
| convnext_tiny__tl_L0_probe +TTA | convnext_tiny | tl_L0_probe | True | 28.0 | 224 | 18 | 0.9871 | 0.7567 | 0.2303 | 0.745 | 0.7409 | 2.7381 |
| resnet50.tv2_in1k__tl_L1_finetune | resnet50.tv2_in1k | tl_L1_finetune | False | 23.9 | 224 | 14 | 0.9715 | 0.6066 | 0.3649 | 0.598 | 0.5885 | 4.5574 |
| resnet50.tv2_in1k__tl_L1_finetune +TTA | resnet50.tv2_in1k | tl_L1_finetune | True | 23.9 | 224 | 14 | 0.9715 | 0.6159 | 0.3556 | 0.6163 | 0.6081 | 4.5574 |
| convnext_tiny__tl_L1_finetune | convnext_tiny | tl_L1_finetune | False | 28.0 | 224 | 18 | 0.0593 | 0.0393 | 0.0199 | 0.044 | 0.0397 | 4.9484 |
| convnext_tiny__tl_L1_finetune +TTA | convnext_tiny | tl_L1_finetune | True | 28.0 | 224 | 18 | 0.0593 | 0.0383 | 0.021 | 0.0449 | 0.0405 | 4.9484 |
| resnet50.tv2_in1k__tl_L2_discriminative | resnet50.tv2_in1k | tl_L2_discriminative | False | 23.9 | 224 | 16 | 0.9984 | 0.8002 | 0.1982 | 0.7749 | 0.7664 | 4.508 |
| resnet50.tv2_in1k__tl_L2_discriminative +TTA | resnet50.tv2_in1k | tl_L2_discriminative | True | 23.9 | 224 | 16 | 0.9984 | 0.7981 | 0.2003 | 0.7857 | 0.7784 | 4.508 |
| convnext_tiny__tl_L2_discriminative | convnext_tiny | tl_L2_discriminative | False | 28.0 | 224 | 10 | 0.9987 | 0.8665 | 0.1322 | 0.8613 | 0.8587 | 3.9433 |
| convnext_tiny__tl_L2_discriminative +TTA | convnext_tiny | tl_L2_discriminative | True | 28.0 | 224 | 10 | 0.9987 | 0.8654 | 0.1333 | 0.858 | 0.8554 | 3.9433 |
| resnet50.tv2_in1k__tl_L3_regularise | resnet50.tv2_in1k | tl_L3_regularise | False | 23.9 | 224 | 20 | 0.999 | 0.8137 | 0.1853 | 0.8048 | 0.7987 | 4.7274 |
| resnet50.tv2_in1k__tl_L3_regularise +TTA | resnet50.tv2_in1k | tl_L3_regularise | True | 23.9 | 224 | 20 | 0.999 | 0.823 | 0.176 | 0.8164 | 0.8122 | 4.7274 |
| convnext_tiny__tl_L3_regularise | convnext_tiny | tl_L3_regularise | False | 28.0 | 224 | 17 | 0.999 | 0.8685 | 0.1304 | 0.8613 | 0.8571 | 5.0258 |
| convnext_tiny__tl_L3_regularise +TTA | convnext_tiny | tl_L3_regularise | True | 28.0 | 224 | 17 | 0.999 | 0.8665 | 0.1325 | 0.8688 | 0.8668 | 5.0258 |
| resnet50.tv2_in1k__tl_L4_strongaug | resnet50.tv2_in1k | tl_L4_strongaug | False | 23.9 | 224 | 20 | 0.999 | 0.8251 | 0.1739 | 0.8073 | 0.8017 | 4.6763 |
| resnet50.tv2_in1k__tl_L4_strongaug +TTA | resnet50.tv2_in1k | tl_L4_strongaug | True | 23.9 | 224 | 20 | 0.999 | 0.8333 | 0.1656 | 0.8223 | 0.8171 | 4.6763 |
| convnext_tiny__tl_L4_strongaug | convnext_tiny | tl_L4_strongaug | False | 28.0 | 224 | 10 | 0.9987 | 0.8654 | 0.1333 | 0.8372 | 0.8343 | 3.9638 |
| convnext_tiny__tl_L4_strongaug +TTA | convnext_tiny | tl_L4_strongaug | True | 28.0 | 224 | 10 | 0.9987 | 0.8696 | 0.1291 | 0.843 | 0.8422 | 3.9638 |
| resnet50.tv2_in1k__tl_L5_longer_strongaug | resnet50.tv2_in1k | tl_L5_longer_strongaug | False | 23.9 | 224 | 27 | 0.9997 | 0.8292 | 0.1705 | 0.8173 | 0.8129 | 8.5761 |
| resnet50.tv2_in1k__tl_L5_longer_strongaug +TTA | resnet50.tv2_in1k | tl_L5_longer_strongaug | True | 23.9 | 224 | 27 | 0.9997 | 0.8251 | 0.1747 | 0.8264 | 0.8234 | 8.5761 |
| convnext_tiny__tl_L5_longer | convnext_tiny | tl_L5_longer | False | 28.0 | 224 | 19 | 0.9987 | 0.8747 | 0.124 | 0.8596 | 0.8568 | 7.3041 |
| convnext_tiny__tl_L5_longer +TTA | convnext_tiny | tl_L5_longer | True | 28.0 | 224 | 19 | 0.9987 | 0.8778 | 0.1209 | 0.8696 | 0.8688 | 7.3041 |
| resnet50.tv2_in1k__tl_L6_hires_strongaug | resnet50.tv2_in1k | tl_L6_hires_strongaug | False | 23.9 | 448 | 40 | 0.9995 | 0.8458 | 0.1537 | 0.8297 | 0.824 | 15.7054 |
| resnet50.tv2_in1k__tl_L6_hires_strongaug +TTA | resnet50.tv2_in1k | tl_L6_hires_strongaug | True | 23.9 | 448 | 40 | 0.9995 | 0.8458 | 0.1537 | 0.8331 | 0.8278 | 15.7054 |
| convnext_tiny__tl_L6_hires | convnext_tiny | tl_L6_hires | False | 28.0 | 448 | 38 | 0.9987 | 0.8944 | 0.1043 | 0.8804 | 0.8757 | 16.9391 |
| convnext_tiny__tl_L6_hires +TTA | convnext_tiny | tl_L6_hires | True | 28.0 | 448 | 38 | 0.9987 | 0.8944 | 0.1043 | 0.8846 | 0.8799 | 16.9391 |
| resnet50.tv2_in1k__tl_L6_hires_strongaug__full | resnet50.tv2_in1k | tl_L6_hires_strongaug | False | 23.9 | 448 | 40 | 0.9992 | nan | nan | 0.8646 | 0.8612 | 18.0073 |
| resnet50.tv2_in1k__tl_L6_hires_strongaug__full +TTA | resnet50.tv2_in1k | tl_L6_hires_strongaug | True | 23.9 | 448 | 40 | 0.9992 | nan | nan | 0.8663 | 0.8631 | 18.0073 |
| convnext_tiny__tl_L6_hires__full | convnext_tiny | tl_L6_hires | False | 28.0 | 448 | 40 | 0.999 | nan | nan | 0.902 | 0.8994 | 19.6751 |
| convnext_tiny__tl_L6_hires__full +TTA | convnext_tiny | tl_L6_hires | True | 28.0 | 448 | 40 | 0.999 | nan | nan | 0.902 | 0.8985 | 19.6751 |
| resnet50.tv2_in1k__tl_L8_sgd_1e-3 | resnet50.tv2_in1k | tl_L8_sgd_1e-3 | False | 23.9 | 448 | 39 | 0.9821 | 0.8043 | 0.1778 | 0.789 | 0.7802 | 15.3828 |
| resnet50.tv2_in1k__tl_L8_sgd_1e-3 +TTA | resnet50.tv2_in1k | tl_L8_sgd_1e-3 | True | 23.9 | 448 | 39 | 0.9821 | 0.8168 | 0.1654 | 0.7857 | 0.7756 | 15.3828 |
| resnet50.tv2_in1k__tl_L8_sgd_3e-3 | resnet50.tv2_in1k | tl_L8_sgd_3e-3 | False | 23.9 | 448 | 39 | 0.999 | 0.8302 | 0.1687 | 0.814 | 0.8069 | 15.2694 |
| resnet50.tv2_in1k__tl_L8_sgd_3e-3 +TTA | resnet50.tv2_in1k | tl_L8_sgd_3e-3 | True | 23.9 | 448 | 39 | 0.999 | 0.8323 | 0.1667 | 0.8189 | 0.812 | 15.2694 |
| resnet50.tv2_in1k__tl_L8_sgd_1e-2 | resnet50.tv2_in1k | tl_L8_sgd_1e-2 | False | 23.9 | 448 | 38 | 0.9992 | 0.8447 | 0.1545 | 0.8272 | 0.8224 | 15.245 |
| resnet50.tv2_in1k__tl_L8_sgd_1e-2 +TTA | resnet50.tv2_in1k | tl_L8_sgd_1e-2 | True | 23.9 | 448 | 38 | 0.9992 | 0.8437 | 0.1555 | 0.8314 | 0.8266 | 15.245 |
| convnext_tiny__tl_L8_llrd_080 | convnext_tiny | tl_L8_llrd_080 | False | 28.0 | 448 | 35 | 0.9984 | 0.8903 | 0.1082 | 0.8704 | 0.8644 | 17.0846 |
| convnext_tiny__tl_L8_llrd_080 +TTA | convnext_tiny | tl_L8_llrd_080 | True | 28.0 | 448 | 35 | 0.9984 | 0.8934 | 0.1051 | 0.8796 | 0.8743 | 17.0846 |
| convnext_tiny__tl_L8_llrd_065 | convnext_tiny | tl_L8_llrd_065 | False | 28.0 | 448 | 38 | 0.9984 | 0.8975 | 0.1009 | 0.8779 | 0.8729 | 17.0516 |
| convnext_tiny__tl_L8_llrd_065 +TTA | convnext_tiny | tl_L8_llrd_065 | True | 28.0 | 448 | 38 | 0.9984 | 0.8923 | 0.1061 | 0.8779 | 0.8729 | 17.0516 |
