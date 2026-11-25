| Model | Run (features) | Seeds | Val macro-F1 | Test accuracy | Test macro-F1 | Head params |
|---|---|---|---|---|---|---|
| Unimodal A (text) | `text_roberta_mean` **(selected)** | 3 | 82.66 ± 0.32 | 83.65 ± 0.24 | 82.04 ± 0.31 | 407,576 |
| Unimodal A (text) | `text_roberta_cls` | 3 | 82.41 ± 0.26 | 83.48 ± 0.32 | 81.62 ± 0.24 | 407,576 |
| Unimodal B (image) | `image_vit` **(selected)** | 3 | 46.07 ± 0.17 | 48.61 ± 0.33 | 46.26 ± 0.21 | 407,576 |
| Simple fusion (concat) | `concat_roberta_mean_vit` **(selected)** | 3 | 81.06 ± 0.23 | 82.45 ± 0.25 | 80.79 ± 0.23 | 802,328 |

Values in %, mean ± std over seeds. Feature choice per model is made on validation macro-F1 only.
