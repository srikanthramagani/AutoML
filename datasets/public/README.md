# Public tabular and text datasets

The four raw CSVs in this folder are direct downloads from the UCI Machine Learning Repository. Please retain attribution when reusing them. UCI lists CC BY 4.0 for these records; the Druglib description additionally says research-only, no commercial use, and no redistribution. The raw files are kept here for this local research run and are not republished from this repository.

| Dataset | Rows × columns | Domain, text and structured fields | Target used | Missing values and class balance |
|---|---:|---|---|---|
| Product Classification and Clustering (UCI 837) | 35,311 × 7 | E-commerce; `Product Title`; merchant identifier and cluster/category metadata | `Category Label` (10 classes) | 0 missing; class counts: Fridge Freezers 5,501; Mobile Phones 4,081; Washing Machines 4,044; CPUs 3,862; Fridges 3,584; TVs 3,564; Dishwashers 3,424; Digital Cameras 2,697; Microwaves 2,342; Freezers 2,212 |
| Turkish Crowdfunding Startups (UCI 1025) | 1,628 × 38 | Business; `proje_adi`, `proje_aciklamasi`; campaign platform/type/category, owner and launch-time statistics | `basari_durumu` | 1,165 missing cells, in `bolge` (1), `proje_baslama_tarihi` (611), `proje_bitis_tarihi` (553); başarısız 1,252, başarılı 376. Post-outcome money, outcome-ratio, backer, comment, update and end-date fields are excluded from predictors to reduce target leakage. |
| Recipe Reviews and User Feedback (UCI 911) | 18,182 × 15 | Food/customer feedback; `text`; recipe ranking/name, user reputation, timestamp, replies, votes | `stars`, excluding 0 (“no rating” per UCI) | 2 missing cells in `text`; after removing unrated rows: 16,486. Stars: 5=13,829; 4=1,655; 3=490; 1=280; 2=232. Identifier fields and `best_score` are excluded. |
| Drug Reviews (Druglib.com, UCI 461) | 4,143 × 9 | Health reviews; benefits, side effects, overall comment; drug, condition, rating/effectiveness fields | `effectiveness` | 135 missing cells: condition 1, benefitsReview 23, sideEffectsReview 98, commentsReview 13. Classes: Highly Effective 1,741; Considerably Effective 1,238; Moderately Effective 572; Ineffective 329; Marginally Effective 263. Rating and sideEffects excluded for the effectiveness target. |

Counts and missingness above were calculated from the downloaded files. The suite samples at most 2,500 rows per dataset using fixed seed 2026 for the default resource budget; it records this sampling in `experiments/multimodal_results/config.json`. The exact samples are regenerated from the raw data and config, not rewritten over the source files.

## Sources and citations

- Akritidis, L. (2020). [Product Classification and Clustering](https://archive.ics.uci.edu/dataset/837/product%2Bclassification). UCI; DOI 10.24432/C5M91Z; CC BY 4.0.
- Kılınç, M. & Aydın, C. (2023). [Turkish Crowdfunding Startups](https://archive.ics.uci.edu/dataset/1025/turkish%2Bcrowdfunding%2Bstartups). UCI; CC BY 4.0.
- Ali, A., Matuszewski, S. & Czupyt, J. (2023). [Recipe Reviews and User Feedback](https://archive.ics.uci.edu/dataset/911/recipe%2Breviews%2Band%2Buser%2Bfeedback). UCI; DOI 10.24432/C5FG95; CC BY 4.0.
- Kallumadi, S. & Gräßer, F. (2018). [Drug Reviews (Druglib.com)](https://archive.ics.uci.edu/dataset/461/drug%2Breview%2Bdataset%2Bdruglib). UCI; DOI 10.24432/C55G6J; CC BY 4.0, with the use restrictions stated above.
