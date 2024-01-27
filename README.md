# Scientific Abstract Moves Dataset


## Overview
The Scientific Abstract Moves Dataset (SAMD) is a manually annotated dataset specifically created for the study of scientific abstract moves. It contains annotated abstracts from conference papers in the NLP domain, including ACL, EMNLP, and NAACL conferences, which are included in the ACL Anthology.

## Dataset Details
- Total number of abstracts: 1,500
- Total number of sentences: 8,934

## Corpus Selection
The corpus for this dataset was carefully selected from abstracts of conference papers in the field of NLP. The abstracts were sourced from reputable conferences such as ACL, EMNLP, and NAACL, all of which are included in the ACL Anthology.

## Annotation Process
The manual annotation process strictly followed the move analysis method, which is the most influential language use analysis method in the ESP (English for Specific Purposes) field. The sentence was taken as the smallest annotation unit. A combination of top-down and bottom-up annotation methods was employed to determine the move label of each sentence. The annotation was conducted by six professional annotators from the School of Foreign Languages. The inter-annotator agreement was measured using the Kappa coefficient, which was calculated to be 0.785, indicating a substantial agreement among the six annotators. This level of agreement is equivalent to about 93.375% of the corpus being independently annotated in a consistent manner.

## Annotation Schema
The dataset is annotated based on an eight-move structure annotation scheme, which includes the following moves:
1. Background: Providing the background and context of the research.
2. GAP: Addressing the research gap or problem statement.
3. Method: Describing the methodology or approach used in the research.
4. Purpose: Stating the objective or research question of the study.
5. Result: Presenting the results or findings of the study.
6. Conclusion: Summarizing the main conclusions of the research.
7. Contribution: Describing the contribution of the research.
8. Implication: Discussing the implications or potential applications of the research.

## Data Format
The dataset is provided in JSON format, where each object represents a sentence from the abstracts. Each object has the following properties:
- "sentence": The original sentence from the abstract.
- "labels": The annotated move labels for the sentence (BAC, GAP, MTD, PUR, RST, CLN, CTN and IMP represent Background, GAP, Method, Purpose, Result, Conclusion, Contribution and Implication respectively).
- "offest": The offset of the sentence in the entire abstract.
- "pro": The offset divided by the number of sentences in the abstract, to extract a proportional sentence position feature.

## Dataset Usage
This dataset can be used for various research tasks related to scientific abstract analysis, move analysis, and discourse analysis. It lays the foundation for subject knowledge mining, and also provides the possibility to give more accurate feedback and evaluation on the move structure and language characteristics of scientific writing, such as: logical continuity, expression integrity, etc.

## License
The Scientific Abstract Moves Dataset (SAMD) is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

You are free to:

- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.

Please note that you must provide appropriate credit to the dataset by citing it properly, indicating any changes made, and providing a link to the license.

For more details about the license, please visit the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) webpage.


## Citation
If you find our work useful or are referencing it in your research, please consider citing our paper. Below is the citation format for our paper:

```bibtex
@inproceedings{lin2023move,
  title={Move Structure Recognition in Scientific Papers with Saliency Attribution},
  author={Lin, Jinkun and Li, Hongzheng and Feng, Chong and Liu, Fang and Shi, Ge and Lei, Lei and Lv, Xing and Wang, Ruojin and Mei, Yangguang and Xu, Lingnan},
  booktitle={China Conference on Knowledge Graph and Semantic Computing},
  pages={246--258},
  year={2023},
  organization={Springer}
}
```

## Acknowledgments
We would like to express our gratitude to the six professional annotators from the School of Foreign Languages for their valuable contributions in manually annotating the dataset.


