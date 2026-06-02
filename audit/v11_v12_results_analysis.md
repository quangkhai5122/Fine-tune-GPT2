# V11/V12 Results Analysis

## Summary
- v11: overlap=4.696, valid=4.329, selected=epoch_07
- v12: overlap=5.374, valid=6.194, selected=epoch_06
  - model-only valid=4.489

## V12 Gate Simulations: valid
- current_all_retrievable: score=6.194, raw=6194, exact10=606
- retrieval_rephrased_ansaug_only: score=6.569, raw=6569, exact10=641
- retrieval_no_foobar: score=6.469, raw=6469, exact10=632
- retrieval_rephrased_only: score=5.170, raw=5170, exact10=492
- retrieval_rephrased_ansaug_plus_math_sv: score=6.577, raw=6577, exact10=642

## V12 Gate Simulations: overlap_valid
- current_all_retrievable: score=5.374, raw=5374, exact10=524
- retrieval_rephrased_ansaug_only: score=6.048, raw=6048, exact10=589
- retrieval_no_foobar: score=5.800, raw=5800, exact10=567
- retrieval_rephrased_only: score=5.051, raw=5051, exact10=480
- retrieval_rephrased_ansaug_plus_math_sv: score=5.996, raw=5996, exact10=585

## V12 Hybrid Delta By Type: valid
- GSM_AnsAug: model_raw=854, hybrid_raw=1513, delta=659, changed=115/209
- GSM_FOBAR: model_raw=295, hybrid_raw=99, delta=-196, changed=108/122
- GSM_Rephrased: model_raw=1501, hybrid_raw=1872, delta=371, changed=51/197
- GSM_SV: model_raw=311, hybrid_raw=203, delta=-108, changed=75/97
- MATH_AnsAug: model_raw=539, hybrid_raw=1279, delta=740, changed=93/173
- MATH_FOBAR: model_raw=178, hybrid_raw=99, delta=-79, changed=31/45
- MATH_Rephrased: model_raw=676, hybrid_raw=986, delta=310, changed=37/116
- MATH_SV: model_raw=135, hybrid_raw=143, delta=8, changed=25/41

## V12 Hybrid Delta By Type: overlap_valid
- GSM_AnsAug: model_raw=416, hybrid_raw=922, delta=506, changed=83/125
- GSM_FOBAR: model_raw=340, hybrid_raw=70, delta=-270, changed=114/125
- GSM_Rephrased: model_raw=989, hybrid_raw=1211, delta=222, changed=28/125
- GSM_SV: model_raw=432, hybrid_raw=236, delta=-196, changed=99/125
- MATH_AnsAug: model_raw=582, hybrid_raw=1073, delta=491, changed=69/125
- MATH_FOBAR: model_raw=463, hybrid_raw=307, delta=-156, changed=86/125
- MATH_Rephrased: model_raw=827, hybrid_raw=1220, delta=393, changed=47/125
- MATH_SV: model_raw=387, hybrid_raw=335, delta=-52, changed=83/125