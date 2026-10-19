# P6: requirement → measured evidence

Не заменяет историческую матрицу договора. Нет доказательства — нет повышения статуса.

| Проверка | Статус | Прямой источник |
|---|---|---|
| speed-demo | не достигнут | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 |
| interfaces-demo | подтверждено | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 |
| ai-share-demo | измерено, не доказательство калибровки | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 |
| speed-hf | не достигнут | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 |
| interfaces-hf | подтверждено | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 |
| ai-share-hf | измерено, не доказательство калибровки | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 |
| frozen-public-parity | подтверждено | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 |
| local-suite | подтверждено | reports/acceptance/full-suite.xml |
| ordinary-ci | подтверждено | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39180292854 |
| windows-rc | не достигнут | https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39180292839 |
| full-2M-demo-answer | подтверждено функционально; скорость не достигнута | reports/acceptance/functional-2M-demo.json |
| native-HF-long-answer-without-truncation | не достигнут | reports/acceptance/long-answer-limitation.json |
| robustness | измерено на saved OOF/dev, не новая независимая проверка | reports/acceptance/robustness.json |
| official-test177 | не запускался | protocol.json |
| whole-contract | не принят | protocol.json |

## Native product timing (three repeats)

| mode | role | chars | state | median s | max s | status |
|---|---|---:|---|---:|---:|---|
| demo | document-context | 1000 | cold-process | 0.007070377000005124 | 0.01361029800000324 | достигнут |
| demo | document-context | 1000 | warm-model | 0.0020004269999986946 | 0.0020111529999979894 | достигнут |
| demo | document-context | 50000 | cold-process | 0.04681097200000295 | 0.04691091099999767 | достигнут |
| demo | document-context | 50000 | warm-model | 0.042196130999997195 | 0.042376238000002786 | достигнут |
| demo | document-context | 2000000 | cold-process | 2.3095699399999887 | 2.316568701999998 | достигнут |
| demo | document-context | 2000000 | warm-model | 2.3123810719999938 | 2.3253580020000015 | достигнут |
| demo | answer-text | 1000 | cold-process | 0.013407542999999578 | 0.013634439999989922 | достигнут |
| demo | answer-text | 1000 | warm-model | 0.00858227000000511 | 0.008736450000000673 | достигнут |
| demo | answer-text | 50000 | cold-process | 0.5451400299999989 | 0.5452188559999911 | достигнут |
| demo | answer-text | 50000 | warm-model | 0.5357318509999942 | 0.5742777539999935 | достигнут |
| demo | answer-text | 2000000 | cold-process | None | None | не достигнут |
| demo | answer-text | 2000000 | warm-model | None | None | не достигнут |
| hf | document-context | 1000 | cold-process | 5.759504465000006 | 5.766685546000005 | не достигнут |
| hf | document-context | 1000 | warm-model | 1.4694539609999993 | 1.4819093520000024 | не достигнут |
| hf | document-context | 50000 | cold-process | 8.605325195999995 | 8.725292445999997 | не достигнут |
| hf | document-context | 50000 | warm-model | 4.220389256000004 | 4.262031673000024 | не достигнут |
| hf | document-context | 2000000 | cold-process | 9.524675689999981 | 9.649384985000012 | достигнут |
| hf | document-context | 2000000 | warm-model | 5.344215689000009 | 5.349798615999987 | достигнут |
| hf | answer-text | 1000 | cold-process | 7.681687864999958 | 17.582006344999968 | не достигнут |
| hf | answer-text | 1000 | warm-model | 3.3287632779999967 | 3.3453974079999966 | не достигнут |
| hf | answer-text | 50000 | cold-process | 8.131707099000039 | 8.16342842299997 | не достигнут |
| hf | answer-text | 50000 | warm-model | 3.664374754999983 | 3.680787724999959 | не достигнут |
| hf | answer-text | 2000000 | cold-process | None | None | не достигнут |
| hf | answer-text | 2000000 | warm-model | None | None | не достигнут |
