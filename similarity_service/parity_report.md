# Service-Notebook Parity Report

Generated: 2026-05-21 12:06:45
Service: http://localhost:8000
Configuration tested: `outcomes_raw` (Notebook 1 config 1, LaBSE)

## Summary

- Rows checked: 6
- Rows within tolerance: 6
- Tolerance: 1e-04
- Mean absolute difference: 5.96e-08
- Max absolute difference: 1.19e-07
- Overall status: **PASS**

## Per-row results

| Row | Label | Service cosine | Notebook cosine | Abs. diff | Status | Latency (ms) |
|----:|------:|---------------:|----------------:|----------:|:-------|-------------:|
| 0 | 1 | 0.921735 | 0.921735 | 0.00e+00 | PASS | 2212.7 |
| 1 | 0 | 0.776866 | 0.776866 | 0.00e+00 | PASS | 2219.7 |
| 5 | 1 | 0.949368 | 0.949368 | 1.19e-07 | PASS | 2125.5 |
| 2 | 0 | 0.728016 | 0.728016 | 1.19e-07 | PASS | 2146.4 |
| 6 | 1 | 0.914386 | 0.914386 | 0.00e+00 | PASS | 2146.1 |
| 3 | 0 | 0.677207 | 0.677207 | 1.19e-07 | PASS | 2129.5 |

## Interpretation

The service-side cosine similarity matches the stored Notebook 2 embedding cosine to within the tolerance for every row checked. The service therefore reproduces the LaBSE config 1 evaluation path of the thesis pipeline. The F1 numbers reported in the thesis for this configuration apply to the deployed service.

## Sample inputs

### Row 0 (label 1)

Finnish outcomes:

> Opiskelija ymmärtää kyberturvallisuuden ja tietoturvallisuuden hallintajärjestelmien merkityksen, periaatteet ja vaatimu...

English outcomes:

> The student understands the importance, principles and requirements of cyber security and information security managemen...

Service tokens: FI=360, EN=277. Truncated FI=False, EN=False.

### Row 1 (label 0)

Finnish outcomes:

> Opintojakson suoritettuasi ymmärrät ohjelmistojen laadunvarmistuksen ja testauksen merkityksen osana ohjelmistotuotannon...

English outcomes:

> After completing the course, you understand the importance of project management from the perspective of the software in...

Service tokens: FI=355, EN=410. Truncated FI=False, EN=False.

### Row 5 (label 1)

Finnish outcomes:

> Sinä hallitset keskeisimmät kyberharjoituksen suunnitteluun ja valmisteluun liittyvät osa-alueet: käsitteet, käytetyt to...

English outcomes:

> You master the most essential areas related to the planning and preparation of a cyber security exercise: concepts, used...

Service tokens: FI=183, EN=133. Truncated FI=False, EN=False.

### Row 2 (label 0)

Finnish outcomes:

> Regressiomallien avulla voi ennustaa numeerisia arvoja uusille havainnoille. Tutustut erilaisiin regressiomenetelmiin ja...

English outcomes:

> Purpose: The operation of many applications is based on experimental data. The utilization of the data requires an error...

Service tokens: FI=159, EN=146. Truncated FI=False, EN=False.

### Row 6 (label 1)

Finnish outcomes:

> Opintojakson suoritettuasi ymmärrät mitä tarkoittavat käsitteet ohjelmisto, ohjelmistopalvelu, palveluinfrastruktuuri ja...

English outcomes:

> After completing the course, you understand what the concepts software, software service, service infrastructure and dif...

Service tokens: FI=225, EN=166. Truncated FI=False, EN=False.

### Row 3 (label 0)

Finnish outcomes:

> Opiskelija tuntee ja osaa soveltaa eri tietoturvatekniikoita. Opiskelija tietää ja ymmärtää erilaisten tietoturvateknolo...

English outcomes:

> In the course, you will familiarize yourself with the mathematics that is the basis of 2D and 3D graphics programming: v...

Service tokens: FI=127, EN=163. Truncated FI=False, EN=False.
