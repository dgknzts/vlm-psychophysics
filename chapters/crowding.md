# Crowding

### What we will do here

1. What crowding is, and Bouma's simple spacing rule for it.
2. The original experiment: Bouma (1970) and his data.
3. Crowding in vision-language models: what we want to find out.

### What is crowding?

An object (target) you can read when its isolated becomes very difficult to discriminate when surrounded by neighboring objects (flankers).  

![](img/crowding_example.svg)

**Figure 1.** Look at the cross. The N on the left is easy to name; the same N on the right, at the same distance, is lost between D and O.

## A surprisingly simple spacing rule: Bouma's window

Flankers hurt when they are closer than about half the eccentricity: Bouma's law. The critical spacing grows in a straight line with distance from fixation.

[demo: bouma]

**Figure 2.** Keep your eyes on the cross and move the slider. With the flankers close, the middle letter is hard to name; past about half the eccentricity (Bouma's window), it becomes easy. Click the image for new letters.

## The original experiment: Bouma (1970)

Here we use Bouma's own data from 1970, refitted with modern psychometric functions by Coates and colleagues (2021).

### Stimuli and the task

![](img/example_trial.svg)

**Figure 3.** An example trial, drawn after Bouma (1970). The observer says the middle letter aloud.

Bouma's observer named the letter out loud. We ask the model the same question, "What was the middle letter?", but it must pick one of the 25 letters, even when guessing (a forced choice).

### Psychometric function fitting

Bouma tested each spacing about 100 times, at distances from 1° to 10.5° from fixation. Accuracy rises with spacing along an S-shaped curve, one curve per distance. The critical spacing is where a curve gets close to the accuracy for a letter on its own.

![560](img/coates2021_fig2a.png)

**Figure 4.** Bouma's data (dots) with fitted psychometric functions, one colour per distance from fixation. The further from fixation, the further right the curve: the letters must be further apart to be read. From Coates et al. (2021).


# Crowding in Vision Language Models

## What we want to find out

A VLM sees the whole image at the same resolution, so it may not crowd at all.

1. **Do VLMs crowd?** First with plain images, then with images degraded the way human peripheral vision degrades them (Texture Tiling Model; Balas et al., 2009).
2. **Do they follow Bouma's rule?** Does the critical spacing grow with eccentricity, at about half?
3. **Can we fit the same psychometric functions?** Model curves next to Bouma's.

## References

- Balas, B., Nakano, L., & Rosenholtz, R. (2009). A summary-statistic representation in peripheral vision explains visual crowding. *Journal of Vision*, 9(12):13, 1–18.
- Bouma, H. (1970). Interaction effects in parafoveal letter recognition. *Nature*, 226, 177–178.
- Coates, D. R., Ludowici, C. J. H., & Chung, S. T. L. (2021). The generality of the critical spacing for crowded optotypes: From Bouma to the 21st century. *Journal of Vision*, 21(11):18, 1–22.
- Deza, A., Jonnalagadda, A., & Eckstein, M. P. (2019). Towards metamerism via foveated style transfer. *ICLR*.
- Pelli, D. G., Palomares, M., & Majaj, N. J. (2004). Crowding is unlike ordinary masking: Distinguishing feature integration from detection. *Journal of Vision*, 4(12), 1136–1169.
