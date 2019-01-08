## A1. General

- Competition Name: PLAsTiCC Astronomical Classification
- Team Name: Major Tom
- Private Leaderboard Score: 0.70016
- Private Leaderboard Place: 3rd
- Name: Taiga Nomi (nyanp)
- Location: Osaka, Japan
- Email: Noumi.Taiga@gmail.com

## A2. Background on me and the team

### What is your academic/professional background?

I work as a R&D engineer in factory automation industry for 8 years.
As an OSS developer, I created and maintained a small deep learning library,
 [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) for 6 years.

### Did you have any prior experience that helped you succeed in this competition?

I majored aerospace engineering and joined a project of developing astrometry satellite.
It may helped me reading articles / papers regarding photometric classification and LSST project.

### What made you decide to enter this competition?

I wanted to learn how to handle time-series data through this competition.

### How much time did you spend on the competition?

About 100 hours in total. 60% spent for feature engineering, 15% for survey, 25% for others.


### If part of a team, how did you decide to team up?

Mamas sent me a message through Twitter.

### If you competed as part of a team, who did what?
We all had different models with different approaches:

- mamas: CatBoost with a lot of hand-crafted features
- yuval: 1D CNN with on-the-fly augmentation
- me: LightGBM with sncosmo features and pseudo-label

## A3. Summary

I splitted data into galactic/extragalactic and trained 2 LightGBM models on them with ~200 features (with silightly different feature sets).
I performed light curve fittng using sncosmo with various source types (salt-2, nugent, snana…),
and use these parameters as a feature of my extragalactic model. These features gave me significant boost on my score.

To handle a difference between train and test set, pseudo-labelling on class90 is used for extragalactic model.

## A4. Features Selection / Engineering
Below are the variable importance plot of my models.

- TBD: importance plot, short explanation of features

You can see that light curve fitting features is important in my model.
While previous research shows salt-2 parameters are effective for supernova classification,
my experiment shows that mixing various type of sources gives us another boost in the PLAsTiCC dataset.


### How did you select features?

### Did you make any important feature transformations?

### Did you find any interesting interactions between features?

### Did you use external data? (if permitted)
I indirectly used existing template model and LSST's passband model through sncosmo's API.

## A5. Training Method(s)

- 10-Fold CV
- Pseudo-Label

## A6. Execution Time

- How long does it take to train your model?
- How long does it take to generate predictions using your model?
- How long does it take to train the simplified model (referenced in section A6)?
- How long does it take to generate predictions from the simplified model?

## A7. Results of the model

## A8. References
