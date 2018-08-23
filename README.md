# dataisbeautiful_july_challenge_birdseed

This is my attempt at the Reddit July /r/dataisbeautiful visualization challenge.  See birdseeds_prince.py for my solution.  The other python files are attempts that either did not succeed or did not yield desirable results.

I used Python with pandas, matplotlib, prince (for the correspondence analysis), sklearn (for k-means clustering), and adjustText (to fix the labels on the CA scores/loadings plot).

For this dataset, given the quantitative data, correspondence analysis was a good fit. This method derives a new set of dimensions aligned with the most inertia within the original dataset (formatted as a frequency table). The second image shows that principle components (the new dimensions) 0 and 1 account for >30% of the inertia each, justifying the use of this method.

The first image shows the scores/loadings, which projects the original dimensions/observations on to the new dimensional space. This allows analysis of features/observations that are similar/different. From this there appears to be three clusters:

0: The majority of the birds and those which highly prefer the various sunflower seed types. Shelled peanuts and Safflower seeds also define this cluster, with Chickadees having a greater preference for peanuts.

1: This cluster is comprised of the Finches and Siskins and they prefer Nyjer seed. They don't hate sunflower seeds but don't prefer peanuts or Milo seed.

2: This cluster is Doves/Juncos/Sparrows, which like Milo seed and Millet White/Red, with Juncos and Doves having a stronger dislike of Sunflower seeds.

Given the clustering I passed the new dimensions through k-means clustering algorithm with 3 clusters. Using this I automated the heat map figure generation for the last two figures, which breaks down the seeds or birds in descending cluster size, allowing another look at the ways in which the birds differ.

Future goals: adjust the compartmentalization for the final plot to separate clusters in both the birds AND seeds simultaneously.
