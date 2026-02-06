# Mini-social-media-analysis
Mini Social Media Project Explanation

In this project, an analysis was conducted on how user interactions are shaped on a social media platform. Various data related to each post's likes and comments were collected, and statistical analyses were performed. Machine learning techniques were then applied to predict "popular" content. Here is a summary of the methods used and the results obtained:

Methods and Techniques Used:

Statistical Analysis:

Distribution Metrics: The distribution of likes and comments per post was analyzed. Basic metrics such as Mean, Median, and Mode were calculated to understand the general shape of the data.

Quartiles and IQR (Interquartile Range): The spread of the data was examined, providing insights into outliers and the overall distribution. The standard deviation for likes was approximately 1.33, and for comments, it was 0.93.

Correlation Analysis: The relationship between likes and comments was studied, and it was found that there was no statistically significant difference between the two variables (p-value = 0.899).

Machine Learning:

Data Preparation: Factors such as post content length and the user's previous interaction levels were used to predict popularity. Popularity here is associated with a post receiving more engagement.

Logistic Regression Model: Logistic regression was used to predict popular posts. The accuracy of the model was found to be 75%, with a precision of approximately 77%. This indicates that 77% of the cases predicted as "popular" were correct.

Recall: The model correctly identified 70% of the posts that were actually popular. This suggests that there is a risk of some high-quality content being overlooked by the model.

Outliers and Data Distribution:

For the likes per post, values of 4 and above were considered outliers, and these posts were found to lie at the extreme ends of the data set.

Additionally, it was observed that the distribution of likes had a wider range, while the distribution of comments was more concentrated and symmetric.

Results and Insights:

Popularity Predictions: The model's predictions of "popularity" were closely tied to content length and the user's previous interaction levels. For instance, posts 101 and 104, which had long content and high previous engagement, were predicted to have a higher potential for popularity.

Data Distribution and Outliers: The distribution of likes and comments differs. Like counts exhibit a wider spread, while comment counts are more narrowly distributed and closer to a symmetric shape.

Machine Learning Application: The logistic regression model provided satisfactory classification results in terms of accuracy, but there is a risk of some genuinely popular content being ignored. Therefore, improving the model and adding additional control mechanisms for sensitive content might be necessary.
