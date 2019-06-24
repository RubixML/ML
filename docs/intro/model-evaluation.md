# Model Evaluation
Making predictions is not very useful unless the estimator can correctly generalize what it has learned during training to the real world. [Cross Validation](#cross-validation) is a process by which we can test the model for its generalization ability. For the purposes of this introduction, we will use a simple form of cross validation called *Hold Out*. The [Hold Out](#hold-out) validator will take care of splitting the dataset into training and testing sets automatically, such that a portion of the data is *held out* to be used for testing (or *validating*) the model. The reason we do not use *all* of the data for training is because we want to test the Estimator on samples that it has never seen before.

The Hold Out validator requires you to set the ratio of testing to training samples as a constructor parameter. In this case, let's choose to use a factor of 0.2 (20%) of the dataset for testing leaving the rest (80%) for training. Typically, 0.2 is a good default choice however your mileage may vary. The important thing to note here is the trade off between more data for training and more data to produce precise testing results. Once you get the hang of Hold Out, the next step is to consider more advanced cross validation techniques such as [K Fold](#k-fold), [Leave P Out](#leave-p-out), and [Monte Carlo](#monte-carlo) simulations.

To return a score from the Hold Out validator using the Accuracy metric just pass it the untrained estimator instance and a dataset:

```php
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new HoldOut(0.2);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

**Output:**

```sh
float(0.945)
```

## Next Steps
After you've gone through this basic introduction to machine learning, we highly recommend checking out all of the [example projects](https://github.com/RubixML) and reading over the [API Reference](#api-reference) to get a powerful understanding for what you can do with Rubix. If you have a question or need help, feel free to post on our Github page. We'd love to hear from you.