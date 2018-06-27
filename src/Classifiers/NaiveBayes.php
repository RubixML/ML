<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use InvalidArgumentException;

class NaiveBayes implements Multiclass, Probabilistic, Persistable
{
    /**
     * The precomputed means for the continuous feature columns of the training
     * set.
     *
     * @var array
     */
    protected $means = [
        //
    ];

    /**
     * The precomputed standard deviations for the continuous feature columns
     * of the training set.
     *
     * @var array
     */
    protected $stddevs = [
        //
    ];

    /**
     * The precomputed probabilities for the categorical feature columns of the
     * training set.
     *
     * @var array
     */
    protected $probabilities = [
        //
    ];

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The data type for each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

    /**
     * @return array
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * @return array
     */
    public function stddevs() : array
    {
        return $this->stddevs;
    }

    /**
     * @return array
     */
    public function probabilities() : array
    {
        return $this->probabilities;
    }

    /**
     * @return array
     */
    public function weights() : array
    {
        return $this->weights;
    }

    /**
     * Compute the means and standard deviations of the values per class.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->columnTypes = $dataset->columnTypes();
        $this->classes = $dataset->possibleOutcomes();

        $n = $dataset->numRows();

        $this->means = $this->stddevs = $this->probabilities =
            $this->weights = [];

        foreach ($dataset->stratify() as $class => $dataset) {
            foreach (array_map(null, ...$dataset) as $column => $values) {
                if ($this->columnTypes[$column] === self::CATEGORICAL) {
                    $this->probabilities[$class][$column]
                        = $this->calculateProbabilities((array) $values);
                } else {
                    list($mean, $stddev)
                        = $this->calculateStatistics((array) $values);

                    $this->means[$class][$column] = $mean;
                    $this->stddevs[$class][$column] = $stddev;
                }
            }

            $this->weights[$class] = log(count($dataset) / $n);
        }
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $probabilities) {
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($probabilities as $class => $probability) {
                if ($probability > $best['probability']) {
                    $best['probability'] = $probability;
                    $best['outcome'] = $class;
                }
            }

            $predictions[] = $best['outcome'];
        }

        return $predictions;
    }

    /**
     * Calculate the probabilities of the sample being a member of all trained
     * classes and chose the highest probaility outcome as the prediction.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probabilities = [];

        foreach ($dataset as $i => $sample) {
            foreach ($this->classes as $class) {
                $probability = $this->weights[$class];

                foreach ($sample as $column => $feature) {
                    if ($this->columnTypes[$column] === self::CATEGORICAL) {
                        $probability +=
                            $this->probabilities[$class][$column][$feature] ?? 0.0;
                    } else {
                        $mean = $this->means[$class][$column];
                        $stddev = $this->stddevs[$class][$column];

                        $pdf = -0.5 * log(2.0 * M_PI * $stddev);
                        $pdf -= 0.5 * (($feature - $mean) ** 2) / $stddev;

                        $probability += $pdf;
                    }
                }

                $probabilities[$i][$class] = $probability;
            }
        }

        return $probabilities;
    }

    /**
     * Calculate the probabilities of a column of features.
     *
     * @param  array  $values
     * @return array
     */
    protected function calculateProbabilities(array $values) : array
    {
        $n = count($values);

        $probabilities = [];

        foreach (array_count_values($values) as $category => $count) {
            $probabilities[$category] = $count / $n;
        }

        return $probabilities;
    }

    /**
     * Calculate the mean and standard deviation of a column of features.
     *
     * @param  array  $values
     * @return array
     */
    protected function calculateStatistics(array $values) : array
    {
        $mean = Average::mean($values);

        $stddev = sqrt(array_reduce($values, function ($carry, $value) use ($mean) {
            return $carry += ($value - $mean) ** 2;
        }, 0.0) / count($values)) + self::EPSILON;

        return [$mean, $stddev];
    }
}
