<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Supervised;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use MathPHP\Statistics\Average;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class NaiveBayes implements Supervised, Multiclass, Probabilistic, Persistable
{
    /**
     * The precomputed probabilities for categorical data and means and standard
     * deviations for continuous data per outcome.
     *
     * @var array
     */
    protected $stats = [
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
    public function stats() : array
    {
        return $this->stats;
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
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        $this->columnTypes = $dataset->columnTypes();

        $this->stats = $this->weights = [];

        foreach ($dataset->stratify() as $class => $samples) {
            foreach (array_map(null, ...$samples) as $column => $values) {
                if ($this->columnTypes[$column] === self::CATEGORICAL) {
                    $this->stats[$class][$column]
                        = $this->calculateProbabilities((array) $values);
                } else {
                    $this->stats[$class][$column]
                        = $this->calculateStatistics((array) $values);
                }
            }

            $this->weights[$class] = count($samples) / $dataset->numRows();
        }
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($this->proba($samples) as $probabilities) {
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
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $probabilities = [];

        foreach ($samples as $i => $sample) {
            foreach ($this->stats as $class => $stats) {
                $probability = $this->weights[$class];

                foreach ($sample as $column => $feature) {
                    if ($this->columnTypes[$column] === self::CATEGORICAL) {
                        $probability += $stats[$column][$feature] ?? 0.0;
                    } else {
                        list($mean, $stddev) = $stats[$column];

                        $pdf = -0.5 * log(2.0 * M_PI * $stddev);
                        $pdf -= 0.5 * ($feature - $mean) ** 2 / $stddev;

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
