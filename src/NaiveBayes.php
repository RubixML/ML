<?php

namespace Rubix\Engine;

use MathPHP\Statistics\Average;
use Rubix\Engine\Persisters\Persistable;
use InvalidArgumentException;

class NaiveBayes implements Classifier, Persistable
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
     * The weights of the unique outcomes of the training set.
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
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function train(Dataset $data) : void
    {
        if (!$data instanceof SupervisedDataset) {
            throw new InvalidArgumentException('This estimator requires a supervised dataset.');
        }

        $this->columnTypes = $data->columnTypes();
        $this->stats = $this->weights = [];

        $classes = $data->stratify();

        foreach ($classes[0] as $class => $samples) {
            foreach (array_map(null, ...$samples) as $column => $features) {
                if ($this->columnTypes[$column] === self::CATEGORICAL) {
                    $this->stats[$class][$column] = $this->calculateProbabilities($features);
                } else {
                    $this->stats[$class][$column] = $this->calculateStatistics($features);
                }
            }

            $this->weights[$class] = count($samples) / count($data);
        }
    }

    /**
     * Calculate the probabilities of the sample being a member of all trained
     * classes and chose the highest probaility outcome as the prediction.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $best = ['probability' => -INF, 'outcome' => null];

        foreach ($this->stats as $class => $stats) {
            $probability = $this->weights[$class];

            foreach ($sample as $column => $feature) {
                if ($this->columnTypes[$column] === self::CATEGORICAL) {
                    $probability += $stats[$column][$feature] ?? 0.0;
                } else {
                    list($mean, $stddev) = $stats[$column];

                    $pdf = -0.5 * log(2.0 * M_PI * $stddev ** 2);
                    $pdf -= 0.5 * ($feature - $mean) ** 2 / ($stddev ** 2);

                    $probability += $pdf;
                }
            }

            if ($probability > $best['probability']) {
                $best = [
                    'outcome' => $class,
                    'probability' => $probability,
                ];
            }
        }

        return new Prediction($best['outcome'], [
            'probability' => $best['probability'],
        ]);
    }

    /**
     * Calculate the probabilities of a column of features.
     *
     * @param  array  $values
     * @return array
     */
    protected function calculateProbabilities(array $values) : array
    {
        $counts = array_count_values($values);
        $probabilities = [];

        foreach ($counts as $label => $count) {
            $probabilities[$label] = $count / count($values);
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
