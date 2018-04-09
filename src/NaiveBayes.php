<?php

namespace Rubix\Engine;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class NaiveBayes implements Classifier
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

        $strata = $data->stratify();

        foreach ($strata[0] as $class => $samples) {
            $this->weights[$class] = count($samples) / count($data);

            foreach (array_map(null, ...$samples) as $column => $features) {
                if ($this->columnTypes[$column] === self::CATEGORICAL) {
                    $counts = array_count_values($features);

                    foreach ($counts as $label => $count) {
                        $this->stats[$class][$column][$label] = $count / count($features);
                    }
                } else {
                    $mean = Average::mean($features);

                    $stddev = sqrt(array_reduce($features, function ($carry, $feature) use ($mean) {
                        return $carry += ($feature - $mean) ** 2;
                    }, 0.0) / count($features)) + self::EPSILON;

                    $this->stats[$class][$column] = [$mean, $stddev];
                }
            }
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
}
