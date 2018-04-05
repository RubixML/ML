<?php

namespace Rubix\Engine;

class NaiveBayes implements Classifier
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    const EPSILON = 1e-10;

    /**
     * The data type for each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $types = [
        //
    ];

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
     * @return array
     */
    public function stats() : array
    {
        return $this->stats;
    }

    /**
     * Compute the means and standard deviations of the values per class.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return void
     */
    public function train(SupervisedDataset $data) : void
    {
        $this->types = $data->columnTypes();
        $this->stats = [];

        foreach ($data->stratify()[0] as $class => $samples) {
            foreach (array_map(null, ...$samples) as $column => $values) {
                $n = count($values);

                if ($this->types[$column] === self::CATEGORICAL) {
                    $this->stats[$class][$column] = array_map(function ($value) use ($n) {
                        return $value / $n;
                    }, array_count_values($values)) + self::EPSILON;
                } else {
                    $mean = array_sum($values) / $n;

                    $stddev = sqrt(array_reduce($values, function ($carry, $value) use ($mean) {
                        return $carry += ($value - $mean) ** 2;
                    }, 0) / $n) + self::EPSILON;

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
     * @return array
     */
    public function predict(array $sample) : array
    {
        $best = [
            'outcome' => null,
            'probability' => 0.0,
        ];

        foreach ($this->stats as $class => $columns) {
            $probability = 0.0;

            foreach ($sample as $column => $feature) {
                if ($this->types[$column] === self::CATEGORICAL) {
                    $probability += $columns[$column][$feature] ?? 0.0;
                } else {
                    list($mean, $stddev) = $columns[$column];

                    $probability += (1 / (sqrt(2 * M_PI) * $stddev)) * exp(-(($feature - $mean) ** 2 / (2 * $stddev ** 2)));
                }
            }

            if ($probability > $best['probability']) {
                $best = [
                    'outcome' => $class,
                    'probability' => $probability,
                ];
            }
        }

        return [
            'outcome' => $best['outcome'],
            'certainty' => $best['probability'],
        ];
    }
}
