<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Estimator
{
    public const CLASSIFIER = 1;
    public const REGRESSOR = 2;
    public const CLUSTERER = 3;
    public const ANOMALY_DETECTOR = 4;

    public const TYPES = [
        self::CLASSIFIER => 'classifier',
        self::REGRESSOR => 'regressor',
        self::CLUSTERER => 'clusterer',
        self::ANOMALY_DETECTOR => 'anomaly detector',
    ];

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int;

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array;

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array;
}
