<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Estimator
{
    /**
     * The classifier estimator type.
     *
     * @var int
     */
    public const CLASSIFIER = 1;

    /**
     * The regressor estimator type.
     *
     * @var int
     */
    public const REGRESSOR = 2;

    /**
     * The clusterer estimator type.
     *
     * @var int
     */
    public const CLUSTERER = 3;

    /**
     * The anomaly detector estimator type.
     *
     * @var int
     */
    public const ANOMALY_DETECTOR = 4;

    /**
     * An array of human-readable string representations of the estimator types.
     *
     * @var int
     */
    public const TYPE_STRINGS = [
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
     * @return \Rubix\ML\DataType[]
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
