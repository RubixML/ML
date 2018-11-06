<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Estimator
{
    const CLASSIFIER = 1;
    const REGRESSOR = 2;
    const CLUSTERER = 3;
    const DETECTOR = 4;
    const EMBEDDER = 5;

    const TYPES = [
        1 => 'Classifier',
        2 => 'Regressor',
        3 => 'Clusterer',
        4 => 'Anomaly Detector',
        5 => 'Embedder',
    ];

    const EPSILON = 1e-8;

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int;

    /**
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array;
}
