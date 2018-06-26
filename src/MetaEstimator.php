<?php

namespace Rubix\ML;

use Rubix\ML\Clusterers\Clusterer;
use Rubix\ML\Regressors\Regressor;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\AnomalyDetectors\Detector;

interface MetaEstimator extends Classifier, Clusterer, Regressor, Detector
{
    //
}
