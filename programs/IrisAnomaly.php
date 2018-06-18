<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Metrics\Distance\Euclidean;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\AnomalyDetection\LocalOutlierFactor;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Anomaly Detector using Local Outlier Factor    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
]));

$dataset = new Unlabeled($samples);

$estimator = new Pipeline(new LocalOutlierFactor(10, 20, 0.7, new Euclidean()), [
    new NumericStringConverter(),
    new ZScaleStandardizer(),
]);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator->train($training);

var_dump($estimator->predict($testing));
