<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\AnomalyDetectors\IsolationTree;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\AnomalyDetectors\IsolationForest;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\OutlierRatio;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\PredictionSpeed;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Anomaly Detector using Isolation Forest        ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')
    ->setDelimiter(';')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH',
    'sulphates', 'alcohol',
]));

$dataset = new Unlabeled($samples);

$estimator = new Pipeline(new IsolationForest(300, 0.2, 0.55), [
    new NumericStringConverter(),
]);

$report = new AggregateReport([
    new OutlierRatio(),
    new PredictionSpeed(),
]);

list($training, $testing) = $dataset->split(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));
