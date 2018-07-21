<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Other\Generators\Blob;
use Rubix\ML\Other\Generators\Circle;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\Other\Generators\Agglomerate;
use Rubix\ML\Classifiers\KNearestNeighbors;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Circle Classifier using K Nearest Neighbors         ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$generator = new Agglomerate([
    'zero' => new Blob([0, 0], 0.3),
    'one' => new Circle([0, 0], 1.0, 1.0),
    'two' => new Circle([0, 0], 3.0, 1.0),
    'three' => new Circle([0, 0], 5.0, 1.0),
    'four' => new Circle([0, 0], 8.0, 2.0),
], [
    8, 5, 7, 10, 4,
]);

$dataset = $generator->generate(1000);

$estimator = new KNearestNeighbors(3, new Euclidean());

$validator = new KFold(10);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new MulticlassBreakdown(),
    new PredictionSpeed(),
]);

var_dump($validator->test($estimator, $dataset, new MCC()));

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($testing->head(5)));
