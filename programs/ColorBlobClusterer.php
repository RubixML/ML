<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Other\Generators\Blob;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Reports\ContingencyTable;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Transformers\QuartileStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Color Clusterer using Blob Generator and Mean Shift ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$generator = new Agglomerate([
    'blue' => new Blob([0, 0, 255], 0.1),
    'green' => new Blob([0, 153, 0], 0.1),
    'lime' => new Blob([128, 255, 0], 0.05),
    'purple' => new Blob([102, 0, 102], 0.05),
    'orange' => new Blob([238, 128, 45], 0.05),
], [
    10, 7, 4, 2, 3,
]);

$dataset = $generator->generate(2000);

$estimator = new Pipeline(new MeanShift(0.9, new Euclidean(), 1e-5, 30), [
    new NumericStringConverter(),
    new QuartileStandardizer(),
]);

$validator = new KFold(10);

$report = new AggregateReport([
    new ContingencyTable(),
    new PredictionSpeed(),
]);

var_dump($validator->test($estimator, $dataset, new VMeasure()));

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator->train($training);

var_dump($estimator->progress());

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($testing->head(3)));
