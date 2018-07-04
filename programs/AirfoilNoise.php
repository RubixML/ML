<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\PredictionSpeed;
use Rubix\ML\CrossValidation\Reports\ResidualAnalysis;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Airfoil Noise using a Regression Tree               ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/airfoil-noise.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'frequency', 'angle', 'chord_length', 'velocity', 'displacement',
]));

$labels = iterator_to_array($reader->fetchColumn('decibels'));

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new RegressionTree(100, 5, 2), [
        new NumericStringConverter(),
    ]);

$report = new AggregateReport([
    new ResidualAnalysis(),
    new PredictionSpeed(),
]);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($testing->head(3)));
