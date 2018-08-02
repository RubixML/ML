<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Reports\ResidualBreakdown;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Airfoil Noise using Bootstrapped Regression Tree    ║' . "\n";
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

$estimator = new Pipeline(new BootstrapAggregator(RegressionTree::class, [10, 3, 3], 20, 0.3), [
        new NumericStringConverter(),
    ]);

$report = new AggregateReport([
    new ResidualBreakdown(),
    new PredictionSpeed(),
]);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($testing->head(3)));
