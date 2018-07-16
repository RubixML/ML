<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\Reports\ResidualAnalysis;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ MPG Estimator using MLP regressor                   ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/mpg.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'cyclinders', 'displacement', 'weight', 'acceleration', 'horsepower', 'year',
]));

$labels = iterator_to_array($reader->fetchColumn('mpg'));

$dataset = new Labeled($samples, $labels);

$hidden = [
    new Dense(50, new LeakyReLU()),
];

$estimator = new Pipeline(new MLPRegressor($hidden, 10, new Adam(0.001),
    1e-4, new MeanSquaredError(), 0.1, 3, 1e-5, 100), [
        new NumericStringConverter(),
        new MissingDataImputer('?'),
        new ZScaleStandardizer(),
    ]);

$report = new AggregateReport([
    new ResidualAnalysis(),
    new PredictionSpeed(),
]);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator->train($training);

var_dump($estimator->progress());

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($testing->head(3)));
