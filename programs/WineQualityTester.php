<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\NeuralNet\Layers\Dense;
use Rubix\Engine\Regressors\MLPRegressor;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Metrics\Validation\RSquared;
use Rubix\Engine\CrossValidation\ReportGenerator;
use Rubix\Engine\NeuralNet\ActivationFunctions\PReLU;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\ResidualAnalysis;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Wine Quality Tester using an MLP Regressor          ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/winequality-red.csv')
    ->setDelimiter(';')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH',
    'sulphates', 'alcohol',
]));

$labels = iterator_to_array($reader->fetchColumn('quality'));

$dataset = new Labeled($samples, $labels);

$hidden = [
    new Dense(30, new PReLU()),
    new Dense(30, new PReLU()),
    new Dense(30, new PReLU()),
];

$estimator = new Pipeline(new MLPRegressor($hidden, 50, new Adam(0.001),
    1e-4, new RSquared(), 0.2, 3, 100), [
        new NumericStringConverter(),
    ]);

$report = new ReportGenerator(new ResidualAnalysis(), 0.2);

var_dump($report->generate($estimator, $dataset));

var_dump($estimator->progress());

var_dump($estimator->predict($dataset->randomize()->head(5)));
