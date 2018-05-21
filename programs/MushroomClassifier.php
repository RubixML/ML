<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\NeuralNet\Layers\Dense;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Transformers\OneHotEncoder;
use Rubix\Engine\Estimators\Wrappers\Pipeline;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Estimators\MultiLayerPerceptron;
use Rubix\Engine\NeuralNet\ActivationFunctions\PReLU;
use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\Engine\CrossValidation\Reports\AggregateReport;
use Rubix\Engine\CrossValidation\Reports\ConfusionMatrix;
use Rubix\Engine\CrossValidation\Reports\ClassificationReport;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using Multi Layer Perceptron    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$header = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
    'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population',
    'habitat',
];

$samples = iterator_to_array($reader->getRecords($header));

$labels = iterator_to_array($reader->fetchColumn('class'));

$dataset = new Supervised($samples, $labels);

$estimator = new Pipeline(new MultiLayerPerceptron([
    new Dense(10, new PReLU()),
    new Dense(10, new PReLU()),
    new Dense(10, new PReLU()),
], 5, new Adam(0.001), 1e-4, 0.999, new MCC(), 0.2, 3, 30), [
    new OneHotEncoder(),
]);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new ClassificationReport(),
]);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

$predictions = $estimator->predict($testing);

var_dump($report->generate($predictions, $testing->labels()));
