<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Perceptron;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Estimators\DummyClassifier;
use Rubix\Engine\Estimators\Wrappers\Pipeline;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\Transformers\Strategies\PopularityContest;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Detector using Perceptron               ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/banknotes.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$header = [
    'variance', 'skewness', 'curtosis', 'entropy',
];

$samples = iterator_to_array($reader->getRecords($header));

$labels = iterator_to_array($reader->fetchColumn('label'));

$dataset = new Supervised($samples, $labels);

$dummy = new DummyClassifier(new PopularityContest());

$estimator = new Pipeline(new Perceptron(10, 5, new Adam(0.001), 0), [
    new NumericStringConverter(),
]);

$validator = new KFold(new MCC(), 10);

var_dump($validator->score($dummy, $dataset));

echo "\n";

var_dump($validator->score($estimator, $dataset));
