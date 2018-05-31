<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Classifiers\DummyClassifier;
use Rubix\Engine\Classifiers\LogisticRegression;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\Transformers\Strategies\PopularityContest;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Detector using Logistic Regression      ║' . "\n";
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

$dataset = new Labeled($samples, $labels);

$dummy = new DummyClassifier(new PopularityContest());

$estimator = new Pipeline(new LogisticRegression(50, 10, new Adam(0.001), 1e-4), [
    new NumericStringConverter(),
]);

$validator = new KFold(new MCC(), 10);

var_dump($validator->score($dummy, $dataset));

echo "\n";

var_dump($validator->score($estimator, $dataset));
