<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\Strategies\PopularityContest;
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

$estimator = new Pipeline(new LogisticRegression(10, new Adam(0.001), 1e-4, 1e-4), [
    new NumericStringConverter(),
    new ZScaleStandardizer(),
]);

$validator = new KFold(10);

$dataset->randomize();

var_dump($validator->test($dummy, $dataset, new MCC()));

echo "\n";

var_dump($validator->test($estimator, $dataset, new MCC()));

var_dump($estimator->proba($dataset->randomize()->head(5)));
