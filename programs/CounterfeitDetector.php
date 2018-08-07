<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
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

$estimator = new Pipeline(new LogisticRegression(300, 10, new Adam(0.001), 1e-4, new CrossEntropy(), 1e-4), [
    new NumericStringConverter(),
    new ZScaleStandardizer(),
]);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new MulticlassBreakdown(),
    new PredictionSpeed(),
]);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

var_dump($estimator->steps());

var_dump($report->generate($estimator, $testing));

var_dump($estimator->proba($testing->head(3)));
